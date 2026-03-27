"""
OPC Predictor Functions - ZEN Chiller Digital Twin
====================================================
Custom predictor functions for ZEN Chiller CH-01 and CH-02 (Trane UC800).

Covers all 119 BMS tag types across both chillers:
  - EP_Calculated  : derived from EnergyPlus outputs (flow, status flags)
  - ML_Model       : physics-based stand-ins until trained models are available
  - Logic          : state machine / fault logic
  - TBD            : stub returns pending further analysis

Each function MUST have this signature:
    def predict_<name>(config, sensors_df, context) -> value

Args:
    config      : ConfigParser object (full config.ini)
    sensors_df  : DataFrame with current EP sensor values. Columns include:
                    SensorName      - EnergyPlus variable name (title case)
                    SensorInstance  - EnergyPlus object name (e.g. 'Chiller 1')
                    current_val     - Current value from EP
                    opc_tag_name    - OPC tag name if configured
    context     : Dict persisting for the life of the OPC server.
                  Use for: cached ML models, running counters, last-known state.

Returns:
    Numeric, bool, or string value matching the tag's declared data_type.

IDF Chiller Parameters (from in.idf):
    Reference capacity:  1,406,741 W  (400 tons) per chiller
    Reference COP:       6.09
    Condenser flow:      0.0047 m³/s design
    Min PLR:             0.10
    Min unloading ratio: 0.20
    Voltage (assumed):   480V, 3-phase, 60 Hz

Author: ZEN Digital Twin Development
Date: February 2026
"""

import os
import math
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - ZEN Chiller Physical Parameters
# =============================================================================

# From IDF: Chiller:Electric:EIR reference capacity (W)
CHILLER_REF_CAPACITY_W = 1_406_741.1368
CHILLER_REF_CAPACITY_TONS = CHILLER_REF_CAPACITY_W / 3516.853

# From IDF: reference COP
CHILLER_REF_COP = 6.09

# Electrical assumptions (typical Trane centrifugal in this size class)
SUPPLY_VOLTAGE_V = 480.0       # 3-phase supply voltage
SUPPLY_FREQ_HZ = 60.0
SQRT3 = math.sqrt(3.0)

# Full load amps - estimated from design power at full capacity
# P_elec_design = Capacity / COP = 1,406,741 / 6.09 ≈ 231,000 W
# I_FLA = P / (sqrt(3) * V * PF) at PF=0.93 ≈ 300 A
CHILLER_FLA_A = 300.0

# Differential pressure sizing coefficients
# At design evap flow for 400-ton chiller (≈ 400 GPM), ΔP ≈ 15 PSI
# k = ΔP / flow² → at 400 GPM: k = 15 / 400² = 9.375e-5 PSI/GPM²
EVAP_DP_COEFF = 9.375e-5   # PSI / GPM²  (calibrate with real data)

# At design cond flow (0.0047 m³/s = 74.5 GPM), ΔP ≈ 10 PSI
# k = 10 / 74.5² ≈ 1.8e-3 PSI/GPM²
COND_DP_COEFF = 1.8e-3     # PSI / GPM²  (calibrate with real data)

# Flow switch thresholds
EVAP_FLOW_SW_THRESHOLD_KGS = 0.5   # kg/s  (~8 GPM) - below this = no flow
COND_FLOW_SW_THRESHOLD_KGS = 0.5   # kg/s

# Oil system typical values (Trane centrifugal, R-134a class)
OIL_TANK_PRESS_BASE_KPA = 270.0    # kPa baseline when running
OIL_PUMP_DELTA_KPA = 170.0         # pump discharge above tank
OIL_DIFF_PRESS_KPA = 140.0         # nominal oil differential
OIL_TANK_TEMP_IDLE_C = 40.0        # °C at idle
OIL_TANK_TEMP_FULL_C = 55.0        # °C at full load

# Refrigerant saturation curve coefficients for R-134a approximation
# P_sat (kPa) ≈ exp(A - B/(T+C))  where T is in °C
# Fitted to ASHRAE R-134a data over -10°C to 50°C range
R134A_A = 14.3014
R134A_B = 2435.8
R134A_C = 234.0

# Bearing and winding temperature constants (°C)
BEARING_TEMP_AMBIENT_C = 35.0     # Mechanical room ambient
BEARING_TEMP_FULL_LOAD_C = 70.0   # Max bearing temp at full load
WINDING_TEMP_AMBIENT_C = 40.0
WINDING_TEMP_FULL_LOAD_C = 110.0  # Motor winding at full load

# AFD (VFD) temperature constants
AFD_TRANSISTOR_TEMP_IDLE_C = 45.0
AFD_TRANSISTOR_TEMP_FULL_C = 85.0
AFD_BASE_TEMP_IDLE_C = 40.0
AFD_BASE_TEMP_FULL_C = 75.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sensor_value(
    sensors_df: pd.DataFrame,
    sensor_name: str,
    sensor_instance: str = '*',
    default: float = 0.0
) -> float:
    """
    Safely retrieve a sensor value by EnergyPlus sensor name and instance.

    Args:
        sensors_df:      Sensor DataFrame from Digital Twin
        sensor_name:     EnergyPlus output variable name (case-insensitive)
        sensor_instance: EnergyPlus object name, or '*' for first match
        default:         Return value if sensor not found or NaN

    Returns:
        float sensor value, or default
    """
    try:
        if 'SensorName' not in sensors_df.columns:
            logger.error("sensors_df missing 'SensorName' column")
            return default

        mask = sensors_df['SensorName'].str.lower() == sensor_name.lower()

        if sensor_instance != '*' and 'SensorInstance' in sensors_df.columns:
            mask = mask & (
                sensors_df['SensorInstance'].str.lower() == sensor_instance.lower()
            )

        if mask.any():
            value = sensors_df.loc[mask, 'current_val'].iloc[0]
            if pd.isna(value):
                return default
            return float(value)
        else:
            logger.debug(
                f"Sensor '{sensor_name}' / '{sensor_instance}' not found, "
                f"using default {default}"
            )
            return default

    except Exception as e:
        logger.error(f"Error retrieving sensor '{sensor_name}': {e}")
        return default


def get_plr(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return chiller part load ratio for given EP instance ('Chiller 1' etc)."""
    return get_sensor_value(sensors_df, 'Chiller Part Load Ratio', instance, 0.0)


def get_evap_flow_kgs(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return evaporator mass flow rate in kg/s."""
    return get_sensor_value(
        sensors_df, 'Chiller Evaporator Mass Flow Rate', instance, 0.0
    )


def get_cond_flow_kgs(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return condenser mass flow rate in kg/s."""
    return get_sensor_value(
        sensors_df, 'Chiller Condenser Mass Flow Rate', instance, 0.0
    )


def get_power_w(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return chiller electricity rate in watts."""
    return get_sensor_value(sensors_df, 'Chiller Electricity Rate', instance, 0.0)


def get_cooling_rate_w(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return chiller evaporator cooling rate in watts."""
    return get_sensor_value(
        sensors_df, 'Chiller Evaporator Cooling Rate', instance, 0.0
    )


def get_evap_leave_temp_c(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return evaporator leaving (outlet) water temperature in °C."""
    return get_sensor_value(
        sensors_df, 'Chiller Evaporator Outlet Temperature', instance, 7.0
    )


def get_cond_leave_temp_c(sensors_df: pd.DataFrame, instance: str) -> float:
    """Return condenser leaving (outlet) water temperature in °C."""
    return get_sensor_value(
        sensors_df, 'Chiller Condenser Outlet Temperature', instance, 29.0
    )


def r134a_sat_pressure_kpa(temp_c: float) -> float:
    """
    Approximate R-134a saturation pressure at given temperature.

    Uses Antoine-style equation fitted to ASHRAE R-134a data.
    Valid range approximately -10°C to 55°C.

    Args:
        temp_c: Temperature in °C

    Returns:
        Saturation pressure in kPa
    """
    try:
        ln_p = R134A_A - R134A_B / (temp_c + R134A_C)
        return max(50.0, math.exp(ln_p))
    except Exception:
        return 0.0


def line_current_from_power(power_w: float, plr: float) -> float:
    """
    Estimate line current from power consumption.

    Power factor varies with load (typical VFD centrifugal chiller):
      Low load  (PLR=0.1): PF ≈ 0.85
      Full load (PLR=1.0): PF ≈ 0.95

    I = P / (sqrt(3) * V * PF)

    Args:
        power_w: Electrical power in watts
        plr:     Part load ratio (0-1)

    Returns:
        Line current in amps
    """
    if power_w <= 0.0:
        return 0.0
    power_factor = 0.85 + (max(0.0, min(1.0, plr)) * 0.10)
    return power_w / (SQRT3 * SUPPLY_VOLTAGE_V * power_factor)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numerator by denominator, returning default if denominator near zero."""
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def increment_context_counter(context: Dict, key: str, condition: bool) -> int:
    """
    Increment a counter in context if condition is True.
    Returns current counter value.
    """
    if key not in context:
        context[key] = 0
    if condition:
        context[key] += 1
    return context[key]


# =============================================================================
# SHARED CONSTANT PREDICTORS
# (Used directly by name in opc_variables.csv for both chillers)
# =============================================================================

def predict_constant_zero(config: Any, sensors_df: pd.DataFrame,
                          context: Dict) -> float:
    """Return 0. Used for stubs, disabled flags, and TBD tags."""
    return 0.0


def predict_constant_one(config: Any, sensors_df: pd.DataFrame,
                         context: Dict) -> float:
    """Return 1. Used for always-enabled flags (CH_Auto, Control_Mode=Remote)."""
    return 1.0


def predict_constant_100(config: Any, sensors_df: pd.DataFrame,
                         context: Dict) -> float:
    """Return 100. Used for Current_Limit_SP (no limit imposed = 100% FLA)."""
    return 100.0


# =============================================================================
# CHILLER 1  (EP instance: 'Chiller 1')
# =============================================================================

_CH1 = 'Chiller 1'
_CH2 = 'Chiller 2'


# --- Status flags ---

def predict_ch_01_running(config, sensors_df, context):
    """CH-01 running: True when PLR > 0."""
    return bool(get_plr(sensors_df, _CH1) > 0.0)


def predict_ch_01_comp_running(config, sensors_df, context):
    """CH-01 compressor running (same logic as Running)."""
    return bool(get_plr(sensors_df, _CH1) > 0.0)


def predict_ch_01_evap_pump(config, sensors_df, context):
    """CH-01 evaporator pump on: True when evap flow exceeds threshold."""
    return bool(get_evap_flow_kgs(sensors_df, _CH1) > EVAP_FLOW_SW_THRESHOLD_KGS)


def predict_ch_01_cond_pump(config, sensors_df, context):
    """CH-01 condenser pump on: True when condenser flow exceeds threshold."""
    return bool(get_cond_flow_kgs(sensors_df, _CH1) > COND_FLOW_SW_THRESHOLD_KGS)


def predict_ch_01_evap_flow_sw(config, sensors_df, context):
    """CH-01 evaporator flow switch: True when flow > threshold."""
    return bool(get_evap_flow_kgs(sensors_df, _CH1) > EVAP_FLOW_SW_THRESHOLD_KGS)


def predict_ch_01_cond_flow_sw(config, sensors_df, context):
    """CH-01 condenser flow switch: True when flow > threshold."""
    return bool(get_cond_flow_kgs(sensors_df, _CH1) > COND_FLOW_SW_THRESHOLD_KGS)


def predict_ch_01_maximum_capacity_relay(config, sensors_df, context):
    """CH-01 maximum capacity relay: True when PLR >= 0.95."""
    return bool(get_plr(sensors_df, _CH1) >= 0.95)


# --- Temperature tags (°C → °F conversion for historian/SCADA) ---
# EP outputs all temperatures in Celsius; the BMS historian and iFix SCADA
# clients expect Fahrenheit.  These are ep_type=predictor in opc_variables.csv.

def predict_ch_01_evap_leave_temp_f(config, sensors_df, context):
    """
    CH-01 evaporator leaving (chilled) water temperature (°F).

    Source: EP 'Chiller Evaporator Outlet Temperature' [°C] → converted to °F.
    Design setpoint: ~44°F (6.7°C).
    """
    c = get_sensor_value(
        sensors_df, 'Chiller Evaporator Outlet Temperature', _CH1, 7.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_01_evap_enter_temp_f(config, sensors_df, context):
    """
    CH-01 evaporator entering (return chilled) water temperature (°F).

    Source: EP 'Chiller Evaporator Inlet Temperature' [°C] → converted to °F.
    Design return: ~54°F (12.2°C).
    """
    c = get_sensor_value(
        sensors_df, 'Chiller Evaporator Inlet Temperature', _CH1, 12.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_01_cond_enter_temp_f(config, sensors_df, context):
    """
    CH-01 condenser entering (cooling tower supply) water temperature (°F).

    Source: EP 'Chiller Condenser Inlet Temperature' [°C] → converted to °F.
    Typical range: 65–85°F depending on wet-bulb.
    """
    c = get_sensor_value(
        sensors_df, 'Chiller Condenser Inlet Temperature', _CH1, 29.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_01_cond_leave_temp_f(config, sensors_df, context):
    """
    CH-01 condenser leaving (cooling tower return) water temperature (°F).

    Source: EP 'Chiller Condenser Outlet Temperature' [°C] → converted to °F.
    Typically 8–10°F above condenser entering temperature.
    """
    c = get_sensor_value(
        sensors_df, 'Chiller Condenser Outlet Temperature', _CH1, 35.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


# --- Flow rates ---

def predict_ch_01_evap_water_flow_gpm(config, sensors_df, context):
    """CH-01 evaporator water flow: kg/s -> GPM."""
    return get_evap_flow_kgs(sensors_df, _CH1) * 15.850323


def predict_ch_01_cond_water_flow_gpm(config, sensors_df, context):
    """CH-01 condenser water flow: kg/s -> GPM."""
    return get_cond_flow_kgs(sensors_df, _CH1) * 15.850323


# --- Power and capacity ---

def predict_ch_01_power_consumption_kw(config, sensors_df, context):
    """CH-01 power consumption: W -> kW (float64, 2 decimal places).
    Historian stores this as REAL (kW) — must not be int32.
    """
    return round(get_power_w(sensors_df, _CH1) / 1000.0, 2)


def predict_ch_01_power_kw(config, sensors_df, context):
    """CH-01 power: W -> kW (same as Power_Consumption)."""
    return round(get_power_w(sensors_df, _CH1) / 1000.0)


def predict_ch_01_calc_capacity_tons(config, sensors_df, context):
    """CH-01 calculated capacity: W -> tons of refrigeration."""
    return get_cooling_rate_w(sensors_df, _CH1) / 3516.853


# --- Differential pressures (physics: ΔP = k * flow_gpm²) ---

def predict_ch_01_evap_diff_press(config, sensors_df, context):
    """
    CH-01 evaporator differential pressure (PSI).

    Stand-in physics model: ΔP = k_evap × flow_GPM²
    Coefficient k sized so ΔP ≈ 15 PSI at design flow (~400 GPM).

    Replace with ML model once historical data is available.
    """
    flow_gpm = predict_ch_01_evap_water_flow_gpm(config, sensors_df, context)
    dp = EVAP_DP_COEFF * flow_gpm ** 2
    return round(max(0.0, dp), 2)


def predict_ch_01_cond_diff_press(config, sensors_df, context):
    """
    CH-01 condenser differential pressure (PSI).

    Stand-in physics model: ΔP = k_cond × flow_GPM²
    Coefficient k sized so ΔP ≈ 10 PSI at design cond flow (~74 GPM).

    Replace with ML model once historical data is available.
    """
    flow_gpm = predict_ch_01_cond_water_flow_gpm(config, sensors_df, context)
    dp = COND_DP_COEFF * flow_gpm ** 2
    return round(max(0.0, dp), 2)


# --- Line current and electrical ---

def predict_ch_01_line_current(config, sensors_df, context):
    """
    CH-01 line current (A).

    Calculated from power and load-dependent power factor.
    I = P / (√3 × 480V × PF)
    PF varies linearly: 0.85 at PLR=0.1 to 0.95 at PLR=1.0
    """
    power_w = get_power_w(sensors_df, _CH1)
    plr = get_plr(sensors_df, _CH1)
    return round(line_current_from_power(power_w, plr), 1)


def predict_ch_01_starter_voltage(config, sensors_df, context):
    """
    CH-01 starter phase voltage (V).

    Returns 480V when chiller is running, 0 when off.
    All three phases and average return same nominal value.
    """
    return 480 if get_plr(sensors_df, _CH1) > 0.0 else 0


def predict_ch_01_starter_current_ln(config, sensors_df, context):
    """
    CH-01 starter per-phase current (A).

    For balanced 3-phase, each line current equals the calculated line current.
    Returns integer amps.
    """
    power_w = get_power_w(sensors_df, _CH1)
    plr = get_plr(sensors_df, _CH1)
    return round(line_current_from_power(power_w, plr))


def predict_ch_01_starter_current_pct(config, sensors_df, context):
    """
    CH-01 starter current as % of FLA.

    % FLA = (line current / FLA) × 100
    """
    power_w = get_power_w(sensors_df, _CH1)
    plr = get_plr(sensors_df, _CH1)
    current = line_current_from_power(power_w, plr)
    return round(safe_divide(current, CHILLER_FLA_A) * 100.0, 1)


def predict_ch_01_starter_load_pf(config, sensors_df, context):
    """
    CH-01 power factor.

    Varies linearly with PLR: 0.85 at low load, 0.95 at full load.
    Returns 0.0 when chiller is off.
    """
    plr = get_plr(sensors_df, _CH1)
    if plr <= 0.0:
        return 0.0
    return round(0.85 + (min(1.0, plr) * 0.10), 3)


# --- Refrigerant-side pressures and temperatures ---

def predict_ch_01_evap_refrig_press(config, sensors_df, context):
    """
    CH-01 evaporator refrigerant pressure (kPa).

    Estimated as saturation pressure of R-134a at evaporator leaving water
    temperature minus a 2°C approach (refrigerant cooler than leaving water).
    """
    evap_leave_c = get_evap_leave_temp_c(sensors_df, _CH1)
    refrig_temp_c = evap_leave_c - 2.0
    return round(r134a_sat_pressure_kpa(refrig_temp_c), 1)


def predict_ch_01_cond_refrig_press(config, sensors_df, context):
    """
    CH-01 condenser refrigerant pressure (kPa).

    Estimated as saturation pressure of R-134a at condenser leaving water
    temperature plus a 3°C approach (refrigerant warmer than leaving water).
    """
    cond_leave_c = get_cond_leave_temp_c(sensors_df, _CH1)
    refrig_temp_c = cond_leave_c + 3.0
    return round(r134a_sat_pressure_kpa(refrig_temp_c), 1)


def predict_ch_01_diff_refrig_press(config, sensors_df, context):
    """CH-01 refrigerant differential pressure: cond - evap (kPa)."""
    cond_p = predict_ch_01_cond_refrig_press(config, sensors_df, context)
    evap_p = predict_ch_01_evap_refrig_press(config, sensors_df, context)
    return round(max(0.0, cond_p - evap_p), 1)


def predict_ch_01_evap_sat_rfgt_temp(config, sensors_df, context):
    """
    CH-01 evaporator saturated refrigerant temperature (°C).

    Approximated as evap leaving water temp minus 2°C approach.
    """
    return round(get_evap_leave_temp_c(sensors_df, _CH1) - 2.0, 1)


def predict_ch_01_cond_sat_rfgt_temp(config, sensors_df, context):
    """
    CH-01 condenser saturated refrigerant temperature (°C).

    Approximated as cond leaving water temp plus 3°C approach.
    """
    return round(get_cond_leave_temp_c(sensors_df, _CH1) + 3.0, 1)


def predict_ch_01_comp_rfgt_disch_temp(config, sensors_df, context):
    """
    CH-01 compressor refrigerant discharge temperature (°C).

    Estimated as condenser saturated temp plus superheat.
    Superheat increases with PLR: ~10°C at low load, ~25°C at full load.
    """
    plr = get_plr(sensors_df, _CH1)
    cond_sat = predict_ch_01_cond_sat_rfgt_temp(config, sensors_df, context)
    superheat = 10.0 + (min(1.0, plr) * 15.0)
    return round(cond_sat + superheat if plr > 0.0 else 0.0, 1)


# --- Oil system ---

def predict_ch_01_oil_tank_press(config, sensors_df, context):
    """
    CH-01 oil tank pressure (kPa).

    Approximately constant when running, 0 when off.
    """
    return round(OIL_TANK_PRESS_BASE_KPA if get_plr(sensors_df, _CH1) > 0.0 else 0.0, 1)


def predict_ch_01_oil_pump_disch_press(config, sensors_df, context):
    """
    CH-01 oil pump discharge pressure (kPa).

    Tank pressure plus pump differential.
    """
    tank = predict_ch_01_oil_tank_press(config, sensors_df, context)
    return round(tank + OIL_PUMP_DELTA_KPA if tank > 0.0 else 0.0, 1)


def predict_ch_01_oil_diff_press(config, sensors_df, context):
    """
    CH-01 oil differential pressure (kPa).

    Approximately constant when running.
    """
    return round(OIL_DIFF_PRESS_KPA if get_plr(sensors_df, _CH1) > 0.0 else 0.0, 1)


def predict_ch_01_oil_tank_temp(config, sensors_df, context):
    """
    CH-01 oil tank temperature (°C).

    Scales linearly with PLR between idle and full-load temperatures.
    """
    plr = get_plr(sensors_df, _CH1)
    if plr <= 0.0:
        return round(OIL_TANK_TEMP_IDLE_C, 1)
    return round(
        OIL_TANK_TEMP_IDLE_C + plr * (OIL_TANK_TEMP_FULL_C - OIL_TANK_TEMP_IDLE_C),
        1
    )


# --- Inlet guide vanes ---

def predict_ch_01_igv1_pct_open(config, sensors_df, context):
    """
    CH-01 inlet guide vane 1 position (% open).

    Approximately linear with PLR, with minimum position of 15% when running.
    IGV closes at shutdown.
    """
    plr = get_plr(sensors_df, _CH1)
    if plr <= 0.0:
        return 0.0
    igv = plr * 100.0
    igv = max(15.0, min(100.0, igv))  # min 15% when running
    return round(igv, 1)


def predict_ch_01_igv2_pct_open(config, sensors_df, context):
    """CH-01 inlet guide vane 2 (same as IGV1 for single-stage compressor)."""
    return predict_ch_01_igv1_pct_open(config, sensors_df, context)


# --- Thermal parameters ---

def predict_ch_01_bearing_temp(config, sensors_df, context):
    """
    CH-01 bearing temperature (°C).

    Scales linearly with PLR between ambient and full-load bearing temperature.
    Used for both inboard and outboard bearings.
    """
    plr = get_plr(sensors_df, _CH1)
    temp = BEARING_TEMP_AMBIENT_C + plr * (BEARING_TEMP_FULL_LOAD_C - BEARING_TEMP_AMBIENT_C)
    return round(temp, 1)


def predict_ch_01_motor_winding_temp(config, sensors_df, context):
    """
    CH-01 motor winding temperature (°C).

    Scales with power dissipation (approximately PLR-dependent).
    Used for all three winding temperature sensors.
    """
    plr = get_plr(sensors_df, _CH1)
    temp = WINDING_TEMP_AMBIENT_C + plr * (WINDING_TEMP_FULL_LOAD_C - WINDING_TEMP_AMBIENT_C)
    return round(temp, 1)


# --- AFD (VFD) parameters ---

def predict_ch_01_afd_transistor_temp(config, sensors_df, context):
    """
    CH-01 AFD transistor temperature (°C).

    Scales with current load on the drive.
    """
    plr = get_plr(sensors_df, _CH1)
    temp = AFD_TRANSISTOR_TEMP_IDLE_C + plr * (
        AFD_TRANSISTOR_TEMP_FULL_C - AFD_TRANSISTOR_TEMP_IDLE_C
    )
    return round(temp, 1)


def predict_ch_01_afd_inverter_base_temp(config, sensors_df, context):
    """CH-01 AFD inverter base temperature (°C). Scales with PLR."""
    plr = get_plr(sensors_df, _CH1)
    temp = AFD_BASE_TEMP_IDLE_C + plr * (AFD_BASE_TEMP_FULL_C - AFD_BASE_TEMP_IDLE_C)
    return round(temp, 1)


def predict_ch_01_afd_rectifier_base_temp(config, sensors_df, context):
    """CH-01 AFD rectifier base temperature (°C). Scales with PLR."""
    plr = get_plr(sensors_df, _CH1)
    temp = AFD_BASE_TEMP_IDLE_C + plr * (AFD_BASE_TEMP_FULL_C - AFD_BASE_TEMP_IDLE_C)
    return round(temp, 1)


def predict_ch_01_afd_input_freq(config, sensors_df, context):
    """CH-01 AFD input/output frequency (Hz). Fixed US supply frequency."""
    return SUPPLY_FREQ_HZ


# --- Logic / state machine ---

def predict_ch_01_running_status(config, sensors_df, context):
    """
    CH-01 running status code (5-state Trane UC800 machine).

    States:
        0 = Off / Idle       — PLR == 0 and was already stopped
        1 = Starting         — PLR just crossed 0→>0.05 (transitional)
        2 = Running          — PLR > 0.05 and was already starting or running
        3 = Stopping         — PLR just crossed >0→0 (transitional)
        4 = Needs Service    — injected via fault callback (context key ch_01_fault)

    State 4 is set externally by custom/callback.py fault injection; this
    function will hold state 4 until the fault is cleared in context.
    """
    plr = get_plr(sensors_df, _CH1)
    prev = context.get('ch_01_run_state', 0)

    # Fault injection takes priority — hold until cleared externally
    if context.get('ch_01_fault', False):
        state = 4
    elif plr > 0.05:
        # Chiller is loaded
        if prev == 0:
            state = 1   # Off → Starting (one-timestep transient)
        else:
            state = 2   # Starting → Running, or remain Running
    else:
        # PLR at or below min threshold — chiller is unloaded/off
        if prev in (1, 2):
            state = 3   # Running/Starting → Stopping (one-timestep transient)
        elif prev == 3:
            state = 0   # Stopping → Off/Idle
        else:
            state = 0   # Already off

    context['ch_01_run_state'] = state
    return state


def predict_ch_01_last_diag_code(config, sensors_df, context):
    """
    CH-01 last diagnostic code.

    Returns 0 (no fault) during normal simulation.
    Use fault injection (custom callback) to set non-zero codes for training.
    """
    return context.get('ch_01_diag_code', 0)


# --- Accumulating counters ---

def predict_ch_01_compressor_starts(config, sensors_df, context):
    """
    CH-01 lifetime compressor start count.

    Increments each time PLR transitions from 0 to > 0.
    Uses context to track previous PLR state.
    """
    plr = get_plr(sensors_df, _CH1)
    was_running = context.get('ch_01_was_running', False)
    is_running = plr > 0.0

    if is_running and not was_running:
        context['ch_01_starts'] = context.get('ch_01_starts', 0) + 1

    context['ch_01_was_running'] = is_running
    return context.get('ch_01_starts', 0)


def predict_ch_01_compressor_runtime(config, sensors_df, context):
    """
    CH-01 lifetime compressor runtime (hours).

    Increments by EP timestep duration each time chiller is running.
    Assumes 6 timesteps/hour (10-min EP timestep) - adjust if needed.
    """
    plr = get_plr(sensors_df, _CH1)
    if plr > 0.0:
        # Each predictor call corresponds to one EP timestep
        # Default ZEN model: 6 timesteps/hour = 1/6 hour per step
        timestep_hours = float(
            config.get('DEFAULT', 'TimestepHours', fallback=str(1.0 / 6.0))
        )
        context['ch_01_runtime_h'] = context.get('ch_01_runtime_h', 0.0) + timestep_hours
    return round(context.get('ch_01_runtime_h', 0.0))


# =============================================================================
# CHILLER 2  (EP instance: 'Chiller 2')
# CH-02 functions mirror CH-01 exactly, substituting _CH2 instance name
# and ch_02 context keys.
# =============================================================================

def predict_ch_02_running(config, sensors_df, context):
    """CH-02 running: True when PLR > 0."""
    return bool(get_plr(sensors_df, _CH2) > 0.0)


def predict_ch_02_comp_running(config, sensors_df, context):
    """CH-02 compressor running."""
    return bool(get_plr(sensors_df, _CH2) > 0.0)


def predict_ch_02_evap_pump(config, sensors_df, context):
    """CH-02 evaporator pump on."""
    return bool(get_evap_flow_kgs(sensors_df, _CH2) > EVAP_FLOW_SW_THRESHOLD_KGS)


def predict_ch_02_cond_pump(config, sensors_df, context):
    """CH-02 condenser pump on."""
    return bool(get_cond_flow_kgs(sensors_df, _CH2) > COND_FLOW_SW_THRESHOLD_KGS)


def predict_ch_02_evap_flow_sw(config, sensors_df, context):
    """CH-02 evaporator flow switch."""
    return bool(get_evap_flow_kgs(sensors_df, _CH2) > EVAP_FLOW_SW_THRESHOLD_KGS)


def predict_ch_02_cond_flow_sw(config, sensors_df, context):
    """CH-02 condenser flow switch."""
    return bool(get_cond_flow_kgs(sensors_df, _CH2) > COND_FLOW_SW_THRESHOLD_KGS)


def predict_ch_02_maximum_capacity_relay(config, sensors_df, context):
    """CH-02 maximum capacity relay: True when PLR >= 0.95."""
    return bool(get_plr(sensors_df, _CH2) >= 0.95)


# --- Temperature tags (°C → °F conversion for historian/SCADA) ---

def predict_ch_02_evap_leave_temp_f(config, sensors_df, context):
    """CH-02 evaporator leaving water temperature (°F). EP source → °C converted to °F."""
    c = get_sensor_value(
        sensors_df, 'Chiller Evaporator Outlet Temperature', _CH2, 7.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_02_evap_enter_temp_f(config, sensors_df, context):
    """CH-02 evaporator entering water temperature (°F). EP source → °C converted to °F."""
    c = get_sensor_value(
        sensors_df, 'Chiller Evaporator Inlet Temperature', _CH2, 12.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_02_cond_enter_temp_f(config, sensors_df, context):
    """CH-02 condenser entering water temperature (°F). EP source → °C converted to °F."""
    c = get_sensor_value(
        sensors_df, 'Chiller Condenser Inlet Temperature', _CH2, 29.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


def predict_ch_02_cond_leave_temp_f(config, sensors_df, context):
    """CH-02 condenser leaving water temperature (°F). EP source → °C converted to °F."""
    c = get_sensor_value(
        sensors_df, 'Chiller Condenser Outlet Temperature', _CH2, 35.0
    )
    return round((c * 9.0 / 5.0) + 32.0, 1)


# --- Flow rates ---

def predict_ch_02_evap_water_flow_gpm(config, sensors_df, context):
    """CH-02 evaporator water flow: kg/s -> GPM."""
    return get_evap_flow_kgs(sensors_df, _CH2) * 15.850323


def predict_ch_02_cond_water_flow_gpm(config, sensors_df, context):
    """CH-02 condenser water flow: kg/s -> GPM."""
    return get_cond_flow_kgs(sensors_df, _CH2) * 15.850323


def predict_ch_02_power_consumption_kw(config, sensors_df, context):
    """CH-02 power consumption: W -> kW (float64, 2 decimal places).
    Historian stores this as REAL (kW) — must not be int32.
    """
    return round(get_power_w(sensors_df, _CH2) / 1000.0, 2)


def predict_ch_02_power_kw(config, sensors_df, context):
    """CH-02 power: W -> kW."""
    return round(get_power_w(sensors_df, _CH2) / 1000.0)


def predict_ch_02_calc_capacity_tons(config, sensors_df, context):
    """CH-02 calculated capacity: W -> tons."""
    return get_cooling_rate_w(sensors_df, _CH2) / 3516.853


def predict_ch_02_evap_diff_press(config, sensors_df, context):
    """
    CH-02 evaporator differential pressure (PSI).
    Physics stand-in: ΔP = k_evap × flow_GPM²
    """
    flow_gpm = predict_ch_02_evap_water_flow_gpm(config, sensors_df, context)
    return round(max(0.0, EVAP_DP_COEFF * flow_gpm ** 2), 2)


def predict_ch_02_cond_diff_press(config, sensors_df, context):
    """
    CH-02 condenser differential pressure (PSI).
    Physics stand-in: ΔP = k_cond × flow_GPM²
    """
    flow_gpm = predict_ch_02_cond_water_flow_gpm(config, sensors_df, context)
    return round(max(0.0, COND_DP_COEFF * flow_gpm ** 2), 2)


def predict_ch_02_line_current(config, sensors_df, context):
    """CH-02 line current (A)."""
    return round(
        line_current_from_power(get_power_w(sensors_df, _CH2),
                                get_plr(sensors_df, _CH2)), 1
    )


def predict_ch_02_starter_voltage(config, sensors_df, context):
    """CH-02 starter phase voltage (V). 480V when running."""
    return 480 if get_plr(sensors_df, _CH2) > 0.0 else 0


def predict_ch_02_starter_current_ln(config, sensors_df, context):
    """CH-02 per-phase current (A)."""
    return round(
        line_current_from_power(get_power_w(sensors_df, _CH2),
                                get_plr(sensors_df, _CH2))
    )


def predict_ch_02_starter_current_pct(config, sensors_df, context):
    """CH-02 current as % FLA."""
    current = line_current_from_power(
        get_power_w(sensors_df, _CH2), get_plr(sensors_df, _CH2)
    )
    return round(safe_divide(current, CHILLER_FLA_A) * 100.0, 1)


def predict_ch_02_starter_load_pf(config, sensors_df, context):
    """CH-02 power factor."""
    plr = get_plr(sensors_df, _CH2)
    if plr <= 0.0:
        return 0.0
    return round(0.85 + (min(1.0, plr) * 0.10), 3)


def predict_ch_02_evap_refrig_press(config, sensors_df, context):
    """CH-02 evaporator refrigerant pressure (kPa)."""
    return round(r134a_sat_pressure_kpa(get_evap_leave_temp_c(sensors_df, _CH2) - 2.0), 1)


def predict_ch_02_cond_refrig_press(config, sensors_df, context):
    """CH-02 condenser refrigerant pressure (kPa)."""
    return round(r134a_sat_pressure_kpa(get_cond_leave_temp_c(sensors_df, _CH2) + 3.0), 1)


def predict_ch_02_diff_refrig_press(config, sensors_df, context):
    """CH-02 refrigerant differential pressure (kPa)."""
    cond_p = predict_ch_02_cond_refrig_press(config, sensors_df, context)
    evap_p = predict_ch_02_evap_refrig_press(config, sensors_df, context)
    return round(max(0.0, cond_p - evap_p), 1)


def predict_ch_02_evap_sat_rfgt_temp(config, sensors_df, context):
    """CH-02 evaporator saturated refrigerant temperature (°C)."""
    return round(get_evap_leave_temp_c(sensors_df, _CH2) - 2.0, 1)


def predict_ch_02_cond_sat_rfgt_temp(config, sensors_df, context):
    """CH-02 condenser saturated refrigerant temperature (°C)."""
    return round(get_cond_leave_temp_c(sensors_df, _CH2) + 3.0, 1)


def predict_ch_02_comp_rfgt_disch_temp(config, sensors_df, context):
    """CH-02 compressor refrigerant discharge temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    cond_sat = predict_ch_02_cond_sat_rfgt_temp(config, sensors_df, context)
    superheat = 10.0 + (min(1.0, plr) * 15.0)
    return round(cond_sat + superheat if plr > 0.0 else 0.0, 1)


def predict_ch_02_oil_tank_press(config, sensors_df, context):
    """CH-02 oil tank pressure (kPa)."""
    return round(OIL_TANK_PRESS_BASE_KPA if get_plr(sensors_df, _CH2) > 0.0 else 0.0, 1)


def predict_ch_02_oil_pump_disch_press(config, sensors_df, context):
    """CH-02 oil pump discharge pressure (kPa)."""
    tank = predict_ch_02_oil_tank_press(config, sensors_df, context)
    return round(tank + OIL_PUMP_DELTA_KPA if tank > 0.0 else 0.0, 1)


def predict_ch_02_oil_diff_press(config, sensors_df, context):
    """CH-02 oil differential pressure (kPa)."""
    return round(OIL_DIFF_PRESS_KPA if get_plr(sensors_df, _CH2) > 0.0 else 0.0, 1)


def predict_ch_02_oil_tank_temp(config, sensors_df, context):
    """CH-02 oil tank temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    if plr <= 0.0:
        return round(OIL_TANK_TEMP_IDLE_C, 1)
    return round(
        OIL_TANK_TEMP_IDLE_C + plr * (OIL_TANK_TEMP_FULL_C - OIL_TANK_TEMP_IDLE_C), 1
    )


def predict_ch_02_igv1_pct_open(config, sensors_df, context):
    """CH-02 inlet guide vane 1 position (% open)."""
    plr = get_plr(sensors_df, _CH2)
    if plr <= 0.0:
        return 0.0
    return round(max(15.0, min(100.0, plr * 100.0)), 1)


def predict_ch_02_igv2_pct_open(config, sensors_df, context):
    """CH-02 inlet guide vane 2 (same as IGV1)."""
    return predict_ch_02_igv1_pct_open(config, sensors_df, context)


def predict_ch_02_bearing_temp(config, sensors_df, context):
    """CH-02 bearing temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    return round(
        BEARING_TEMP_AMBIENT_C + plr * (BEARING_TEMP_FULL_LOAD_C - BEARING_TEMP_AMBIENT_C), 1
    )


def predict_ch_02_motor_winding_temp(config, sensors_df, context):
    """CH-02 motor winding temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    return round(
        WINDING_TEMP_AMBIENT_C + plr * (WINDING_TEMP_FULL_LOAD_C - WINDING_TEMP_AMBIENT_C), 1
    )


def predict_ch_02_afd_transistor_temp(config, sensors_df, context):
    """CH-02 AFD transistor temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    return round(
        AFD_TRANSISTOR_TEMP_IDLE_C + plr * (
            AFD_TRANSISTOR_TEMP_FULL_C - AFD_TRANSISTOR_TEMP_IDLE_C), 1
    )


def predict_ch_02_afd_inverter_base_temp(config, sensors_df, context):
    """CH-02 AFD inverter base temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    return round(
        AFD_BASE_TEMP_IDLE_C + plr * (AFD_BASE_TEMP_FULL_C - AFD_BASE_TEMP_IDLE_C), 1
    )


def predict_ch_02_afd_rectifier_base_temp(config, sensors_df, context):
    """CH-02 AFD rectifier base temperature (°C)."""
    plr = get_plr(sensors_df, _CH2)
    return round(
        AFD_BASE_TEMP_IDLE_C + plr * (AFD_BASE_TEMP_FULL_C - AFD_BASE_TEMP_IDLE_C), 1
    )


def predict_ch_02_afd_input_freq(config, sensors_df, context):
    """CH-02 AFD input frequency (Hz). Fixed 60 Hz."""
    return SUPPLY_FREQ_HZ


def predict_ch_02_running_status(config, sensors_df, context):
    """
    CH-02 running status code (5-state Trane UC800 machine).

    States:
        0 = Off / Idle       — PLR == 0 and was already stopped
        1 = Starting         — PLR just crossed 0→>0.05 (transitional)
        2 = Running          — PLR > 0.05 and was already starting or running
        3 = Stopping         — PLR just crossed >0→0 (transitional)
        4 = Needs Service    — injected via fault callback (context key ch_02_fault)

    State 4 is set externally by custom/callback.py fault injection; this
    function will hold state 4 until the fault is cleared in context.
    """
    plr = get_plr(sensors_df, _CH2)
    prev = context.get('ch_02_run_state', 0)

    # Fault injection takes priority — hold until cleared externally
    if context.get('ch_02_fault', False):
        state = 4
    elif plr > 0.05:
        if prev == 0:
            state = 1   # Off → Starting (one-timestep transient)
        else:
            state = 2   # Starting → Running, or remain Running
    else:
        if prev in (1, 2):
            state = 3   # Running/Starting → Stopping (one-timestep transient)
        elif prev == 3:
            state = 0   # Stopping → Off/Idle
        else:
            state = 0   # Already off

    context['ch_02_run_state'] = state
    return state


def predict_ch_02_last_diag_code(config, sensors_df, context):
    """
    CH-02 last diagnostic code.
    Returns 0 (no fault) during normal simulation.
    """
    return context.get('ch_02_diag_code', 0)


def predict_ch_02_compressor_starts(config, sensors_df, context):
    """
    CH-02 lifetime compressor start count.
    Increments on each off->on PLR transition.
    """
    plr = get_plr(sensors_df, _CH2)
    was_running = context.get('ch_02_was_running', False)
    is_running = plr > 0.0

    if is_running and not was_running:
        context['ch_02_starts'] = context.get('ch_02_starts', 0) + 1

    context['ch_02_was_running'] = is_running
    return context.get('ch_02_starts', 0)


def predict_ch_02_compressor_runtime(config, sensors_df, context):
    """
    CH-02 lifetime compressor runtime (hours).
    Accumulates each timestep the chiller is running.
    """
    plr = get_plr(sensors_df, _CH2)
    if plr > 0.0:
        timestep_hours = float(
            config.get('DEFAULT', 'TimestepHours', fallback=str(1.0 / 6.0))
        )
        context['ch_02_runtime_h'] = context.get('ch_02_runtime_h', 0.0) + timestep_hours
    return round(context.get('ch_02_runtime_h', 0.0))
