import pvlib
from pvlib import solarposition
import pandas as pd
from datetime import datetime
import math
# -*- coding: utf-8 -*- #
"""
Created on Wed Jul 19 11:08:22 2023

@author: doylef
"""

def convert_F_to_C(config, time, temp):
    new_temp = (temp-32) * (5/9)
    return new_temp

def solar_zenith_angle(config, time):

    # Create location object
    location = pvlib.location.Location(float(config.get('DEFAULT', 'bldg_latitude')), float(config.get('DEFAULT', 'bldg_longitude')), tz=config.get('DEFAULT', 'bldg_tz'), altitude=float(config.get('DEFAULT', 'bldg_altitude')))
    pdtime = pd.Timestamp(time, tz=config.get('DEFAULT', 'bldg_tz'))
    solar_position = location.get_solarposition(pdtime)
    zenith_angle = solar_position['zenith'].values[0]
    return zenith_angle

def convert_mph_to_metps(config, time, speed):
    """Calculate speed in meters/s from mph"""
    mps_speed = speed * 0.44704
    return mps_speed

def convert_ghi_to_dhi(config, time, ghi):
    """Calculate DHI and DNI from GHI (W/m²) using Erbs model."""
    # Step 1: Get solar zenith angle
    theta = solar_zenith_angle(config, time)
    #print("theta:" + str(theta))
    cos_theta = math.cos(math.radians(theta))
    #print("cos_theta:" + str(cos_theta))

    # Step 2: extraterrestrial radiation (GHI₀) on horizontal surface
    solar_constant = 1361  # W/m²
    ghi_0 = solar_constant * cos_theta  # Adjust for zenith

    # Step 3: Clearness index (kT)
    if ghi_0 <= 0:  # Avoid division by zero (sun below horizon)
        return 0  # DHI = 0
    kt = ghi / ghi_0

    # Step 4: Diffuse fraction using Erbs model
    if kt <= 0.22:
        diffuse_fraction = 1.0 - 0.09 * kt
    elif 0.22 < kt <= 0.8:
        diffuse_fraction = 1.0 - 0.09 * kt - 0.6 * (kt - 0.22)**2
    else:  # kt > 0.8
        diffuse_fraction = 0.165

    # Step 5: Calculate DHI and DNI
    dhi = ghi * diffuse_fraction

    return dhi

def convert_inHg_to_Pa(config, time, inhg):
    """Calculate pressure in Pascals from inHg"""
    pressure_in_pa = 3386.39 * inhg

    return pressure_in_pa

def convert_ghi_to_dni(config, time, ghi):
    """Calculate DHI and DNI from GHI (W/m²) using Erbs model."""
    # Step 1: Get solar zenith angle
    theta = solar_zenith_angle(config, time)
    #print("theta:" + str(theta))
    cos_theta = math.cos(math.radians(theta))
    #print("cos_theta:" + str(cos_theta))

    # Step 2: extraterrestrial radiation (GHI₀) on horizontal surface
    solar_constant = 1361  # W/m²
    ghi_0 = solar_constant * cos_theta  # Adjust for zenith

    # Step 3: Clearness index (kT)
    if ghi_0 <= 0:  # Avoid division by zero (sun below horizon)
        return 0  # DNI = 0
    kt = ghi / ghi_0

    # Step 4: Diffuse fraction using Erbs model
    if kt <= 0.22:
        diffuse_fraction = 1.0 - 0.09 * kt
    elif 0.22 < kt <= 0.8:
        diffuse_fraction = 1.0 - 0.09 * kt - 0.6 * (kt - 0.22)**2
    else:  # kt > 0.8
        diffuse_fraction = 0.165

    # Step 5: Calculate DHI and DNI
    dhi = ghi * diffuse_fraction
    if cos_theta <= 0:  # Sun below horizon
        dni = 0
    else:
        dni = (ghi - dhi) / cos_theta

    return dni
def autostop_to_fraction(config, time, autostop):
    """
    Convert OPC cmd_AutoStop boolean to EP On/Off Supervisory fraction.
    AutoStop=True (stop command) -> 0.0 (off)
    AutoStop=False (run command) -> 1.0 (on)
    EP Plant Component Chiller:Electric:EIR On/Off Supervisory: 0=off, 1=on
    """
    try:
        return 0.0 if bool(autostop) else 1.0
    except Exception:
        return 1.0  # default to on if value is unexpected

def clamp_chw_setpoint(config, time, temp_c):
    """
    Clamp OPC CHW setpoint command to safe EP operating range.
    Chiller lower temp limit is 2C (from IDF), upper is 15C for safety.
    Returns value unchanged if already in range.
    """
    try:
        return max(2.0, min(15.0, float(temp_c)))
    except Exception:
        return 6.7  # IDF design setpoint as fallback
