# -*- coding: utf-8 -*-
"""
EnergyPlus Simulation Manager with Dynamic Callback Registration

This module manages the interface between the digital twin framework and 
the EnergyPlus simulation engine.

Created on Wed Jul 5 10:42:09 2023
Enhanced: February 2026

@author: doylef
"""

from pyenergyplus.api import EnergyPlusAPI
import sys
import os
import time
import datetime as dt
from datetime import datetime
from typing import Callable, Dict, Set, Optional
import logging
import pandas as pd

# Import user extensibility modules
import custom.conversion as reflect_conv
import custom.callback as reflect_callbk
import re

# Configure logging
logger = logging.getLogger(__name__)

def parse_rdd_units(rdd_filepath):
    """
    Parse EnergyPlus .rdd file to extract variable units.
    
    Args:
        rdd_filepath: Path to eplusout.rdd file
        
    Returns:
        dict: {(variable_name, key_value): unit_string}
    """
    import re
    
    units_map = {}
    
    try:
        with open(rdd_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Handle Output:Variable format
                # Example: Output:Variable,*,Site Outdoor Air Drybulb Temperature,hourly; !- HVAC Average [C]
                if line.startswith('Output:Variable'):
                    match = re.search(
                        r'Output:Variable,([^,]+),([^,]+),[^;]+;\s*!-[^\[]*\[([^\]]*)\]',
                        line
                    )
                    
                    if match:
                        key_value = match.group(1).strip()
                        var_name = match.group(2).strip()
                        unit = match.group(3).strip()
                        units_map[(var_name, key_value)] = unit
                
                # Handle Output:Meter format
                # Example: Output:Meter,Electricity:Facility,hourly; !- [J]
                elif line.startswith('Output:Meter'):
                    match = re.search(
                        r'Output:Meter,([^,]+),[^;]+;\s*!-\s*\[([^\]]*)\]',
                        line
                    )
                    
                    if match:
                        meter_name = match.group(1).strip()
                        unit = match.group(2).strip()
                        units_map[(meter_name, 'METER')] = unit
        
        return units_map
        
    except FileNotFoundError:
        logger.error(f"RDD file not found: {rdd_filepath}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing RDD file: {e}")
        return {}


def match_sensor_to_rdd(sensor_name, sensor_instance, units_map):
    """
    Find unit for a sensor from the .rdd units map.
    
    Args:
        sensor_name: Variable name (e.g., "Chiller Evaporator Cooling Rate")
        sensor_instance: Instance name (e.g., "CHILLER 1") or "METER"
        units_map: Dict from parse_rdd_units()
        
    Returns:
        Unit string or None if not found
    """
    # Try exact match with instance
    for (rdd_name, rdd_instance), unit in units_map.items():
        if rdd_name.lower() == sensor_name.lower() and rdd_instance == sensor_instance:
            return unit
    
    # Try wildcard match (*)
    unit = units_map.get((sensor_name, '*'))
    if unit:
        return unit
    
    return None

class EpManager:
    """
    Manages EnergyPlus simulation lifecycle and API interactions.
    
    Features:
    - Dynamic callback registration based on actuator/sensor/custom configuration
    - Extensible API hooks for custom EnergyPlus API interactions
    - Real-time synchronization with physical building
    - Comprehensive error handling and logging
    """
    
    # Complete registry of available EnergyPlus callbacks
    CALLBACK_REGISTRY = {
        'begin_new_environment': 'callback_begin_new_environment',
        'after_component_get_input': 'callback_after_component_get_input',
        'after_new_environment_warmup_complete': 'callback_after_new_environment_warmup_complete',
        'begin_zone_timestep_before_init_heat_balance': 'callback_begin_zone_timestep_before_init_heat_balance',
        'begin_zone_timestep_after_init_heat_balance': 'callback_begin_zone_timestep_after_init_heat_balance',
        'after_predictor_before_hvac_managers': 'callback_after_predictor_before_hvac_managers',
        'after_predictor_after_hvac_managers': 'callback_after_predictor_after_hvac_managers',
        'begin_system_timestep_before_predictor': 'callback_begin_system_timestep_before_predictor',
        'end_system_sizing': 'callback_end_system_sizing',
        'end_system_timestep_after_hvac_reporting': 'callback_end_system_timestep_after_hvac_reporting',
        'end_system_timestep_before_hvac_reporting': 'callback_end_system_timestep_before_hvac_reporting',
        'end_zone_sizing': 'callback_end_zone_sizing',
        'end_zone_timestep_before_zone_reporting': 'callback_end_zone_timestep_before_zone_reporting',
        'end_zone_timestep_after_zone_reporting': 'callback_end_zone_timestep_after_zone_reporting',
        'inside_system_iteration_loop': 'callback_inside_system_iteration_loop',
        'message': 'callback_message',
        'progress': 'callback_progress',
        'unitary_system_sizing': 'callback_unitary_system_sizing',
    }
    
    def __init__(self, digital_twin):
        """Initialize EnergyPlus manager"""
        logger.info("Initializing EnergyPlus Manager")
        
        self.dtwin = digital_twin
        self.config = digital_twin.config
        
        # Simulation state
        self.got_handles = False
        self.proceed_with_step_logic = False
        self.simulation_datetime = None
        self.units_updated = False
                
        # EnergyPlus API initialization
        try:
            self.ep_api = EnergyPlusAPI()
            self.ep_state = self.ep_api.state_manager.new_state()
            logger.info("EnergyPlus API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnergyPlus API: {e}", exc_info=True)
            raise
        
        # Determine which callbacks are needed
        self.active_callbacks = self._determine_active_callbacks()
        logger.info(f"Determined {len(self.active_callbacks)} active callbacks")
        
        # Register callbacks
        self._register_callbacks()
        
        # File paths
        self.custom_input_file_path = os.path.join(
            self.dtwin.working_directory, 
            'dt_in.idf'
        )
        
        logger.info("EnergyPlus Manager initialization complete")

    def update_missing_units_from_rdd(self):
        """
        Update missing units in database from .rdd file.
        
        Called during after_new_environment_warmup_complete callback
        if UPDATE_SIGNAL_UNITS=TRUE in config.
        """
        # Check if feature is enabled
        update_enabled = self.config.getboolean('DATABASE', 'UPDATE_SIGNAL_UNITS', fallback=False)
        
        if not update_enabled:
            logger.debug("UPDATE_SIGNAL_UNITS is FALSE, skipping unit population")
            return
        
        logger.info("=" * 70)
        logger.info("UPDATING MISSING UNITS FROM .rdd FILE")
        logger.info("=" * 70)
        
        # Get .rdd file path
        rdd_path = os.path.join(self.dtwin.working_directory, 'out', 'eplusout.rdd')      
        if not os.path.exists(rdd_path):
            logger.warning(f"RDD file not found: {rdd_path}")
            logger.warning("Skipping unit update (file will be created after warmup)")
            return
        
        # Parse .rdd file
        logger.info(f"Parsing RDD file: {rdd_path}")
        units_map = parse_rdd_units(rdd_path)
        logger.info(f"Found {len(units_map)} variable/unit pairs in .rdd")
        
        # Check if persistence is configured
        if not hasattr(self.dtwin, 'persistence_agent') or self.dtwin.persistence_agent is None:
            logger.warning("No database persistence configured, skipping unit update")
            return
        
        db = self.dtwin.persistence_agent
        
        # Get sensors that need units
        updated_count = 0
        not_found_count = 0
        
        try:
            # Iterate through sensors_df to find missing units
            for idx in self.dtwin.sensors_df.index:
                sensor_name = self.dtwin.sensors_df.loc[idx, 'SensorName']
                sensor_instance = self.dtwin.sensors_df.loc[idx, 'SensorInstance']
                persistence_name = self.dtwin.sensors_df.loc[idx, 'PersistenceName']
                
                # Get signal_id from database's cache
                signal_id = db.signal_id_cache.get(persistence_name)
                
                if not signal_id:
                    logger.debug(f"Signal '{persistence_name}' not found in database cache, skipping")
                    continue
                
                # Check if unit already exists in database
                conn = db.connection_pool.getconn()
                try:
                    cur = conn.cursor()
                    cur.execute(f"SELECT unit FROM {db.signals_table} WHERE signal_id = %s", (signal_id,))
                    result = cur.fetchone()
                    existing_unit = result[0] if result else None
                    cur.close()
                finally:
                    db.connection_pool.putconn(conn)
                
                # Skip if unit already populated in database
                if existing_unit and existing_unit not in ['', 'unknown']:
                    # But make sure it's in sensors_df (might have been cleared)
                    current_unit_in_df = self.dtwin.sensors_df.loc[idx, 'unit']
                    if pd.notna(current_unit_in_df):
                        continue  # Already in both DB and sensors_df
                    else:
                        # In DB but not in sensors_df - populate it
                        self.dtwin.sensors_df.loc[idx, 'unit'] = existing_unit
                        logger.debug(f"  ℹ Restored unit from DB: {sensor_name} → [{existing_unit}]")
                    continue
                
                # Determine instance key for matching
                sensor_type = self.dtwin.sensors_df.loc[idx, 'Type']
                if sensor_type == 'meter':
                    instance_key = 'METER'
                else:
                    instance_key = sensor_instance if sensor_instance else '*'
                
                # Find unit from .rdd
                unit = match_sensor_to_rdd(sensor_name, instance_key, units_map)
                
                if unit:
                    # Update in sensors_df
                    self.dtwin.sensors_df.loc[idx, 'unit'] = unit
                    
                    # Update in database (we already have signal_id from above)
                    try:
                        db.update_signal_unit(signal_id, unit)
                        logger.info(f"  ✓ Updated: {sensor_name} ({sensor_instance}) → [{unit}]")
                        updated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to update unit in DB for '{persistence_name}': {e}")

                else:
                    logger.debug(f"  ✗ Not found: {sensor_name} ({sensor_instance})")
                    not_found_count += 1
            
            # Summary
            logger.info("=" * 70)
            logger.info(f"UNIT UPDATE SUMMARY:")
            logger.info(f"  Units updated:       {updated_count}")
            logger.info(f"  Units not found:     {not_found_count}")
            logger.info("=" * 70)
            
            if updated_count > 0:
                logger.info("✓ Unit update complete!")
                logger.info("  Recommendation: Set UPDATE_SIGNAL_UNITS=FALSE in config.ini")
                logger.info("  to avoid unnecessary processing on future runs.")
        
        except Exception as e:
            logger.error(f"Error updating units from .rdd: {e}", exc_info=True)
    
    
    def _determine_active_callbacks(self) -> Set[str]:
        """Determine which callbacks are needed by examining configuration"""
        active = set()
        
        # Check actuators
        if 'Override_stage' in self.dtwin.actuators_df.columns:
            actuator_callbacks = self.dtwin.actuators_df['Override_stage'].dropna().unique()
            active.update(actuator_callbacks)
            logger.info(f"Actuators require callbacks at: {sorted(actuator_callbacks)}")
        
        # Check sensors
        if 'Read_stage' in self.dtwin.sensors_df.columns:
            sensor_callbacks = self.dtwin.sensors_df['Read_stage'].dropna().unique()
            active.update(sensor_callbacks)
            logger.info(f"Sensors require callbacks at: {sorted(sensor_callbacks)}")
        
        # Check custom callbacks
        if hasattr(self.dtwin, 'custom_callbacks_df') and 'TimePeriod' in self.dtwin.custom_callbacks_df.columns:
            custom_callbacks = self.dtwin.custom_callbacks_df['TimePeriod'].dropna().unique()
            active.update(custom_callbacks)
            logger.info(f"Custom callbacks require: {sorted(custom_callbacks)}")

        required_callbacks = {
            'begin_zone_timestep_before_init_heat_balance',
            'end_zone_timestep_after_zone_reporting',
            'after_new_environment_warmup_complete'
        }
        active.update(required_callbacks)
        
        # Validate
        unknown_callbacks = active - set(self.CALLBACK_REGISTRY.keys())
        if unknown_callbacks:
            logger.warning(f"Unknown callbacks requested (will be ignored): {unknown_callbacks}")
            active -= unknown_callbacks
        
        logger.info(f"Total active callbacks: {sorted(active)}")
        return active
    
    def _register_callbacks(self):
        """Register only the callbacks that are actually used"""
        logger.info("Registering EnergyPlus callbacks...")
        logger.info(f"Active callbacks to register: {sorted(self.active_callbacks)}")
        
        registered_count = 0
        for callback_name in self.active_callbacks:
            if callback_name not in self.CALLBACK_REGISTRY:
                logger.warning(f"Cannot register unknown callback: {callback_name}")
                continue
            
            try:
                registration_method_name = self.CALLBACK_REGISTRY[callback_name]
                registration_method = getattr(self.ep_api.runtime, registration_method_name)
                callback_func = self._create_callback_handler(callback_name)
                registration_method(self.ep_state, callback_func)
                logger.info(f"  ✓ Registered: {callback_name} -> {registration_method_name}")
                registered_count += 1
            except Exception as e:
                logger.error(f"Failed to register callback '{callback_name}': {e}", exc_info=True)
        
        logger.info(f"Successfully registered {registered_count}/{len(self.active_callbacks)} callbacks")
    
    def _create_callback_handler(self, callback_name: str) -> Callable:
        """Create a callback handler function for the given callback name"""
        def handler(state):
            try:
                self._handle_callback(state, callback_name)
            except Exception as e:
                logger.error(f"Error in callback '{callback_name}': {e}", exc_info=True)
        return handler
    
    def _handle_callback(self, state, callback_name: str):
        """Generic callback handler"""
        logger.debug(f"Callback fired: {callback_name}")
        
        if callback_name == 'begin_zone_timestep_before_init_heat_balance':
            self._handle_initialization_callback(state)
            return
        
        if callback_name == 'end_zone_timestep_after_zone_reporting':
            self._handle_reporting_callback(state, callback_name)
            return
        
        if callback_name == 'after_new_environment_warmup_complete':
            logger.debug("Routing to _handle_warmup_complete")
            self._handle_warmup_complete(state)
            return  
        
        # Standard callback processing        
        if self.proceed_with_step_logic:
            self._execute_callback_actions(callback_name)
        else:
            logger.debug(f"Skipping callback {callback_name} (proceed_with_step_logic=False)")
    
    def _handle_initialization_callback(self, state):
        """Handle the primary initialization and time synchronization callback"""
        self.proceed_with_step_logic = False
        self.ep_state = state
        
        if not self.got_handles:
            if not self.ep_api.exchange.api_data_fully_ready(self.ep_state):
                logger.debug("API data not fully ready, waiting...")
                return
            
            logger.info("API data ready, obtaining handles...")
            
            try:
                sensors_ok = self.set_sensor_handles()
                if not sensors_ok:
                    raise RuntimeError("Failed to obtain all sensor handles")
                
                actuators_ok = self.set_actuator_handles()
                if not actuators_ok:
                    raise RuntimeError("Failed to obtain all actuator handles")
                
                self.got_handles = True
                logger.info("✓ All handles obtained successfully")
            except Exception as e:
                logger.error(f"Handle initialization failed: {e}", exc_info=True)
                self.ep_api.runtime.stop_simulation(self.ep_state)
                raise
        
        elif self.ep_api.exchange.warmup_flag(self.ep_state):
            logger.debug("Simulation in warmup period")
            return
        
        else:
            try:
                self.setCurrentSimulationTime()
                logger.debug(f"Simulation time: {self.simulation_datetime}")
                
                self._wait_for_realtime_sync()
                
                self.proceed_with_step_logic = True
                self.get_actuator_values_by_signals()
                self._execute_callback_actions('begin_zone_timestep_before_init_heat_balance')
            except Exception as e:
                logger.error(f"Error in initialization callback: {e}", exc_info=True)
    
    def _handle_reporting_callback(self, state, callback_name: str):
        """Handle the final reporting/persistence callback"""
        if self.proceed_with_step_logic:
            try:
                self._execute_callback_actions(callback_name)
                self.dtwin.store_simulated_signals(self.simulation_datetime)
                logger.debug(f"Data persisted for {self.simulation_datetime}")
            except Exception as e:
                logger.error(f"Error in reporting callback: {e}", exc_info=True)

    def _handle_warmup_complete(self, state):
        """Handle the warmup complete callback"""
       
        logger.info("=" * 70)
        logger.info("WARMUP CALLBACK FIRED")
        logger.info("=" * 70)

        # Update units if enabled
        update_enabled = self.config.getboolean('DATABASE', 'UPDATE_SIGNAL_UNITS', fallback=False)
        logger.info(f"UPDATE_SIGNAL_UNITS setting: {update_enabled}")
        
        if update_enabled  and not self.units_updated:
        # Only enters if BOTH conditions true - short circuits if update_enabled is False
            logger.info("Calling update_missing_units_from_rdd()...")
            self.update_missing_units_from_rdd()
            self.units_updated = True
        elif update_enabled and self.units_updated:
        # Only logs if feature enabled but already ran
            logger.info("Update for missing units has already been called.")
        else:
            logger.info("UPDATE_SIGNAL_UNITS disabled, skipping")
        
        # Run custom callbacks
        logger.info("Running custom callbacks for warmup complete...")
        self._execute_callback_actions('after_new_environment_warmup_complete')
        
        logger.info("Warmup callback processing complete")
        logger.info("=" * 70)
    
    def _execute_callback_actions(self, callback_name: str):
        """Execute the standard actions for a callback"""
        try:
            self.setActuators(callback_name)
            self.run_custom(callback_name)
            self.collectSensorData(callback_name)
        except Exception as e:
            logger.error(f"Error executing callback actions for '{callback_name}': {e}", exc_info=True)
    
    def _wait_for_realtime_sync(self):
        """Wait for real-time synchronization if configured"""
        try:
            buffer_minutes = int(self.config.get('DEFAULT', 'TimeBufferMinutes', fallback='5'))
            
            if buffer_minutes > 0:
                time_buffer = dt.timedelta(minutes=buffer_minutes)
                target_time = self.simulation_datetime + time_buffer
                
                while datetime.now() < target_time:
                    time.sleep(5)
                    logger.debug(f"Waiting for real-time sync (target: {target_time})")
        except Exception as e:
            logger.warning(f"Error in real-time sync: {e}")
 
    def setCurrentSimulationTime(self):
        """Set current simulation time from EnergyPlus state"""
        try:
            year = self.dtwin.start_year
            month = self.ep_api.exchange.month(self.ep_state)
            day = self.ep_api.exchange.day_of_month(self.ep_state)
            hour = self.ep_api.exchange.hour(self.ep_state)
            minute = self.ep_api.exchange.minutes(self.ep_state)
            
            # Convert float hour/minute to proper datetime
            # EnergyPlus can return values like hour=23.75 or minute=65
            total_minutes = int(hour * 60 + minute)
            extra_hours, final_minute = divmod(total_minutes, 60)
            extra_days, final_hour = divmod(extra_hours, 24)
            
            dtime = dt.datetime(
                year=year, 
                month=int(month), 
                day=int(day), 
                hour=0, 
                minute=0
            )
            dtime += dt.timedelta(days=extra_days, hours=final_hour, minutes=final_minute)
            
            self.simulation_datetime = dtime
            
        except Exception as e:
            logger.error(f"Error setting simulation time: {e}", exc_info=True)
            if self.simulation_datetime is None:
                raise  
    ''' 
    def setCurrentSimulationTime(self):
        """Set current simulation time from EnergyPlus state"""
        try:
            year = self.dtwin.start_year
            month = self.ep_api.exchange.month(self.ep_state)
            day = self.ep_api.exchange.day_of_month(self.ep_state)
            hour = self.ep_api.exchange.hour(self.ep_state)
            minute = self.ep_api.exchange.minutes(self.ep_state)
            
            timedelta = dt.timedelta()
            
            if hour >= 24.0:
                logger.warning(f"EP returned invalid hour: {hour}, adjusting to 23")
                hour = 23.0
                timedelta += dt.timedelta(hours=1)
            
            if minute >= 60.0:
                logger.warning(f"EP returned invalid minute: {minute}, adjusting to 59")
                minute = 59
                timedelta += dt.timedelta(minutes=1)
            
            dtime = dt.datetime(year=year, month=int(month), day=int(day), hour=int(hour), minute=int(minute))
            dtime += timedelta
            self.simulation_datetime = dtime
        except Exception as e:
            logger.error(f"Error setting simulation time: {e}", exc_info=True)
            if self.simulation_datetime is None:
                raise '''
    
    def collectSensorData(self, timepoint: str):
        """Collect sensor data for the specified timepoint"""
        sensors_read = 0
        sensors_failed = 0
        
        for idx in self.dtwin.sensors_df.index:
            if self.dtwin.sensors_df['Read_stage'][idx] != timepoint:
                continue
            
            try:
                curr_sensor_handle = self.dtwin.sensors_df['ep_handle'][idx]
                sensor_type = self.dtwin.sensors_df['Type'][idx]
                sensor_name = self.dtwin.sensors_df['SensorName'][idx]
                
                if sensor_type == 'sensor':
                    value = self.ep_api.exchange.get_variable_value(self.ep_state, curr_sensor_handle)
                elif sensor_type == 'meter':
                    value = self.ep_api.exchange.get_meter_value(self.ep_state, curr_sensor_handle)
                else:
                    logger.warning(f"Unknown sensor type '{sensor_type}' for {sensor_name}")
                    sensors_failed += 1
                    continue
                
                self.dtwin.sensors_df.iloc[idx, self.dtwin.sensors_df.columns.get_loc('current_val')] = value
                sensors_read += 1
                logger.debug(f"Read sensor '{sensor_name}' = {value}")
            except Exception as e:
                sensors_failed += 1
                logger.error(f"Failed to read sensor at index {idx}: {e}", exc_info=False)
        
        if sensors_read > 0:
            logger.debug(f"Collected {sensors_read} sensors at '{timepoint}' ({sensors_failed} failed)")
    
    def get_actuator_values_by_signals(self):
        """Retrieve signal values and apply to actuators with conversions"""
        try:
            self.dtwin.get_signals_for_timepoint(self.simulation_datetime)
            
            actuators_updated = 0
            actuators_failed = 0
            
            for idx in self.dtwin.actuators_df.index:
                try:
                    curr_source_tagname = self.dtwin.actuators_df['SourceTagName'][idx]
                    curr_conversion = self.dtwin.actuators_df['ConversionFunction'][idx]
                    
                    signal_mask = self.dtwin.signals_df['SignalTagName'] == curr_source_tagname
                    
                    if not signal_mask.any():
                        logger.warning(f"Signal not found for actuator: {curr_source_tagname}")
                        actuators_failed += 1
                        continue
                    
                    curr_signal_value = self.dtwin.signals_df.loc[signal_mask, 'current_val'].iloc[0]
                    
                    if curr_conversion and curr_conversion != "none":
                        try:
                            conversion_func = getattr(reflect_conv, curr_conversion)
                            curr_signal_value = conversion_func(self.config, self.simulation_datetime, curr_signal_value)
                            logger.debug(f"Applied conversion '{curr_conversion}' to {curr_source_tagname}")
                        except AttributeError:
                            logger.error(f"Conversion function not found: {curr_conversion}")
                        except Exception as e:
                            logger.error(f"Error in conversion '{curr_conversion}': {e}")
                    
                    self.dtwin.actuators_df.iloc[idx, self.dtwin.actuators_df.columns.get_loc('current_val')] = curr_signal_value
                    actuators_updated += 1
                except Exception as e:
                    actuators_failed += 1
                    logger.error(f"Failed to update actuator at index {idx}: {e}", exc_info=False)
            
            logger.debug(f"Updated {actuators_updated} actuator values ({actuators_failed} failed)")
        except Exception as e:
            logger.error(f"Error getting actuator values by signals: {e}", exc_info=True)
    
    def setActuators(self, timepoint: str):
        """Set actuator values for the specified timepoint"""
        actuators_set = 0
        actuators_failed = 0
        
        for idx in self.dtwin.actuators_df.index:
            if self.dtwin.actuators_df['Override_stage'][idx] != timepoint:
                continue
            
            try:
                handle = self.dtwin.actuators_df['ep_handle'][idx]
                value = self.dtwin.actuators_df['current_val'][idx]
                actuator_name = self.dtwin.actuators_df['ActuatorName'][idx]
                
                self.ep_api.exchange.set_actuator_value(self.ep_state, handle, value)
                actuators_set += 1
                logger.debug(f"Set actuator '{actuator_name}' = {value} at '{timepoint}'")
            except Exception as e:
                actuators_failed += 1
                logger.error(f"Failed to set actuator at index {idx}: {e}", exc_info=False)
        
        if actuators_set > 0:
            logger.debug(f"Set {actuators_set} actuators at '{timepoint}' ({actuators_failed} failed)")
    
    def run_custom(self, timeperiod: str):
        """Execute custom user callbacks for the specified timeperiod"""
        callbacks_run = 0
        callbacks_failed = 0
        
        for idx in self.dtwin.custom_callbacks_df.index:
            if self.dtwin.custom_callbacks_df['TimePeriod'][idx] != timeperiod:
                continue
            
            curr_callback = self.dtwin.custom_callbacks_df['Function'][idx]
            
            try:
                custom_func = getattr(reflect_callbk, curr_callback)
                custom_func(self.dtwin)
                callbacks_run += 1
                logger.debug(f"Executed custom callback: {curr_callback}")
            except AttributeError:
                callbacks_failed += 1
                logger.error(f"Custom callback function not found: {curr_callback}")
            except Exception as e:
                callbacks_failed += 1
                logger.error(f"Error executing custom callback '{curr_callback}': {e}", exc_info=True)
        
        if callbacks_run > 0:
            logger.debug(f"Executed {callbacks_run} custom callbacks at '{timeperiod}' ({callbacks_failed} failed)")
    
    def set_actuator_handles(self) -> bool:
        """Set handles for all actuators"""
        logger.info("Setting actuator handles...")
        failed_actuators = []
        
        for idx in self.dtwin.actuators_df.index:
            try:
                curr_actuator_category = self.dtwin.actuators_df['ActuatorCategory'][idx]
                curr_actuator_name = self.dtwin.actuators_df['ActuatorName'][idx]
                curr_actuator_instance = self.dtwin.actuators_df['ActuatorInstance'][idx]
                
                handle = self.ep_api.exchange.get_actuator_handle(
                    self.ep_state,
                    curr_actuator_category,
                    curr_actuator_name,
                    curr_actuator_instance
                )
                
                self.dtwin.actuators_df.iloc[idx, self.dtwin.actuators_df.columns.get_loc('ep_handle')] = handle
                
                if handle == -1:
                    failed_actuators.append({
                        'category': curr_actuator_category,
                        'name': curr_actuator_name,
                        'instance': curr_actuator_instance
                    })
            except Exception as e:
                logger.error(f"Error getting actuator handle at index {idx}: {e}")
                failed_actuators.append({'index': idx, 'error': str(e)})
        
        if failed_actuators:
            logger.error(f"Failed to get {len(failed_actuators)} actuator handles:")
            for actuator in failed_actuators:
                logger.error(f"  {actuator}")
            return False
        else:
            logger.info(f"✓ All {len(self.dtwin.actuators_df)} actuator handles obtained")
            return True
    
    def set_sensor_handles(self) -> bool:
        """Set handles for all sensors and meters"""
        logger.info("Setting sensor handles...")
        failed_sensors = []
        
        for idx in self.dtwin.sensors_df.index:
            try:
                curr_sensor_name = self.dtwin.sensors_df['SensorName'][idx]
                curr_sensor_instance = self.dtwin.sensors_df['SensorInstance'][idx]
                curr_sensor_type = self.dtwin.sensors_df['Type'][idx]
                
                if curr_sensor_type == 'sensor':
                    handle = self.ep_api.exchange.get_variable_handle(self.ep_state, curr_sensor_name, curr_sensor_instance)
                elif curr_sensor_type == 'meter':
                    handle = self.ep_api.exchange.get_meter_handle(self.ep_state, curr_sensor_name)
                else:
                    logger.error(f"Unknown sensor type: {curr_sensor_type}")
                    handle = -1
                
                self.dtwin.sensors_df.iloc[idx, self.dtwin.sensors_df.columns.get_loc('ep_handle')] = handle
                
                if handle == -1:
                    failed_sensors.append({
                        'name': curr_sensor_name,
                        'instance': curr_sensor_instance,
                        'type': curr_sensor_type
                    })
            except Exception as e:
                logger.error(f"Error getting sensor handle at index {idx}: {e}")
                failed_sensors.append({'index': idx, 'error': str(e)})
        
        if failed_sensors:
            logger.error(f"Failed to get {len(failed_sensors)} sensor handles:")
            for sensor in failed_sensors:
                logger.error(f"  {sensor}")
            return False
        else:
            logger.info(f"✓ All {len(self.dtwin.sensors_df)} sensor handles obtained")
            return True
    
    def prep_input_file_for_simulation(self):
        """Override the building model IDF file RunPeriod to match requested time period"""
        logger.info("Preparing input file for simulation...")
        
        try:
            with open(self.config.get('ENERGYPLUS', 'EPBuildingModel'), 'rt') as base_file:
                with open(self.custom_input_file_path, 'wt') as custom_file:
                    in_run_period = False
                    
                    for line in base_file:
                        if line.strip().startswith('RunPeriod'):
                            in_run_period = True
                            custom_file.write(line)
                            continue
                        
                        if in_run_period:
                            found_key = False
                            for key, replacement in self.dtwin.override_map.items():
                                if key in line:
                                    custom_file.write(replacement)
                                    found_key = True
                                    logger.debug(f"Overriding: {key}")
                                    break
                            
                            if not found_key:
                                custom_file.write(line)
                            
                            if ';' in line:
                                in_run_period = False
                        else:
                            custom_file.write(line)
            
            logger.info(f"✓ Input file prepared: {self.custom_input_file_path}")
        except Exception as e:
            logger.error(f"Failed to prepare input file: {e}", exc_info=True)
            raise
    
    def invoke_simulation(self):
        """Prepare and invoke the EnergyPlus simulation"""
        logger.info("=" * 70)
        logger.info("INVOKING ENERGYPLUS SIMULATION")
        logger.info("=" * 70)
        
        try:
            self.prep_input_file_for_simulation()
            
            logger.info("Requesting sensor variables...")
            for idx in self.dtwin.sensors_df.index:
                try:
                    sensor_name = self.dtwin.sensors_df['SensorName'][idx]
                    sensor_instance = self.dtwin.sensors_df['SensorInstance'][idx]
                    
                    self.ep_api.exchange.request_variable(self.ep_state, sensor_name, sensor_instance)
                    logger.debug(f"Requested: {sensor_name} ({sensor_instance})")
                except Exception as e:
                    logger.error(f"Failed to request sensor at index {idx}: {e}")
            
            logger.info(f"✓ Requested {len(self.dtwin.sensors_df)} sensors")
            
            logger.info("Starting EnergyPlus simulation...")
            logger.info(f"  Weather file: {self.config.get('ENERGYPLUS', 'EPWeatherFile')}")
            logger.info(f"  Building model: {self.custom_input_file_path}")
            logger.info(f"  Output directory: out")
            
            self.ep_api.runtime.run_energyplus(
                self.ep_state,
                [
                    '-w', self.config.get('ENERGYPLUS', 'EPWeatherFile'),
                    '-d', 'out',
                    self.custom_input_file_path
                ]
            )
            
            logger.info("=" * 70)
            logger.info("ENERGYPLUS SIMULATION COMPLETE")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Failed to invoke simulation: {e}", exc_info=True)
            raise