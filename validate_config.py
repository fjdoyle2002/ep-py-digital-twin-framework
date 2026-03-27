#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Twin Configuration Validator

Validates all configuration files before running simulation to catch
errors early and provide clear feedback.

Usage:
    python validate_config.py /path/to/working_directory

Returns:
    Exit code 0 if all validations pass
    Exit code 1 if any validation fails
"""

import sys
import os
import configparser
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates digital twin configuration files"""
    
    def __init__(self, working_directory: str):
        self.working_dir = Path(working_directory)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.config: Optional[configparser.ConfigParser] = None
        
        # Track validation statistics
        self.files_checked = 0
        self.validations_passed = 0
        self.validations_failed = 0
    
    def validate_all(self) -> bool:
        """
        Run all validations.
        
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info("=" * 70)
        logger.info("DIGITAL TWIN CONFIGURATION VALIDATION")
        logger.info("=" * 70)
        logger.info(f"Working Directory: {self.working_dir}")
        logger.info("")
        
        # 1. Validate working directory exists
        if not self._validate_working_directory():
            return False
        
        # 2. Validate config.ini
        if not self._validate_config_ini():
            return False
        
        # 3. Validate all CSV files
        self._validate_signals_csv()
        self._validate_sensors_csv()
        self._validate_actuators_csv()
        self._validate_custom_callbacks_csv()
        
        # 4. Validate optional files
        self._validate_opc_files()
        self._validate_api_extensions_csv()
        
        # 5. Cross-validate relationships
        self._validate_signal_actuator_links()
        self._validate_callback_timepoints()
        
        # 6. Validate EnergyPlus files
        self._validate_energyplus_files()
        
        # 7. Print summary
        self._print_summary()
        
        return len(self.errors) == 0
    
    def _validate_working_directory(self) -> bool:
        """Validate working directory exists and is accessible"""
        logger.info("→ Validating working directory...")
        
        if not self.working_dir.exists():
            self.errors.append(f"Working directory does not exist: {self.working_dir}")
            logger.error(f"✗ Working directory not found: {self.working_dir}")
            return False
        
        if not self.working_dir.is_dir():
            self.errors.append(f"Path is not a directory: {self.working_dir}")
            logger.error(f"✗ Not a directory: {self.working_dir}")
            return False
        
        logger.info(f"✓ Working directory valid")
        self.validations_passed += 1
        return True
    
    def _validate_config_ini(self) -> bool:
        """Validate config.ini file"""
        logger.info("\n→ Validating config.ini...")
        self.files_checked += 1
        
        config_path = self.working_dir / "config.ini"
        
        # Check file exists
        if not config_path.exists():
            self.errors.append("config.ini not found in working directory")
            logger.error("✗ config.ini not found")
            self.validations_failed += 1
            return False
        
        # Load config
        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_path)
        except Exception as e:
            self.errors.append(f"Failed to parse config.ini: {e}")
            logger.error(f"✗ Failed to parse config.ini: {e}")
            self.validations_failed += 1
            return False
        
        # Validate required sections
        required_sections = ['DEFAULT', 'ENERGYPLUS', 'CONFIGURATIONFILES', 'DATABASE']
        for section in required_sections:
            if section not in self.config.sections() and section != 'DEFAULT':
                self.errors.append(f"Missing required section in config.ini: [{section}]")
                logger.error(f"✗ Missing section: [{section}]")
                self.validations_failed += 1
        
        # Validate DEFAULT section parameters
        default_params = ['DigitalTwinIdentifier', 'RunLength', 'WarmUpPeriodInDays']
        for param in default_params:
            if not self.config.get('DEFAULT', param, fallback=None):
                self.errors.append(f"Missing parameter in [DEFAULT]: {param}")
                logger.error(f"✗ Missing: [DEFAULT].{param}")
                self.validations_failed += 1
        
        # Validate ENERGYPLUS section
        ep_params = ['EnergyPlusDirectory', 'EPBuildingModel', 'EPWeatherFile']
        for param in ep_params:
            if not self.config.get('ENERGYPLUS', param, fallback=None):
                self.errors.append(f"Missing parameter in [ENERGYPLUS]: {param}")
                logger.error(f"✗ Missing: [ENERGYPLUS].{param}")
                self.validations_failed += 1
        
        # Validate CONFIGURATIONFILES section
        config_params = ['SignalsFile', 'SensorsFile', 'ActuatorsFile', 'CustomFile']
        for param in config_params:
            if not self.config.get('CONFIGURATIONFILES', param, fallback=None):
                self.errors.append(f"Missing parameter in [CONFIGURATIONFILES]: {param}")
                logger.error(f"✗ Missing: [CONFIGURATIONFILES].{param}")
                self.validations_failed += 1
        
        # Validate DATABASE section
        db_params = ['DatabaseName', 'DatabaseHost', 'DatabasePort', 'DatabaseUser']
        for param in db_params:
            if not self.config.get('DATABASE', param, fallback=None):
                self.errors.append(f"Missing parameter in [DATABASE]: {param}")
                logger.error(f"✗ Missing: [DATABASE].{param}")
                self.validations_failed += 1
        
        if len(self.errors) == 0:
            logger.info("✓ config.ini valid")
            self.validations_passed += 1
            return True
        else:
            return False
    
    def _validate_csv_file(
        self, 
        filename: str, 
        required_columns: List[str],
        optional_columns: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Generic CSV validation.
        
        Returns:
            DataFrame if valid, None if invalid
        """
        self.files_checked += 1
        
        csv_path = self.working_dir / filename
        
        # Check file exists
        if not csv_path.exists():
            self.errors.append(f"{filename} not found")
            logger.error(f"✗ {filename} not found")
            self.validations_failed += 1
            return None
        
        # Try to load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.errors.append(f"Failed to parse {filename}: {e}")
            logger.error(f"✗ Failed to parse {filename}: {e}")
            self.validations_failed += 1
            return None
        
        # Check for empty file
        if df.empty:
            self.warnings.append(f"{filename} is empty")
            logger.warning(f"⚠ {filename} is empty")
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.errors.append(f"{filename} missing required columns: {missing_cols}")
            logger.error(f"✗ {filename} missing columns: {missing_cols}")
            self.validations_failed += 1
            return None
        
        # Warn about unexpected columns
        if optional_columns is not None:
            expected_cols = set(required_columns + optional_columns)
            unexpected_cols = set(df.columns) - expected_cols
            if unexpected_cols:
                self.warnings.append(f"{filename} has unexpected columns: {unexpected_cols}")
                logger.warning(f"⚠ {filename} unexpected columns: {unexpected_cols}")
        
        logger.info(f"✓ {filename} valid ({len(df)} rows)")
        self.validations_passed += 1
        return df
    
    def _validate_signals_csv(self):
        """Validate signals.csv"""
        logger.info("\n→ Validating signals.csv...")
        
        filename = self.config.get('CONFIGURATIONFILES', 'SignalsFile', fallback='signals.csv')
        
        required_cols = ['SignalTagName', 'SignalSource']
        optional_cols = ['SourceId', 'current_val']
        
        df = self._validate_csv_file(filename, required_cols, optional_cols)
        if df is None:
            return
        
        # Validate signal sources
        valid_sources = {'seeq', 'opc', 'none'}
        invalid_sources = set(df['SignalSource'].str.lower().unique()) - valid_sources
        if invalid_sources:
            self.warnings.append(
                f"signals.csv contains unknown sources: {invalid_sources}. "
                f"Valid: {valid_sources}"
            )
            logger.warning(f"⚠ Unknown signal sources: {invalid_sources}")
        
        # Check for duplicate signal names
        duplicates = df[df.duplicated('SignalTagName', keep=False)]
        if not duplicates.empty:
            self.errors.append(
                f"signals.csv contains duplicate SignalTagName values: "
                f"{duplicates['SignalTagName'].tolist()}"
            )
            logger.error(f"✗ Duplicate signal names found")
            self.validations_failed += 1
        
        # Store for cross-validation
        self.signals_df = df
    
    def _validate_sensors_csv(self):
        """Validate sensors.csv"""
        logger.info("\n→ Validating sensors.csv...")
        
        filename = self.config.get('CONFIGURATIONFILES', 'SensorsFile', fallback='sensors.csv')
        
        required_cols = ['SensorName', 'SensorInstance', 'Type', 'PersistenceName', 'DataType', 'Read_stage']
        optional_cols = ['ep_handle', 'current_val', 'opc_tag_name']
        
        df = self._validate_csv_file(filename, required_cols, optional_cols)
        if df is None:
            return
        
        # Validate sensor types
        valid_types = {'sensor', 'meter'}
        invalid_types = set(df['Type'].unique()) - valid_types
        if invalid_types:
            self.errors.append(f"sensors.csv contains invalid Type values: {invalid_types}")
            logger.error(f"✗ Invalid sensor types: {invalid_types}")
            self.validations_failed += 1
        
        # Validate data types
        valid_datatypes = {'real', 'int', 'float', 'double'}
        invalid_datatypes = set(df['DataType'].str.lower().unique()) - valid_datatypes
        if invalid_datatypes:
            self.warnings.append(f"sensors.csv contains unusual DataType values: {invalid_datatypes}")
            logger.warning(f"⚠ Unusual data types: {invalid_datatypes}")
        
        # Check for duplicate persistence names
        duplicates = df[df.duplicated('PersistenceName', keep=False)]
        if not duplicates.empty:
            self.warnings.append(
                f"sensors.csv contains duplicate PersistenceName values (may cause DB conflicts)"
            )
            logger.warning(f"⚠ Duplicate persistence names found")
        
        # Store for cross-validation
        self.sensors_df = df
    
    def _validate_actuators_csv(self):
        """Validate actuators.csv"""
        logger.info("\n→ Validating actuators.csv...")
        
        filename = self.config.get('CONFIGURATIONFILES', 'ActuatorsFile', fallback='actuators.csv')
        
        required_cols = [
            'ActuatorCategory', 'ActuatorName', 'ActuatorInstance', 
            'SourceTagName', 'ConversionFunction', 'Override_stage'
        ]
        optional_cols = ['ep_handle', 'current_val']
        
        df = self._validate_csv_file(filename, required_cols, optional_cols)
        if df is None:
            return
        
        # Check conversion functions exist
        if hasattr(self, 'signals_df'):
            # Verify SourceTagName exists in signals
            missing_signals = set(df['SourceTagName']) - set(self.signals_df['SignalTagName'])
            if missing_signals:
                self.errors.append(
                    f"actuators.csv references signals not in signals.csv: {missing_signals}"
                )
                logger.error(f"✗ Missing signal references: {missing_signals}")
                self.validations_failed += 1
        
        # Store for cross-validation
        self.actuators_df = df
    
    def _validate_custom_callbacks_csv(self):
        """Validate custom_callbacks.csv"""
        logger.info("\n→ Validating custom_callbacks.csv...")
        
        filename = self.config.get('CONFIGURATIONFILES', 'CustomFile', fallback='custom_callbacks.csv')
        
        required_cols = ['Function', 'TimePeriod']
        optional_cols = []
        
        df = self._validate_csv_file(filename, required_cols, optional_cols)
        if df is None:
            return
        
        # Store for cross-validation
        self.custom_callbacks_df = df
    
    def _validate_opc_files(self):
        """Validate OPC-UA configuration files if enabled"""
        opc_enabled = self.config.get('OPCSERVER', 'OpcServerEnabled', fallback='false').lower()
        
        if opc_enabled != 'true':
            logger.info("\n→ OPC-UA not enabled, skipping OPC validation")
            return
        
        logger.info("\n→ Validating OPC-UA configuration...")
        
        # Validate OPC devices file
        opc_devices_file = self.config.get('CONFIGURATIONFILES', 'OpcDevicesFile', fallback=None)
        if opc_devices_file:
            required_cols = ['device_name', 'description', 'device_type']
            self._validate_csv_file(opc_devices_file, required_cols)
        else:
            self.warnings.append("OPC enabled but OpcDevicesFile not configured")
            logger.warning("⚠ OPC enabled but OpcDevicesFile missing")
        
        # Validate OPC variables file
        opc_vars_file = self.config.get('CONFIGURATIONFILES', 'OpcVariablesFile', fallback=None)
        if opc_vars_file:
            required_cols = ['device_name', 'var_name', 'tag_name', 'description', 'data_type', 'ep_type']
            optional_cols = ['unit', 'opc_tag_name', 'source_signal', 'ml_model_path', 'ml_input_signals', 'predictor_function']
            self._validate_csv_file(opc_vars_file, required_cols, optional_cols)
        else:
            self.warnings.append("OPC enabled but OpcVariablesFile not configured")
            logger.warning("⚠ OPC enabled but OpcVariablesFile missing")
    
    def _validate_api_extensions_csv(self):
        """Validate API extensions file if present"""
        logger.info("\n→ Checking for API extensions...")
        
        extensions_file = self.config.get('CONFIGURATIONFILES', 'ApiExtensionsFile', fallback=None)
        
        if not extensions_file:
            logger.info("  No API extensions configured (optional)")
            return
        
        required_cols = ['ExtensionName', 'ExtensionType', 'Function', 'TimePeriod']
        optional_cols = ['Description']
        
        df = self._validate_csv_file(extensions_file, required_cols, optional_cols)
        if df is not None:
            self.api_extensions_df = df
    
    def _validate_signal_actuator_links(self):
        """Cross-validate signals and actuators are properly linked"""
        logger.info("\n→ Cross-validating signal-actuator links...")
        
        if not hasattr(self, 'signals_df') or not hasattr(self, 'actuators_df'):
            logger.warning("⚠ Cannot cross-validate (files not loaded)")
            return
        
        # Check all actuator sources exist in signals
        missing = set(self.actuators_df['SourceTagName']) - set(self.signals_df['SignalTagName'])
        if missing:
            self.errors.append(
                f"Actuators reference non-existent signals: {missing}"
            )
            logger.error(f"✗ Broken signal-actuator links: {missing}")
            self.validations_failed += 1
        else:
            logger.info("✓ All actuator signals valid")
            self.validations_passed += 1
    
    def _validate_callback_timepoints(self):
        """Validate callback timepoints are valid"""
        logger.info("\n→ Validating callback timepoints...")
        
        # Valid EnergyPlus callback names
        valid_callbacks = {
            'begin_new_environment',
            'after_component_get_input',
            'after_new_environment_warmup_complete',
            'begin_zone_timestep_before_init_heat_balance',
            'begin_zone_timestep_after_init_heat_balance',
            'begin_zone_timestep_before_set_current_weather',
            'after_predictor_before_hvac_managers',
            'after_predictor_after_hvac_managers',
            'begin_system_timestep_before_predictor',
            'end_system_sizing',
            'end_system_timestep_after_hvac_reporting',
            'end_system_timestep_before_hvac_reporting',
            'end_zone_sizing',
            'end_zone_timestep_before_zone_reporting',
            'end_zone_timestep_after_zone_reporting',
            'inside_system_iteration_loop',
            'message',
            'progress',
            'unitary_system_sizing'
        }
        
        invalid_found = False
        
        # Check sensors
        if hasattr(self, 'sensors_df'):
            invalid = set(self.sensors_df['Read_stage'].unique()) - valid_callbacks
            if invalid:
                self.errors.append(f"sensors.csv has invalid Read_stage values: {invalid}")
                logger.error(f"✗ Invalid sensor Read_stage: {invalid}")
                invalid_found = True
        
        # Check actuators
        if hasattr(self, 'actuators_df'):
            invalid = set(self.actuators_df['Override_stage'].unique()) - valid_callbacks
            if invalid:
                self.errors.append(f"actuators.csv has invalid Override_stage values: {invalid}")
                logger.error(f"✗ Invalid actuator Override_stage: {invalid}")
                invalid_found = True
        
        # Check custom callbacks
        if hasattr(self, 'custom_callbacks_df'):
            invalid = set(self.custom_callbacks_df['TimePeriod'].unique()) - valid_callbacks
            if invalid:
                self.errors.append(f"custom_callbacks.csv has invalid TimePeriod values: {invalid}")
                logger.error(f"✗ Invalid custom TimePeriod: {invalid}")
                invalid_found = True
        
        # Check API extensions if present
        if hasattr(self, 'api_extensions_df'):
            invalid = set(self.api_extensions_df['TimePeriod'].unique()) - valid_callbacks
            if invalid:
                self.errors.append(f"api_extensions.csv has invalid TimePeriod values: {invalid}")
                logger.error(f"✗ Invalid extension TimePeriod: {invalid}")
                invalid_found = True
        
        if invalid_found:
            self.validations_failed += 1
        else:
            logger.info("✓ All callback timepoints valid")
            self.validations_passed += 1
    
    def _validate_energyplus_files(self):
        """Validate EnergyPlus related files exist"""
        logger.info("\n→ Validating EnergyPlus files...")
        
        # Check building model
        building_model = self.config.get('ENERGYPLUS', 'EPBuildingModel', fallback=None)
        if building_model:
            model_path = self.working_dir / building_model
            if not model_path.exists():
                self.errors.append(f"EnergyPlus building model not found: {building_model}")
                logger.error(f"✗ Building model not found: {building_model}")
                self.validations_failed += 1
            elif not model_path.suffix.lower() == '.idf':
                self.warnings.append(f"Building model has unusual extension: {building_model}")
                logger.warning(f"⚠ Expected .idf file, got: {building_model}")
            else:
                logger.info(f"✓ Building model found: {building_model}")
                self.validations_passed += 1
        
        # Check weather file
        weather_file = self.config.get('ENERGYPLUS', 'EPWeatherFile', fallback=None)
        if weather_file:
            weather_path = Path(weather_file)
            if not weather_path.is_absolute():
                weather_path = self.working_dir / weather_file
            
            if not weather_path.exists():
                self.errors.append(f"Weather file not found: {weather_file}")
                logger.error(f"✗ Weather file not found: {weather_file}")
                self.validations_failed += 1
            elif not weather_path.suffix.lower() == '.epw':
                self.warnings.append(f"Weather file has unusual extension: {weather_file}")
                logger.warning(f"⚠ Expected .epw file, got: {weather_file}")
            else:
                logger.info(f"✓ Weather file found: {weather_file}")
                self.validations_passed += 1
        
        # Check EnergyPlus directory
        ep_dir = self.config.get('ENERGYPLUS', 'EnergyPlusDirectory', fallback=None)
        if ep_dir:
            ep_path = Path(ep_dir)
            if not ep_path.exists():
                self.warnings.append(f"EnergyPlus directory not found: {ep_dir}")
                logger.warning(f"⚠ EnergyPlus directory not found: {ep_dir}")
            else:
                logger.info(f"✓ EnergyPlus directory found: {ep_dir}")
                self.validations_passed += 1
    
    def _print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Files checked: {self.files_checked}")
        logger.info(f"Validations passed: {self.validations_passed}")
        logger.info(f"Validations failed: {self.validations_failed}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info(f"Errors: {len(self.errors)}")
        
        if self.warnings:
            logger.info("\n⚠ WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        if self.errors:
            logger.info("\n✗ ERRORS:")
            for error in self.errors:
                logger.error(f"  • {error}")
        
        logger.info("\n" + "=" * 70)
        
        if len(self.errors) == 0:
            logger.info("✓ ALL VALIDATIONS PASSED")
            if len(self.warnings) > 0:
                logger.info(f"  ({len(self.warnings)} warning(s) - review recommended)")
        else:
            logger.error("✗ VALIDATION FAILED")
            logger.error("  Fix errors before running simulation")
        
        logger.info("=" * 70)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <working_directory>")
        print("\nExample:")
        print("  python validate_config.py /path/to/digital_twin/config")
        sys.exit(1)
    
    working_dir = sys.argv[1]
    
    validator = ConfigValidator(working_dir)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()