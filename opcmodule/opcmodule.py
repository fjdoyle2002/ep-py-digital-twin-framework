"""
OPC-UA Module for Digital Twin

Features:
- Publishes digital twin sensor data as OPC-UA variables
- Allows OPC-UA clients to write actuator values
- Supports custom predictor functions for computed values
- Proper async/sync threading with event loop safety
- Enum-based metadata exposure for SCADA integration

Thread Safety:
- Uses threading.Event for startup synchronization
- Uses asyncio.run_coroutine_threadsafe for cross-thread calls
- Stores event loop reference for safe main thread access
"""
import threading
import os
import pandas as pd
import time
import asyncio
from datetime import datetime, timezone
from asyncua import ua, Server
from asyncua.common.structures104 import new_enum
from opcmodule.opc_device import OPCDevice
import logging
from typing import Dict, Optional

# Import predictor functions via reflection
import custom.opc_predictor as predictor_module

logger = logging.getLogger(__name__)


class OPCUAModule:
    """
    OPC-UA Server for Digital Twin Integration
    
    Publishes:
    - EnergyPlus sensor values (from sensors_df)
    - Predictor function outputs (computed values)
    
    Accepts:
    - Actuator value writes from OPC-UA clients
    """
    
    def __init__(self, working_directory, config):
        self.working_directory = working_directory
        self.config = config
        self._logger = logging.getLogger(__name__)

        # Server components
        self.server = None
        self.uri = None
        self.namespace = None
        self.should_run = True
        
        # Devices and variables
        self.devices = []
        self.tagmap = {}  # tag_name -> OPC variable node
        self.actuator_map = {}  # tag_name -> signals_df signal name
        self.predictors = {}  # tag_name -> predictor function reference
        
        # Thread synchronization (FIXED: proper startup sync)
        self._server_ready = threading.Event()
        self._server_loop = None  # FIXED: store event loop reference
        self._server_thread = None
        
        # Reference to sensors dataframe (updated by digital twin)
        self.sensors_df_reference = None
        self.sensors_updated = False
        
        # Predictor context (for caching models, state)
        self.predictor_context = {}
        
        # Lock for thread-safe access to shared data
        self.data_lock = threading.Lock()
        
        # Load OPC device and variable configurations
        self._load_configurations()
        
        # Initialize predictors (CHANGED: now generic, not ML-specific)
        self._initialize_predictors()
        
        self._logger.info('OPC-UA module initialized')
    
    def _load_configurations(self):
        """Load OPC device and variable configurations from CSV files"""
        devices_path = os.path.join(
            self.working_directory, 
            self.config.get('CONFIGURATIONFILES', 'OpcDevicesFile')
        )
        self.opc_devices_df = pd.read_csv(devices_path)

        variables_path = os.path.join(
            self.working_directory, 
            self.config.get('CONFIGURATIONFILES', 'OpcVariablesFile')
        )
        self.opc_variables_df = pd.read_csv(variables_path)
        self.opc_variables_df = self.opc_variables_df.assign(current_val=-1)
        
        # Instantiate all OPC devices (preserving existing structure)
        for idx in self.opc_devices_df.index:
            curr_device_name = self.opc_devices_df['device_name'][idx]
            curr_device_description = self.opc_devices_df['description'][idx]
            curr_device_type = self.opc_devices_df['device_type'][idx]
            
            # Get variables for this device
            device_variables_df = self.opc_variables_df[
                self.opc_variables_df['device_name'] == curr_device_name
            ]
            
            device = OPCDevice(
                curr_device_name, 
                curr_device_description, 
                curr_device_type, 
                device_variables_df
            )
            
            self.devices.append(device)
        
        self._logger.info(f'Loaded {len(self.devices)} OPC devices')
    
    def _initialize_predictors(self):
        """
        Initialize predictor functions for variables that specify predictor_function.
        
        CHANGED: This is now generic - just stores function references.
        All ML model loading logic belongs in custom/opc_predictor.py.
        """
        if 'predictor_function' not in self.opc_variables_df.columns:
            self._logger.info("No predictor_function column in configuration")
            return
        
        predictor_vars = self.opc_variables_df[
            self.opc_variables_df['predictor_function'].notna()
        ]
        
        for idx in predictor_vars.index:
            tag_name = predictor_vars['tag_name'][idx]
            func_name = predictor_vars['predictor_function'][idx]
            
            # Verify function exists in predictor module
            if not hasattr(predictor_module, func_name):
                self._logger.error(
                    f"Predictor function '{func_name}' not found in "
                    f"custom.opc_predictor for tag '{tag_name}'"
                )
                continue
            
            # Store function reference (generic - no ML-specific code here)
            self.predictors[tag_name] = getattr(predictor_module, func_name)
            
            self._logger.info(
                f"Registered predictor function '{func_name}' for '{tag_name}'"
            )
    
    async def add_variables_to_devices(self):
        """Register all device nodes and variables with the OPC-UA server"""
        for curr_device in self.devices:
            await curr_device.add_variables(
                self.server, 
                self.namespace, 
                self.uri, 
                self.tagmap
            )
        
        # Build actuator map for writeable variables
        self._build_actuator_map()
        
        self._logger.info('All OPC device nodes and variables registered')
    
    def _build_actuator_map(self):
        """
        Build mapping of writeable OPC variables to their source in signals_df.
        This allows OPC writes to update actuator values.
        """
        for device in self.devices:
            for tag_name in device.actuators:
                # Find this tag in opc_variables_df to get source signal
                var_row = self.opc_variables_df[
                    self.opc_variables_df['tag_name'] == tag_name
                ]
                
                if not var_row.empty and 'source_signal' in var_row.columns:
                    source_signal = var_row['source_signal'].iloc[0]
                    if pd.notna(source_signal):
                        self.actuator_map[tag_name] = source_signal
                        self._logger.info(
                            f"Mapped actuator '{tag_name}' -> '{source_signal}'"
                        )
    
    def update_variables(self, sensors_df):
        """
        Called by digital twin to provide updated sensor data.
        Thread-safe update of sensor reference.
        
        Args:
            sensors_df: DataFrame with current sensor values
        """
        with self.data_lock:
            self.sensors_df_reference = sensors_df.copy()
            self.sensors_updated = True
    
    def retrieve_signals_for_actuators_at_timepoint(self, signals_df, timepoint):
        """
        Retrieves actuator values that have been written to the OPC-UA server
        by external clients and updates the signals_df.
        
        This is called by the digital twin's retrieval system to get OPC values.
        
        FIXED: Now uses thread-safe asyncio.run_coroutine_threadsafe to avoid
        event loop conflicts.
        
        Args:
            signals_df: DataFrame to update with OPC values
            timepoint: Current simulation time (not used for OPC, values are live)
        """
        # FIXED: Check if server is ready
        if not self._server_ready.is_set():
            self._logger.warning("OPC server not ready, skipping retrieval")
            return
        
        if self._server_loop is None:
            self._logger.error("OPC server loop not available")
            return
        
        for idx in signals_df.index:
            # Only process signals that come from OPC source
            if signals_df['SignalSource'][idx].lower() == 'opc':
                signal_name = signals_df['SignalTagName'][idx]
                
                try:
                    # FIXED: Find OPC tag name from signal name using actuator_map
                    # The actuator_map links: OPC tag name -> signal name
                    # We need to reverse lookup: signal name -> OPC tag name
                    opc_tag_name = None
                    for opc_tag, mapped_signal in self.actuator_map.items():
                        if mapped_signal == signal_name:
                            opc_tag_name = opc_tag
                            break
                    
                    if not opc_tag_name:
                        self._logger.warning(
                            f"Signal '{signal_name}' not found in actuator_map. "
                            f"Available mappings: {self.actuator_map}"
                        )
                        continue
                    
                    # Get the OPC variable from tagmap using OPC tag name
                    if opc_tag_name not in self.tagmap:
                        self._logger.warning(
                            f"OPC tag '{opc_tag_name}' (for signal '{signal_name}') "
                            f"not found in tagmap"
                        )
                        continue
                    
                    opc_var = self.tagmap[opc_tag_name]
                    
                    # FIXED: Use thread-safe coroutine execution
                    # This submits the coroutine to the OPC server's event loop
                    future = asyncio.run_coroutine_threadsafe(
                        opc_var.read_value(),
                        self._server_loop
                    )
                    
                    # Wait for result with timeout
                    curr_signal_value = future.result(timeout=5.0)
                    
                    # Validate the value
                    if curr_signal_value is not None:
                        signals_df.iloc[
                            idx, 
                            signals_df.columns.get_loc('current_val')
                        ] = curr_signal_value
                        
                        self._logger.debug(
                            f"Retrieved OPC value for signal '{signal_name}' "
                            f"(OPC tag '{opc_tag_name}'): {curr_signal_value}"
                        )
                    else:
                        self._logger.warning(
                            f"Signal '{signal_name}' returned None at "
                            f"time {timepoint}, retaining last valid value"
                        )
                        
                except asyncio.TimeoutError:
                    self._logger.error(
                        f"Timeout reading OPC signal '{signal_name}'"
                    )
                except Exception as e:
                    self._logger.error(
                        f"Error reading OPC signal '{signal_name}': {e}",
                        exc_info=True
                    )
    
    def _compute_predictor_values(self, sensors_df) -> Dict[str, float]:
        """
        Compute predictor values by calling their functions.
        
        CHANGED: This is now completely generic. All ML-specific logic
        lives in custom/opc_predictor.py functions.
        
        Args:
            sensors_df: Current sensor data
            
        Returns:
            Dictionary mapping tag_name -> predicted value
        """
        predictions = {}
        
        for tag_name, predictor_func in self.predictors.items():
            try:
                # Call predictor function (generic - works for any function)
                # The function handles its own ML loading, calculations, etc.
                value = predictor_func(self.config, sensors_df, self.predictor_context)
                predictions[tag_name] = float(value)
                
                self._logger.debug(
                    f"Predictor for '{tag_name}': {value}"
                )
                
            except Exception as e:
                self._logger.error(
                    f"Error computing prediction for '{tag_name}': {e}",
                    exc_info=True
                )
                predictions[tag_name] = 0.0  # Default on error
        
        return predictions
    
    async def core(self):
        """
        Core asynchronous method to set up and run the OPC UA server.
        
        Main server loop that:
        1. Publishes sensor values from EnergyPlus
        2. Publishes predictor function outputs
        3. Handles client writes to actuator variables
        
        FIXED: Now stores event loop reference and signals readiness properly.
        """
        try:
            # FIXED: Store the event loop for thread-safe coroutine submission
            self._server_loop = asyncio.get_running_loop()
            self._logger.info("OPC server event loop initialized")
            
            # Setup server
            self.server = Server()
            await self.server.init()
            
            endpoint = self.config.get('OPCSERVER', 'ep')
            server_name = self.config.get('OPCSERVER', 'OpcServerName')
            
            self.server.set_endpoint(endpoint)
            self.server.set_server_name(server_name)
            
            self._logger.info(f"OPC-UA server endpoint: {endpoint}")
            
            # Setup namespace
            uri = self.config.get('OPCSERVER', 'uri')
            self.uri = uri
            self.namespace = await self.server.register_namespace(uri)
            
            # Add all device nodes and variables
            await self.add_variables_to_devices()
            
            self._logger.info('OPC-UA server configured, starting...')
            
            # FIXED: Signal readiness BEFORE entering main loop
            self._server_ready.set()
            
            # Main server loop
            async with self.server:
                update_interval = float(
                    self.config.get('OPCSERVER', 'UpdateIntervalSeconds', fallback='10')
                )
                
                self._logger.info(
                    f'OPC-UA server running (update interval: {update_interval}s)'
                )
                
                while self.should_run:
                    try:
                        # Check if we have updated sensor data
                        with self.data_lock:
                            sensors_updated = self.sensors_updated
                            sensors_df = self.sensors_df_reference
                            self.sensors_updated = False
                        
                        if sensors_updated and sensors_df is not None:
                            # Compute predictor values (generic call)
                            predictor_values = {}
                            if self.predictors:
                                predictor_values = self._compute_predictor_values(sensors_df)
                            # Update all devices with latest values
                            for device in self.devices:
                                await device.publish_variables(
                                    sensors_df, 
                                    predictor_values
                                )
                        
                        await asyncio.sleep(update_interval)
                        
                    except Exception as e:
                        self._logger.error(
                            f"Exception in OPC server loop: {e}",
                            exc_info=True
                        )
                        await asyncio.sleep(update_interval)
                        
        except Exception as e:
            self._logger.error(
                f"Fatal error in OPC server core: {e}",
                exc_info=True
            )
        finally:
            self._logger.info("OPC server core shutting down")
            self._server_loop = None
    
    def main(self):
        """Main method to start the asyncio event loop and run the core server logic"""
        try:
            asyncio.run(self.core())
        except Exception as e:
            self._logger.error(f"Fatal error in OPC server: {e}", exc_info=True)
    
    def start(self):
        """
        Start the OPC UA server in a separate thread.
        
        FIXED: Now waits for server to be ready with proper synchronization.
        """
        self._logger.info("Starting OPC-UA server thread")
        
        # Clear the ready event in case of restart
        self._server_ready.clear()
        
        # Start server in background thread
        self._server_thread = threading.Thread(target=self.main, daemon=True)
        self._server_thread.start()
        
        # Wait for server to be ready with timeout
        self._logger.info("Waiting for OPC-UA server to initialize...")
        if not self._server_ready.wait(timeout=30):
            raise RuntimeError("OPC-UA server failed to start within 30 seconds")
        
        self._logger.info("✓ OPC-UA server thread started and ready")
    
    def stop(self):
        """
        Stop the OPC-UA server gracefully.
        
        FIXED: Now properly waits for thread to finish.
        """
        self._logger.info("Stopping OPC-UA server")
        self.should_run = False
        
        # Clear ready flag
        self._server_ready.clear()
        
        # Give the server loop time to exit cleanly
        if self._server_thread and self._server_thread.is_alive():
            self._logger.info("Waiting for OPC-UA server thread to stop...")
            self._server_thread.join(timeout=10)
            
            if self._server_thread.is_alive():
                self._logger.warning("OPC-UA server thread did not stop cleanly")
            else:
                self._logger.info("✓ OPC-UA server stopped")