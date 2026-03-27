"""
OPC Device with Dynamic Units Support

This version retrieves units from sensors_df (populated from .rdd file)
rather than requiring them in the CSV configuration.

Key changes:
- Adds 'units' property to each OPC variable
- Updates units dynamically from sensors_df during publish_variables()
- Falls back to CSV units if sensors_df doesn't have unit data
"""
import pandas as pd
import datetime as dt
from datetime import datetime, timezone
from asyncua import ua, Server
from asyncua.common.structures104 import new_enum
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OPCDevice:
    """
    Represents a logical device in the OPC-UA server.
    
    A device groups related variables and can represent:
    - A physical building zone
    - An HVAC system
    - A virtual sensor array
    - An ML model output collection
    """
    
    def __init__(self, device_name, device_description, device_type, variables_df):
        self.device_name = device_name
        self.description = device_description
        self.device_type = device_type
        self.variables_df = variables_df.reset_index(drop=True)
        self.variables_map = {}
        self._logger = logging.getLogger(__name__)
        
        self.server = None
        self.namespace = None
        self.uri = None
        self.node = None
        self.variables = {}
        self.actuators = set()  # Tags that are writeable
        self.ml_variables = set()  # Tags provided by predictors
        self.variable_types = {}
        
        # NEW: Store units property nodes for dynamic updates
        self.units_properties = {}  # tag_name -> units property node
    
    async def register_node(self):
        """Register this device as an object in the OPC-UA server"""
        self.node = await self.server.nodes.objects.add_object(
            self.namespace, 
            self.device_name
        )
        
        self._logger.info(f"Registered OPC device node: {self.device_name}")
        self._logger.debug(f"  NodeId: {self.node.nodeid}")
        
        browse_name = await self.node.read_browse_name()
        self._logger.debug(
            f"  BrowseName: {browse_name.Name} (ns={browse_name.NamespaceIndex})"
        )
    
    def resolve_pandas_dtype_to_opc(self, data_type):
        """
        Convert pandas/python data type to OPC-UA VariantType.
        
        Returns:
            Tuple of (VariantType, default_value)
        """
        type_map = {
            'float64': (ua.VariantType.Double, 0.0),
            'float32': (ua.VariantType.Float, 0.0),
            'float': (ua.VariantType.Double, 0.0),
            'int64': (ua.VariantType.Int64, 0),
            'int32': (ua.VariantType.Int32, 0),
            'int': (ua.VariantType.Int64, 0),
            'object': (ua.VariantType.String, ""),
            'string': (ua.VariantType.String, ""),
            'bool': (ua.VariantType.Boolean, False),
            'boolean': (ua.VariantType.Boolean, False),
        }
        
        result = type_map.get(str(data_type).lower(), (ua.VariantType.String, ""))
        
        self._logger.debug(
            f"Resolved data type '{data_type}' to OPC type '{result[0]}'"
        )
        
        return result
    
    def create_metadata_list(self, idx):
        """
        Create metadata list for a variable (used in enum for metadata).

        Each entry becomes an enum member name in asyncua, so it must be a
        valid Python identifier. We sanitize values by:
          - Replacing NaN / empty with a column-based placeholder
          - Replacing any non-alphanumeric character with underscore
          - Stripping leading digits / underscores
          - Ensuring uniqueness by appending a counter when needed
        """
        import re

        meta_list = []

        exclude_cols = ['device_name', 'current_val']
        subframe_df = self.variables_df.drop(
            columns=[c for c in exclude_cols if c in self.variables_df.columns],
            errors='ignore'
        )

        seen = set()
        for col_pos, column in enumerate(subframe_df.columns):
            value = subframe_df[column].iloc[idx]

            # Convert to string; use column name as fallback for blank/NaN
            raw = str(value) if pd.notna(value) and str(value).strip() else f"col_{column}"

            # Replace every non-alphanumeric character with underscore
            sanitized = re.sub(r'[^A-Za-z0-9]', '_', raw)

            # Strip leading underscores/digits so it starts with a letter
            sanitized = re.sub(r'^[^A-Za-z]+', '', sanitized)

            # Final fallback if nothing remains after stripping
            if not sanitized:
                sanitized = f"col_{col_pos}"

            # Guarantee uniqueness within this enum
            original = sanitized
            counter = 1
            while sanitized in seen:
                sanitized = f"{original}_{counter}"
                counter += 1

            seen.add(sanitized)
            meta_list.append(sanitized)

        return meta_list
    
    async def add_variables(self, server, namespace, uri, module_tagmap):
        """
        Add all variables for this device to the OPC-UA server.
        
        Args:
            server: asyncua Server object
            namespace: Namespace index for this server
            uri: Namespace URI for this server
            module_tagmap: Module-level dict mapping tag_name -> OPC variable node
        """
        self._logger.info(f"Adding variables for device '{self.device_name}'")
        
        self.server = server
        self.namespace = namespace
        self.uri = uri
        
        # Register the device node
        await self.register_node()
        
        # Add each variable
        for idx in self.variables_df.index:
            await self._add_single_variable(idx, module_tagmap)
        
        self._logger.info(
            f"Device '{self.device_name}': Added {len(self.variables)} variables "
            f"({len(self.actuators)} actuators, {len(self.ml_variables)} predictors)"
        )
    
    async def _add_single_variable(self, idx, module_tagmap):
        """Add a single variable to the OPC-UA server"""
        try:
            var_name = self.variables_df['var_name'].iloc[idx]
            tag_name = self.variables_df['tag_name'].iloc[idx]
            tag_desc = self.variables_df['description'].iloc[idx]
            
            # Get ep_type if present
            if 'ep_type' in self.variables_df.columns:
                ep_type = self.variables_df['ep_type'].iloc[idx]
            else:
                ep_type = None
            
            # Check for predictor function
            has_predictor = False
            if 'predictor_function' in self.variables_df.columns:
                predictor_func = self.variables_df['predictor_function'].iloc[idx]
                has_predictor = pd.notna(predictor_func)
            
            # Determine data type
            if 'data_type' in self.variables_df.columns:
                curr_dtype = self.variables_df['data_type'].iloc[idx]
            else:
                curr_dtype = 'float64'  # Default
            
            res_dtype = self.resolve_pandas_dtype_to_opc(curr_dtype)
            
            # Create OPC NodeId
            node_id_str = f'ns={self.namespace};s={tag_name}'
            node_id = ua.NodeId.from_string(node_id_str)
            
            self._logger.debug(f"Creating variable '{tag_name}' with NodeId: {node_id}")
            
            # Add variable to device node
            curr_var = await self.node.add_variable(
                node_id,
                tag_desc,
                res_dtype[1],
                varianttype=res_dtype[0]
            )
            
            # Configure writability and tracking
            if ep_type == 'actuator':
                # Actuators are writeable by OPC clients
                self.actuators.add(tag_name)
                await curr_var.set_writable(True)
                self._logger.debug(f"  Variable '{tag_name}' set as writeable actuator")
            
            elif has_predictor:
                # Predictor outputs are not writeable
                self.ml_variables.add(tag_name)
                await curr_var.set_writable(False)
                self._logger.debug(f"  Variable '{tag_name}' is predictor output")
            
            else:
                # Regular sensors are not writeable
                await curr_var.set_writable(False)
            
            # Store variable references
            self.variables_map[tag_name] = curr_var
            self.variables[tag_name] = curr_var
            self.variable_types[tag_name] = res_dtype[0]
            module_tagmap[tag_name] = curr_var
            
            # Add metadata enum
            await self._add_variable_metadata(curr_var, idx, tag_name, var_name)
            
            # NEW: Add dynamic units property
            await self._add_units_property(curr_var, tag_name, idx)
            
        except Exception as e:
            self._logger.error(
                f"Failed to add variable at index {idx}: {e}",
                exc_info=True
            )
    
    async def _add_variable_metadata(self, curr_var, idx, tag_name, var_name):
        """Add metadata properties to an OPC-UA variable"""
        try:
            # Create metadata enum
            meta_list = self.create_metadata_list(idx)
            enode = await new_enum(
                self.server, 
                self.namespace, 
                f"MetaEnum_{tag_name}", 
                meta_list
            )
            
            # Load custom data types
            await self.server.load_data_type_definitions()
            
            # Add metadata as child variable
            await curr_var.add_variable(
                self.namespace, 
                "metadata", 
                0, 
                datatype=enode.nodeid
            )
            
            # Set display name
            display_dv = ua.DataValue(ua.LocalizedText(var_name))
            await curr_var.write_attribute(ua.AttributeIds.DisplayName, display_dv)
            
            # Set description
            desc_dv = ua.DataValue(ua.LocalizedText(str(tag_name)))
            await curr_var.write_attribute(ua.AttributeIds.Description, desc_dv)
            
        except Exception as e:
            self._logger.warning(
                f"Could not add metadata for '{tag_name}': {e}"
            )
    
    async def _add_units_property(self, curr_var, tag_name, idx):
        """
        Add a dynamic 'units' property to the variable.
        
        This will be updated from sensors_df during publish_variables().
        Falls back to CSV units if available.
        
        Args:
            curr_var: OPC-UA variable node
            tag_name: Tag name for this variable
            idx: Index in variables_df
        """
        try:
            # Get initial units from CSV if available
            initial_units = ""
            if 'units' in self.variables_df.columns:
                csv_units = self.variables_df['units'].iloc[idx]
                if pd.notna(csv_units):
                    initial_units = str(csv_units)
            
            # Add units as a property (child variable)
            units_node = await curr_var.add_property(
                self.namespace,
                "EngineeringUnits",
                initial_units,
                varianttype=ua.VariantType.String
            )
            
            # Store reference for later updates
            self.units_properties[tag_name] = units_node
            
            self._logger.debug(
                f"Added units property for '{tag_name}' "
                f"(initial: '{initial_units}')"
            )
            
        except Exception as e:
            self._logger.warning(
                f"Could not add units property for '{tag_name}': {e}"
            )
    
    async def publish_variables(self, sensors_df, predictor_values: Optional[Dict[str, float]] = None):
        """
        Publish current values to all OPC-UA variables in this device.
        
        UPDATED: Now also updates units from sensors_df if available.
        
        Args:
            sensors_df: DataFrame containing latest sensor values from EnergyPlus
            predictor_values: Optional dict of tag_name -> predicted_value from predictors
        """
        timestamp = datetime.now(timezone.utc)
        predictor_values = predictor_values or {}
        
        for tag_name, curr_variable in self.variables.items():
            try:
                # Skip actuators - they are written by OPC clients, not the twin
                if tag_name in self.actuators:
                    continue
                
                # Determine value source
                if tag_name in self.ml_variables:
                    # Get value from predictor
                    if tag_name in predictor_values:
                        curr_value = predictor_values[tag_name]
                        # Predictor values don't have units in sensors_df
                        curr_units = None
                    else:
                        self._logger.warning(
                            f"No predictor value available for '{tag_name}'"
                        )
                        continue
                else:
                    # Get value from sensors_df
                    mask = sensors_df['opc_tag_name'] == tag_name
                    if not mask.any():
                        self._logger.debug(
                            f"Tag '{tag_name}' not found in sensors_df"
                        )
                        continue
                    
                    curr_value = sensors_df.loc[mask, 'current_val'].values[0]
                    
                    # NEW: Get units from sensors_df if available
                    if 'unit' in sensors_df.columns:
                        curr_units = sensors_df.loc[mask, 'unit'].values[0]
                        # Handle NaN units
                        if pd.isna(curr_units):
                            curr_units = None
                    else:
                        curr_units = None
                
                # Validate value
                if pd.isna(curr_value):
                    self._logger.warning(
                        f"NaN value for '{tag_name}', skipping update"
                    )
                    continue
                
                # Get OPC type and create DataValue
                curr_ua_type = self.variable_types[tag_name]
                curr_ua_dvalue = ua.DataValue(
                    ua.Variant(curr_value, curr_ua_type),
                    ServerTimestamp=timestamp,
                    SourceTimestamp=timestamp
                )
                
                # Write value to OPC variable
                await curr_variable.write_value(curr_ua_dvalue)
                
                # NEW: Update units property if we have units data
                if curr_units is not None and tag_name in self.units_properties:
                    await self._update_units_property(tag_name, curr_units)
                
                self._logger.debug(
                    f"Published '{tag_name}' = {curr_value} {curr_units or ''}"
                )
                
            except Exception as e:
                self._logger.error(
                    f"Error publishing variable '{tag_name}': {e}",
                    exc_info=True
                )
    
    async def _update_units_property(self, tag_name: str, units: str):
        """
        Update the units property for a variable.
        
        Args:
            tag_name: Tag name of the variable
            units: Units string from sensors_df
        """
        try:
            if tag_name not in self.units_properties:
                return
            
            units_node = self.units_properties[tag_name]
            
            # Only update if units changed (to reduce OPC traffic)
            current_units = await units_node.read_value()
            if current_units != units:
                await units_node.write_value(units)
                self._logger.debug(
                    f"Updated units for '{tag_name}': '{current_units}' -> '{units}'"
                )
                
        except Exception as e:
            self._logger.debug(
                f"Could not update units for '{tag_name}': {e}"
            )