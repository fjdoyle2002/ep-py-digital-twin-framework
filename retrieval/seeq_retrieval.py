# -*- coding: utf-8 -*-
"""
Seeq Retrieval Module

Retrieves real-time and historical data from Seeq historian/analytics platform
for use in digital twin simulations.

Seeq (https://www.seeq.com/) is an advanced analytics platform for process
manufacturing data. This module connects to Seeq to retrieve sensor values
at specified timepoints.

Created on Mon Jun 26 14:43:38 2023
Refactored: February 2026

@author: doylef
"""

from seeq import spy
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class SeeqRetrieval:
    """
    Seeq data retrieval agent for digital twin.
    
    Connects to Seeq platform and retrieves sensor values at specified
    timepoints for use in EnergyPlus simulation.
    
    Features:
    - Automatic Seeq item lookup by Datasource ID and Data ID
    - Retry logic for failed retrievals
    - Graceful handling of missing/NaN values
    - Caching of Seeq item metadata
    - Connection health monitoring
    
    Attributes:
        config: Configuration object with Seeq connection parameters
        signals_df: DataFrame of signals to retrieve from Seeq
        items: DataFrame of Seeq item metadata
        connection_healthy: Flag indicating Seeq connection status
    """
    
    # Configuration for retry behavior
    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 1
    DEFAULT_GRID_INTERVAL = "1min"
    
    def __init__(self, config, signals_df: pd.DataFrame):
        """
        Initialize Seeq retrieval agent.
        
        Establishes connection to Seeq platform and caches item metadata
        for all signals that need to be retrieved from Seeq.
        
        Args:
            config: ConfigParser with Seeq connection parameters:
                - SeeqServerURL: Seeq server URL
                - SeeqUser: Username for authentication
                - SeeqPassword: Password for authentication
                - SeeqRequestOrigin: Request origin label for tracking
            signals_df: DataFrame containing signal definitions with columns:
                - SignalSource: Must contain 'seeq' for relevant signals
                - SourceId: Seeq Datasource ID
                - SignalTagName: Seeq Data ID / tag name
        
        Raises:
            ConnectionError: If unable to connect to Seeq
            ValueError: If required configuration is missing
        """
        logger.info("Initializing Seeq Retrieval agent")
        
        self.config = config
        self.signals_df = signals_df
        self.items = pd.DataFrame()
        self.connection_healthy = False
        
        # Validate configuration
        self._validate_config()
        
        # Connect to Seeq
        self._connect_to_seeq()
        
        # Load Seeq items for all relevant signals
        self._load_seeq_items()
        
        logger.info(
            f"✓ Seeq Retrieval initialized with {len(self.items)} item(s)"
        )
    
    def _validate_config(self):
        """
        Validate that required Seeq configuration is present.
        
        Raises:
            ValueError: If required configuration parameters are missing
        """
        required_params = [
            'SeeqServerURL',
            'SeeqUser',
            'SeeqPassword',
            'SeeqRequestOrigin'
        ]
        
        missing_params = []
        for param in required_params:
            try:
                value = self.config.get('Seeq', param)
                if not value or value.strip() == '':
                    missing_params.append(param)
            except Exception:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(
                f"Missing required Seeq configuration parameters: {missing_params}"
            )
        
        logger.debug("Seeq configuration validated")
    
    def _connect_to_seeq(self):
        """
        Establish connection to Seeq platform.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            seeq_url = self.config.get('Seeq', 'SeeqServerURL')
            seeq_user = self.config.get('Seeq', 'SeeqUser')
            seeq_password = self.config.get('Seeq', 'SeeqPassword')
            seeq_origin = self.config.get('Seeq', 'SeeqRequestOrigin')
            
            logger.info(f"Connecting to Seeq at {seeq_url}")
            
            spy.login(
                url=seeq_url,
                username=seeq_user,
                password=seeq_password,
                request_origin_label=seeq_origin
            )
            
            self.connection_healthy = True
            logger.info("✓ Successfully connected to Seeq")
            
        except Exception as e:
            self.connection_healthy = False
            logger.error(f"Failed to connect to Seeq: {e}", exc_info=True)
            raise ConnectionError(f"Seeq connection failed: {e}")
    
    def _load_seeq_items(self):
        """
        Load Seeq item metadata for all signals from Seeq source.
        
        Searches Seeq for each signal and caches the item metadata
        for efficient retrieval later.
        """
        logger.info("Loading Seeq items for configured signals...")
        
        items_loaded = 0
        items_failed = 0
        
        for idx in self.signals_df.index:
            # Only load items for Seeq signals
            if self.signals_df['SignalSource'][idx].lower() != 'seeq':
                continue
            
            try:
                source_id = self.signals_df['SourceId'][idx]
                signal_tag = self.signals_df['SignalTagName'][idx]
                
                logger.debug(f"Searching for Seeq item: {signal_tag} in {source_id}")
                
                # Search Seeq for this item
                new_items = spy.search({
                    'Datasource ID': source_id,
                    'Data ID': signal_tag
                })
                
                if new_items.empty:
                    logger.warning(
                        f"No Seeq item found for signal '{signal_tag}' "
                        f"in datasource '{source_id}'"
                    )
                    items_failed += 1
                    continue
                
                # Append to items dataframe
                self.items = pd.concat([self.items, new_items], ignore_index=True)
                items_loaded += 1
                
                logger.debug(f"✓ Loaded Seeq item: {signal_tag}")
                
            except Exception as e:
                items_failed += 1
                logger.error(
                    f"Failed to load Seeq item at index {idx}: {e}",
                    exc_info=False
                )
        
        # Log summary
        logger.info(
            f"Seeq items loaded: {items_loaded} succeeded, {items_failed} failed"
        )
        
        if items_loaded == 0:
            logger.warning("No Seeq items loaded - retrieval will fail!")
        
        logger.debug(f"Seeq items dataframe:\n{self.items}")
    
    def retrieve_signals_for_actuators_at_timepoint(
        self,
        signals_df: pd.DataFrame,
        timepoint: datetime
    ):
        """
        Retrieve signal values from Seeq at the specified timepoint.
        
        Updates the signals_df with values retrieved from Seeq. Uses grid
        interpolation to ensure values are returned even when timepoint
        doesn't exactly match a sample.
        
        Implements retry logic for failed retrievals and graceful handling
        of NaN values.
        
        Args:
            signals_df: DataFrame to update with retrieved values
            timepoint: Timestamp for which to retrieve values
        
        Notes:
            - Only updates signals where SignalSource='seeq'
            - Uses 1-minute grid interpolation by default
            - Retries once on NaN values
            - Retains last value if retrieval fails
        """
        if not self.connection_healthy:
            logger.error("Seeq connection not healthy - skipping retrieval")
            return
        
        if self.items.empty:
            logger.warning("No Seeq items available - skipping retrieval")
            return
        
        logger.debug(f"Retrieving Seeq data for timepoint: {timepoint}")
        
        # Retrieve data for all items at once (more efficient than one-by-one)
        try:
            data = self._pull_seeq_data(timepoint)
            
            # Update each signal with retrieved value
            self._update_signals_from_data(signals_df, data, timepoint)
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve Seeq data for timepoint {timepoint}: {e}",
                exc_info=True
            )
    
    def _pull_seeq_data(
        self,
        timepoint: datetime,
        grid_interval: str = DEFAULT_GRID_INTERVAL
    ) -> pd.DataFrame:
        """
        Pull data from Seeq for all items at specified timepoint.
        
        Args:
            timepoint: Timestamp for data retrieval
            grid_interval: Grid interpolation interval (default: "1min")
        
        Returns:
            DataFrame with columns for each signal tag and values at timepoint
        
        Raises:
            Exception: If Seeq pull operation fails
        
        Notes:
            Grid parameter ensures interpolated value is returned even if
            timepoint doesn't exactly match a sample. This addresses Seeq's
            default behavior of sometimes returning no values.
        """
        try:
            data = spy.pull(
                self.items,
                start=timepoint,
                end=timepoint,
                grid=grid_interval
            )
            
            logger.debug(
                f"Successfully pulled data for {len(data.columns)} signal(s)"
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Seeq pull failed: {e}", exc_info=True)
            raise
    
    def _update_signals_from_data(
        self,
        signals_df: pd.DataFrame,
        data: pd.DataFrame,
        timepoint: datetime
    ):
        """
        Update signals dataframe with values from Seeq data.
        
        Args:
            signals_df: DataFrame to update
            data: Seeq data returned from spy.pull()
            timepoint: Timepoint for which data was retrieved
        """
        signals_updated = 0
        signals_failed = 0
        signals_nan = 0
        
        for idx in signals_df.index:
            # Only process Seeq signals
            if signals_df['SignalSource'][idx].lower() != 'seeq':
                continue
            
            curr_signal_tagname = signals_df['SignalTagName'][idx]
            
            try:
                # Check if signal exists in returned data
                if curr_signal_tagname not in data.columns:
                    logger.warning(
                        f"Signal '{curr_signal_tagname}' not in Seeq data"
                    )
                    signals_failed += 1
                    continue
                
                # Get value from data
                curr_signal_value = data[curr_signal_tagname][0]
                
                # Check if value is valid
                if pd.isna(curr_signal_value):
                    signals_nan += 1
                    logger.warning(
                        f"Signal '{curr_signal_tagname}' returned NaN at {timepoint}"
                    )
                    
                    # Attempt retry with individual pull
                    curr_signal_value = self._retry_signal_retrieval(
                        curr_signal_tagname,
                        idx,
                        timepoint
                    )
                    
                    # If still NaN after retry, keep last value
                    if pd.isna(curr_signal_value):
                        logger.warning(
                            f"Retry failed for '{curr_signal_tagname}' - "
                            f"retaining last value"
                        )
                        continue
                
                # Update dataframe with valid value
                signals_df.iloc[
                    idx,
                    signals_df.columns.get_loc('current_val')
                ] = curr_signal_value
                
                signals_updated += 1
                logger.debug(
                    f"Updated '{curr_signal_tagname}' = {curr_signal_value}"
                )
                
            except IndexError as e:
                signals_failed += 1
                logger.error(
                    f"Index error retrieving '{curr_signal_tagname}' "
                    f"at time {timepoint}: {e}"
                )
            except Exception as e:
                signals_failed += 1
                logger.error(
                    f"Failed to update signal '{curr_signal_tagname}': {e}",
                    exc_info=False
                )
        
        # Log summary
        logger.debug(
            f"Seeq retrieval: {signals_updated} updated, "
            f"{signals_nan} NaN, {signals_failed} failed"
        )
    
    def _retry_signal_retrieval(
        self,
        signal_tagname: str,
        signal_idx: int,
        timepoint: datetime
    ) -> Optional[float]:
        """
        Retry retrieval for a single signal that returned NaN.
        
        Attempts to retrieve the signal individually, which sometimes
        succeeds when bulk retrieval fails.
        
        Args:
            signal_tagname: Name of signal to retry
            signal_idx: Index in items dataframe
            timepoint: Timepoint for retrieval
        
        Returns:
            Retrieved value, or NaN if retry fails
        """
        logger.info(f"Retrying retrieval for '{signal_tagname}'")
        
        try:
            # Small delay before retry
            time.sleep(self.RETRY_DELAY_SECONDS)
            
            # Pull just this one item
            retry_data = spy.pull(
                self.items[signal_idx:signal_idx+1],
                start=timepoint,
                end=timepoint,
                grid=self.DEFAULT_GRID_INTERVAL
            )
            
            if signal_tagname in retry_data.columns:
                retry_value = retry_data[signal_tagname][0]
                
                if not pd.isna(retry_value):
                    logger.info(
                        f"✓ Retry successful for '{signal_tagname}' = {retry_value}"
                    )
                    return retry_value
            
            logger.warning(f"Retry returned NaN for '{signal_tagname}'")
            return np.nan
            
        except Exception as e:
            logger.error(f"Retry failed for '{signal_tagname}': {e}")
            return np.nan
    
    def test_connection(self) -> bool:
        """
        Test if Seeq connection is healthy.
        
        Attempts a simple operation to verify connectivity.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Try a simple search operation
            test_result = spy.search({'Name': 'test'}, quiet=True)
            self.connection_healthy = True
            logger.debug("Seeq connection test: PASS")
            return True
            
        except Exception as e:
            self.connection_healthy = False
            logger.warning(f"Seeq connection test: FAIL - {e}")
            return False
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Seeq retrieval configuration.
        
        Returns:
            Dictionary containing:
                - item_count: Number of Seeq items loaded
                - connection_healthy: Connection status
                - signal_count: Number of signals from Seeq
        """
        seeq_signals = self.signals_df[
            self.signals_df['SignalSource'].str.lower() == 'seeq'
        ]
        
        return {
            'item_count': len(self.items),
            'connection_healthy': self.connection_healthy,
            'signal_count': len(seeq_signals),
            'items_loaded': not self.items.empty
        }
    
    def __repr__(self) -> str:
        """String representation of SeeqRetrieval instance."""
        return (
            f"SeeqRetrieval("
            f"items={len(self.items)}, "
            f"healthy={self.connection_healthy}"
            f")"
        )


if __name__ == "__main__":
    """
    Simple test/example of SeeqRetrieval usage.
    
    Note: Requires valid Seeq credentials and accessible server.
    """
    import configparser
    
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test configuration
    config = configparser.ConfigParser()
    config['Seeq'] = {
        'SeeqServerURL': 'https://example.seeq.com',
        'SeeqUser': 'test_user',
        'SeeqPassword': 'test_password',
        'SeeqRequestOrigin': 'DigitalTwin_Test'
    }
    
    # Create test signals dataframe
    test_signals = pd.DataFrame({
        'SignalTagName': ['TempSensor1', 'PressureSensor1'],
        'SignalSource': ['seeq', 'seeq'],
        'SourceId': ['DataSource1', 'DataSource1'],
        'current_val': [-1, -1]
    })
    
    logger.info("=== SeeqRetrieval Test ===")
    
    try:
        # Initialize (will fail without real credentials)
        seeq = SeeqRetrieval(config, test_signals)
        
        # Test connection
        is_connected = seeq.test_connection()
        logger.info(f"Connection test: {'PASS' if is_connected else 'FAIL'}")
        
        # Get statistics
        stats = seeq.get_retrieval_statistics()
        logger.info(f"Statistics: {stats}")
        
        # Try retrieval (will fail without real data)
        test_time = datetime.now()
        seeq.retrieve_signals_for_actuators_at_timepoint(test_signals, test_time)
        
    except Exception as e:
        logger.error(f"Test failed (expected without real Seeq): {e}")