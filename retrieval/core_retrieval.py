# -*- coding: utf-8 -*-
"""
Core Retrieval Module

Manages retrieval of real-world building signals from multiple data sources
(Seeq, OPC-UA, etc.) and coordinates their aggregation for use by the digital twin.

This module provides a plugin-style architecture where different retrieval agents
can be registered and will be called in sequence to populate signal values.

Created on: Original date unknown
Refactored: February 2026

@author: doylef
"""

import pandas as pd
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from retrieval.seeq_retrieval import SeeqRetrieval

logger = logging.getLogger(__name__)


class CoreRetrieval:
    """
    Core retrieval coordinator for digital twin signal acquisition.
    
    Manages multiple retrieval agents (Seeq, OPC-UA, custom sources) and
    coordinates their execution to populate the signals dataframe with
    real-world building data.
    
    Features:
    - Plugin-style architecture for adding new data sources
    - Automatic detection of required retrieval agents from configuration
    - Graceful handling of partial failures
    - Comprehensive logging of retrieval operations
    
    Attributes:
        config: Configuration object containing connection parameters
        signals_df: DataFrame of signals to retrieve with their sources
        retrieval_agents: List of active retrieval agent instances
    """
    
    def __init__(self, config, signals_df: pd.DataFrame):
        """
        Initialize the core retrieval coordinator.
        
        Analyzes the signals dataframe to determine which retrieval agents
        are needed and initializes them automatically.
        
        Args:
            config: ConfigParser object with retrieval configuration
            signals_df: DataFrame containing signal definitions with columns:
                - SignalTagName: Unique identifier for the signal
                - SignalSource: Source system (e.g., 'seeq', 'opc')
                - SourceId: Source-specific identifier
                - current_val: Current value (to be populated)
        
        Raises:
            ValueError: If required configuration is missing
            Exception: If critical retrieval agent initialization fails
        """
        logger.info("Initializing Core Retrieval system")
        
        self.config = config
        self.signals_df = signals_df
        self.retrieval_agents: List[Any] = []
        
        # Validate signals dataframe
        self._validate_signals_df()
        
        # Initialize retrieval agents based on signal sources
        self._initialize_retrieval_agents()
        
        logger.info(
            f"Core Retrieval initialized with {len(self.retrieval_agents)} agent(s)"
        )
    
    def _validate_signals_df(self):
        """
        Validate that signals dataframe has required structure.
        
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['SignalTagName', 'SignalSource']
        missing_columns = [col for col in required_columns 
                          if col not in self.signals_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"signals_df missing required columns: {missing_columns}"
            )
        
        if self.signals_df.empty:
            logger.warning("signals_df is empty - no signals to retrieve")
        
        logger.debug(f"Validated signals_df with {len(self.signals_df)} signals")
    
    def _initialize_retrieval_agents(self):
        """
        Initialize retrieval agents based on signal sources in configuration.
        
        Examines the SignalSource column to determine which retrieval agents
        are needed. Currently supports:
        - Seeq (historian/analytics platform)
        - OPC-UA (added via add_retrieval_agent)
        - Custom sources (added via add_retrieval_agent)
        
        Each agent is initialized only if signals reference it.
        """
        # Get unique signal sources (case-insensitive)
        signal_sources = self.signals_df['SignalSource'].str.lower().unique()
        logger.info(f"Signal sources detected: {sorted(signal_sources)}")
        
        # Initialize Seeq retrieval if needed
        if 'seeq' in signal_sources:
            try:
                seeq_agent = SeeqRetrieval(self.config, self.signals_df)
                self.retrieval_agents.append(seeq_agent)
                logger.info("✓ Seeq retrieval agent initialized")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Seeq retrieval agent: {e}",
                    exc_info=True
                )
                # Decide if this is fatal or continue with degraded functionality
                if self._is_seeq_required():
                    raise
                else:
                    logger.warning("Continuing without Seeq agent")
        
        # Note: OPC-UA and other agents are added dynamically via add_retrieval_agent()
        # They're typically initialized in digital_twin.py and passed in
        
        # Warn about unknown sources
        known_sources = {'seeq', 'opc', 'none'}
        unknown_sources = set(signal_sources) - known_sources
        if unknown_sources:
            logger.warning(
                f"Unknown signal sources (will need to be added via add_retrieval_agent): "
                f"{unknown_sources}"
            )
    
    def _is_seeq_required(self) -> bool:
        """
        Determine if Seeq is required for critical signals.
        
        Returns:
            True if Seeq failure should be fatal, False otherwise
        """
        seeq_signals = self.signals_df[
            self.signals_df['SignalSource'].str.lower() == 'seeq'
        ]
        
        # Consider Seeq required if more than 50% of signals come from it
        # This is a heuristic - adjust based on your requirements
        is_required = len(seeq_signals) > len(self.signals_df) * 0.5
        
        if is_required:
            logger.debug("Seeq is required (majority of signals)")
        else:
            logger.debug("Seeq is optional (minority of signals)")
        
        return is_required
    
    def add_retrieval_agent(self, agent):
        """
        Add a retrieval agent to the list of active agents.
        
        Retrieval agents must implement the method:
            retrieve_signals_for_actuators_at_timepoint(signals_df, timepoint)
        
        This allows for runtime addition of data sources (e.g., OPC-UA module
        initialized after CoreRetrieval).
        
        Args:
            agent: Retrieval agent instance with required interface
        
        Raises:
            TypeError: If agent doesn't implement required interface
        
        Example:
            >>> opc_module = OPCUAModule(config)
            >>> core_retrieval.add_retrieval_agent(opc_module)
        """
        # Validate agent has required method
        if not hasattr(agent, 'retrieve_signals_for_actuators_at_timepoint'):
            raise TypeError(
                f"Retrieval agent {type(agent).__name__} must implement "
                f"'retrieve_signals_for_actuators_at_timepoint' method"
            )
        
        self.retrieval_agents.append(agent)
        agent_name = type(agent).__name__
        logger.info(f"Added retrieval agent: {agent_name}")
        logger.debug(f"Total active agents: {len(self.retrieval_agents)}")
    
    def retrieve_signals_for_actuators_at_timepoint(
        self,
        signals_df: pd.DataFrame,
        timepoint: datetime
    ):
        """
        Retrieve all signals for the specified timepoint.
        
        Calls each registered retrieval agent in sequence to populate signal values.
        Each agent is responsible for updating signals from its source.
        
        The function uses a "best effort" approach:
        - If one agent fails, others still execute
        - Partial failures are logged but don't stop retrieval
        - Signals that can't be retrieved retain their last value
        
        Args:
            signals_df: DataFrame to populate with signal values
            timepoint: Timestamp for which to retrieve signal values
        
        Notes:
            - Order of agent execution matters if multiple agents could
              provide the same signal (last writer wins)
            - Agents should handle their own errors and update only the
              signals they're responsible for
        
        Example:
            >>> dt = datetime(2026, 2, 11, 14, 30)
            >>> core_retrieval.retrieve_signals_for_actuators_at_timepoint(
            ...     signals_df, dt
            ... )
            # signals_df now contains values at 14:30
        """
        if not self.retrieval_agents:
            logger.warning(
                "No retrieval agents registered - signals will not be updated"
            )
            return
        
        logger.debug(
            f"Retrieving signals for timepoint: {timepoint} "
            f"using {len(self.retrieval_agents)} agent(s)"
        )
        
        # Track retrieval statistics
        agents_succeeded = 0
        agents_failed = 0
        
        # Call each retrieval agent
        for agent in self.retrieval_agents:
            agent_name = type(agent).__name__
            
            try:
                logger.debug(f"Calling retrieval agent: {agent_name}")
                
                # Call agent's retrieval method
                agent.retrieve_signals_for_actuators_at_timepoint(
                    signals_df,
                    timepoint
                )
                
                agents_succeeded += 1
                logger.debug(f"✓ {agent_name} completed successfully")
                
            except Exception as e:
                agents_failed += 1
                logger.error(
                    f"✗ Retrieval agent '{agent_name}' failed: {e}",
                    exc_info=True
                )
                # Continue with other agents despite this failure
        
        # Log summary
        if agents_failed > 0:
            logger.warning(
                f"Signal retrieval completed with errors: "
                f"{agents_succeeded} succeeded, {agents_failed} failed"
            )
        else:
            logger.debug(
                f"Signal retrieval completed successfully "
                f"({agents_succeeded} agent(s))"
            )
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary containing:
                - agent_count: Number of registered agents
                - agent_names: List of agent class names
                - signal_count: Total number of signals
                - signals_by_source: Count of signals per source
        
        Example:
            >>> stats = core_retrieval.get_retrieval_statistics()
            >>> print(f"Active agents: {stats['agent_count']}")
        """
        stats = {
            'agent_count': len(self.retrieval_agents),
            'agent_names': [type(agent).__name__ for agent in self.retrieval_agents],
            'signal_count': len(self.signals_df),
            'signals_by_source': {}
        }
        
        # Count signals by source
        if not self.signals_df.empty and 'SignalSource' in self.signals_df.columns:
            source_counts = self.signals_df['SignalSource'].value_counts()
            stats['signals_by_source'] = source_counts.to_dict()
        
        return stats
    
    def validate_signal_coverage(self) -> bool:
        """
        Validate that all signals have a registered retrieval agent.
        
        Checks if every signal source has a corresponding retrieval agent
        that can provide data for it.
        
        Returns:
            True if all signals are covered, False if some lack agents
        
        Note:
            This is a best-effort check. It verifies agent registration
            but cannot guarantee the agent will successfully retrieve data.
        """
        # Get unique signal sources (excluding 'none')
        signal_sources = set(
            self.signals_df['SignalSource'].str.lower().unique()
        ) - {'none'}
        
        # Map known agent types to sources they handle
        agent_source_map = {
            'SeeqRetrieval': 'seeq',
            'OPCUAModule': 'opc',
        }
        
        # Get sources handled by registered agents
        handled_sources = set()
        for agent in self.retrieval_agents:
            agent_type = type(agent).__name__
            if agent_type in agent_source_map:
                handled_sources.add(agent_source_map[agent_type])
        
        # Check for uncovered sources
        uncovered_sources = signal_sources - handled_sources
        
        if uncovered_sources:
            logger.warning(
                f"Signal sources without registered agents: {uncovered_sources}"
            )
            return False
        else:
            logger.debug("All signal sources have registered agents")
            return True
    
    def __repr__(self) -> str:
        """String representation of CoreRetrieval instance."""
        return (
            f"CoreRetrieval("
            f"agents={len(self.retrieval_agents)}, "
            f"signals={len(self.signals_df)}"
            f")"
        )


if __name__ == "__main__":
    """
    Simple test/example of CoreRetrieval usage.
    
    This demonstrates the basic initialization and usage pattern.
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
        'SeeqRequestOrigin': 'DigitalTwin'
    }
    
    # Create test signals dataframe
    test_signals = pd.DataFrame({
        'SignalTagName': ['Temp1', 'Pressure1', 'Flow1'],
        'SignalSource': ['seeq', 'opc', 'seeq'],
        'SourceId': ['ABC123', 'N/A', 'DEF456'],
        'current_val': [-1, -1, -1]
    })
    
    logger.info("=== CoreRetrieval Test ===")
    
    try:
        # Initialize (will fail without real Seeq credentials)
        core_retrieval = CoreRetrieval(config, test_signals)
        
        # Get statistics
        stats = core_retrieval.get_retrieval_statistics()
        logger.info(f"Statistics: {stats}")
        
        # Validate coverage
        is_covered = core_retrieval.validate_signal_coverage()
        logger.info(f"All signals covered: {is_covered}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)