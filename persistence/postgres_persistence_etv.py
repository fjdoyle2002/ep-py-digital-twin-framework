# -*- coding: utf-8 -*-
"""
Entity-Timestamp-Value (ETV) Persistence with proper entity metadata management
"""
import psycopg2
from psycopg2 import pool, OperationalError, errorcodes, errors
from psycopg2.extras import execute_batch
import datetime
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PostgresPersistenceETV:
    """
    ETV (Entity-Timestamp-Value) persistence layer for digital twin.
    
    Creates and manages two tables:
    1. Signals metadata table: Tracks all signals/entities with their metadata
    2. Timeseries data table: Stores actual timestamp-value pairs
    
    Benefits of ETV model:
    - Easy to add new signals without schema changes
    - Efficient storage for sparse data
    - Natural fit for time-series databases
    - Metadata stored once, not repeated
    """
    
    def __init__(self, config, sensors_df):
        self.sensors_df = sensors_df
        self.dbname = config.get('DATABASE', 'DatabaseName')       
        self.dbhost = config.get('DATABASE', 'DatabaseHost')
        self.dbport = config.get('DATABASE', 'DatabasePort')
        self.dbuser = config.get('DATABASE', 'DatabaseUser')
        self.dbpass = config.get('DATABASE', 'DatabasePass')
        self.dt_name = config.get('DEFAULT', 'DigitalTwinIdentifier')
        
        # Table names
        self.signals_table = f"{self.dt_name}_signals"
        self.timeseries_table = f"{self.dt_name}_timeseries"
        
        # Cache signal_name -> signal_id mapping to avoid repeated lookups
        self.signal_id_cache: Dict[str, int] = {}
        
        self.successfully_initialized = False
        self.connection_pool = None
        
        try:
            # Use connection pooling for better performance
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=self.dbhost,
                port=self.dbport,
                database=self.dbname,
                user=self.dbuser,
                password=self.dbpass
            )
            logger.info("Database connection pool created successfully")
        except psycopg2.Error as e:
            logger.error(f"Unable to connect to database: {e.pgerror}")
            if hasattr(e, 'diag') and e.diag:
                logger.error(f"Details: {e.diag.message_detail}")
            raise
        
        # Initialize database schema
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Initialize both signals and timeseries tables"""
        conn = self.connection_pool.getconn()
        try:
            # Create signals metadata table
            self._create_signals_table(conn)
            
            # Create timeseries data table
            self._create_timeseries_table(conn)
            
            # Register all sensors from sensors_df
            self._register_signals(conn)
            
            # Build the signal_id cache
            self._build_signal_cache(conn)
            
            conn.commit()
            self.successfully_initialized = True
            logger.info("Database schema initialized successfully")
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to initialize schema: {e.pgerror}")
            if hasattr(e, 'diag') and e.diag:
                logger.error(f"Details: {e.diag.message_detail}")
            raise
        finally:
            self.connection_pool.putconn(conn)
    
    def _create_signals_table(self, conn):
        """Create the signals metadata table if it doesn't exist"""
        cur = conn.cursor()
        try:
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.signals_table} (
                signal_id SERIAL PRIMARY KEY,
                signal_name VARCHAR(255) UNIQUE NOT NULL,
                persistence_name VARCHAR(255),
                description TEXT,
                unit VARCHAR(100),
                data_type VARCHAR(50),
                ep_sensor_name VARCHAR(255),
                ep_sensor_instance VARCHAR(255),
                ep_type VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            cur.execute(sql)
            
            # Create index on signal_name for fast lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.signals_table}_name 
                ON {self.signals_table}(signal_name);
            """)
            
            logger.info(f"Signals table '{self.signals_table}' ready")
        finally:
            cur.close()
    
    def _create_timeseries_table(self, conn):
        """Create the timeseries data table if it doesn't exist"""
        cur = conn.cursor()
        try:
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.timeseries_table} (
                id BIGSERIAL PRIMARY KEY,
                signal_id INTEGER NOT NULL REFERENCES {self.signals_table}(signal_id) ON DELETE CASCADE,
                timestamp TIMESTAMPTZ NOT NULL,
                value DOUBLE PRECISION,
                UNIQUE(signal_id, timestamp)
            );
            """
            cur.execute(sql)
            
            # Create indexes for efficient querying
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.timeseries_table}_time 
                ON {self.timeseries_table}(timestamp);
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.timeseries_table}_signal_time 
                ON {self.timeseries_table}(signal_id, timestamp);
            """)
            
            logger.info(f"Timeseries table '{self.timeseries_table}' ready")
        finally:
            cur.close()
    
    def _register_signals(self, conn):
        """
        Register all signals from sensors_df in the signals metadata table.
        Only inserts signals that don't already exist (based on PersistenceName).
        """
        cur = conn.cursor()
        try:
            for idx in self.sensors_df.index:
                persistence_name = self.sensors_df['PersistenceName'][idx]
                
                # Check if signal already exists
                cur.execute(
                    f"SELECT signal_id FROM {self.signals_table} WHERE signal_name = %s",
                    (persistence_name,)
                )
                
                if cur.fetchone() is None:
                    # Signal doesn't exist, insert it
                    sql = f"""
                    INSERT INTO {self.signals_table} 
                    (signal_name, persistence_name, ep_sensor_name, ep_sensor_instance, ep_type, data_type)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (signal_name) DO NOTHING;
                    """
                    
                    cur.execute(sql, (
                        persistence_name,
                        persistence_name,
                        self.sensors_df.get('SensorName', pd.Series([None] * len(self.sensors_df)))[idx],
                        self.sensors_df.get('SensorInstance', pd.Series([None] * len(self.sensors_df)))[idx],
                        self.sensors_df.get('Type', pd.Series([None] * len(self.sensors_df)))[idx],
                        self.sensors_df.get('DataType', pd.Series(['real'] * len(self.sensors_df)))[idx]
                    ))
                    
                    logger.info(f"Registered new signal: {persistence_name}")
                else:
                    logger.debug(f"Signal already registered: {persistence_name}")
        finally:
            cur.close()
    
    def _build_signal_cache(self, conn):
        """Build a cache mapping signal_name -> signal_id for fast lookups"""
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT signal_id, signal_name FROM {self.signals_table}")
            self.signal_id_cache = {row[1]: row[0] for row in cur.fetchall()}
            logger.info(f"Signal ID cache built with {len(self.signal_id_cache)} signals")
        finally:
            cur.close()
            
    def load_units_into_sensors_df(self):
        """
        Load units from database into sensors_df.
        
        This should be called after schema initialization to populate sensors_df
        with any existing units from the database (including user overrides).
        
        RDD parsing can then update any NULL/missing units.
        """
        if not self.successfully_initialized:
            logger.warning("Database not initialized, cannot load units")
            return
        
        conn = self.connection_pool.getconn()
        try:
            cur = conn.cursor()
            
            # Get all signals with their units
            cur.execute(f"""
                SELECT signal_name, unit 
                FROM {self.signals_table}
                WHERE unit IS NOT NULL AND unit != ''
            """)
            
            units_from_db = {row[0]: row[1] for row in cur.fetchall()}
            cur.close()
            
            # Add 'unit' column to sensors_df if it doesn't exist
            if 'unit' not in self.sensors_df.columns:
                self.sensors_df['unit'] = None
            
            # Update sensors_df with units from database
            updated_count = 0
            for idx in self.sensors_df.index:
                persistence_name = self.sensors_df['PersistenceName'][idx]
                
                if persistence_name in units_from_db:
                    db_unit = units_from_db[persistence_name]
                    self.sensors_df.loc[idx, 'unit'] = db_unit
                    updated_count += 1
                    logger.debug(f"Loaded unit for '{persistence_name}': {db_unit}")
            
            logger.info(f"Loaded {updated_count} units from database into sensors_df")
            
        except psycopg2.Error as e:
            logger.error(f"Failed to load units from database: {e}")
        finally:
            self.connection_pool.putconn(conn)
    
    def add_signal(self, signal_name: str, **metadata) -> Optional[int]:
        """
        Dynamically add a new signal to the metadata table.
        
        Args:
            signal_name: Unique name for the signal
            **metadata: Optional metadata (description, unit, data_type, etc.)
        
        Returns:
            signal_id if successful, None otherwise
        """
        conn = self.connection_pool.getconn()
        try:
            cur = conn.cursor()
            
            # Build column names and values from metadata
            columns = ['signal_name'] + list(metadata.keys())
            values = [signal_name] + list(metadata.values())
            placeholders = ', '.join(['%s'] * len(values))
            columns_str = ', '.join(columns)
            
            sql = f"""
            INSERT INTO {self.signals_table} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (signal_name) DO UPDATE 
            SET updated_at = NOW()
            RETURNING signal_id;
            """
            
            cur.execute(sql, values)
            signal_id = cur.fetchone()[0]
            
            # Update cache
            self.signal_id_cache[signal_name] = signal_id
            
            conn.commit()
            logger.info(f"Added signal '{signal_name}' with ID {signal_id}")
            return signal_id
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to add signal '{signal_name}': {e.pgerror}")
            return None
        finally:
            cur.close()
            self.connection_pool.putconn(conn)
    
    def persist(self, timestamp: datetime.datetime):
        """
        Persist current sensor values to the timeseries table.
        
        Uses batch insert for better performance when storing multiple signals.
        """
        if not self.successfully_initialized:
            logger.error("Persistence layer not properly initialized")
            return
        
        conn = self.connection_pool.getconn()
        try:
            cur = conn.cursor()
            
            # Prepare batch data: [(signal_id, timestamp, value), ...]
            batch_data = []
            
            for idx in self.sensors_df.index:
                persistence_name = self.sensors_df['PersistenceName'][idx]
                current_value = self.sensors_df['current_val'][idx]
                
                # Get signal_id from cache
                signal_id = self.signal_id_cache.get(persistence_name)
                
                if signal_id is None:
                    logger.warning(f"Signal '{persistence_name}' not in cache, skipping")
                    continue
                
                batch_data.append((signal_id, timestamp, current_value))
            
            # Batch insert for performance
            if batch_data:
                sql = f"""
                INSERT INTO {self.timeseries_table} (signal_id, timestamp, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (signal_id, timestamp) 
                DO UPDATE SET value = EXCLUDED.value;
                """
                
                execute_batch(cur, sql, batch_data, page_size=100)
                conn.commit()
                logger.debug(f"Persisted {len(batch_data)} signals at {timestamp}")
            else:
                logger.warning(f"No data to persist at {timestamp}")
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to persist data at {timestamp}: {e.pgerror}")
            if hasattr(e, 'diag') and e.diag:
                logger.error(f"Details: {e.diag.message_detail}")
        finally:
            cur.close()
            self.connection_pool.putconn(conn)
    
    def get_signal_history(self, signal_name: str, 
                          start_time: datetime.datetime,
                          end_time: datetime.datetime) -> pd.DataFrame:
        """
        Retrieve historical data for a specific signal.
        
        Returns:
            DataFrame with columns: timestamp, value
        """
        signal_id = self.signal_id_cache.get(signal_name)
        if signal_id is None:
            logger.error(f"Signal '{signal_name}' not found")
            return pd.DataFrame()
        
        conn = self.connection_pool.getconn()
        try:
            sql = f"""
            SELECT timestamp, value
            FROM {self.timeseries_table}
            WHERE signal_id = %s 
              AND timestamp >= %s 
              AND timestamp <= %s
            ORDER BY timestamp;
            """
            
            df = pd.read_sql_query(sql, conn, params=(signal_id, start_time, end_time))
            return df
            
        except psycopg2.Error as e:
            logger.error(f"Failed to retrieve history for '{signal_name}': {e.pgerror}")
            return pd.DataFrame()
        finally:
            self.connection_pool.putconn(conn)
    
    def update_signal_unit(self, signal_id: int, unit: str):
        """
        Update the unit for a signal in the signals metadata table.
        
        Args:
            signal_id: ID of the signal to update
            unit: Unit string to set
        """
        conn = self.connection_pool.getconn()
        try:
            cur = conn.cursor()
            
            sql = f"""
            UPDATE {self.signals_table}
            SET unit = %s, updated_at = NOW()
            WHERE signal_id = %s;
            """
            
            cur.execute(sql, (unit, signal_id))
            conn.commit()
            logger.debug(f"Updated unit for signal_id {signal_id} to '{unit}'")
            
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Failed to update unit for signal_id {signal_id}: {e.pgerror}")
            if hasattr(e, 'diag') and e.diag:
                logger.error(f"Details: {e.diag.message_detail}")
            raise
        finally:
            cur.close()
            self.connection_pool.putconn(conn)

    def close(self):
        """Close all database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

