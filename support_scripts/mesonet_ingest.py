"""
NYS Mesonet API Ingestion Script
==================================
Fetches data from the NYS Mesonet API for the VOOR station and upserts
into the PostgreSQL weather table, matching the schema and units established
by mesonet_bulk_load.py.

API endpoint:
  https://api.nysmesonet.org/data/dynserv/nycreates-zen/5min/nysm/<start>/<end>

  - Network: nysm (VOOR station only)
  - Units:   metric (degC, m/s, mbar, W/m^2)
  - Limit:   1 month per request
  - Dewpoint is not provided by the API -- derived via Magnus formula

Usage:
  python mesonet_ingest.py                       # incremental: last DB timestamp -> now
  python mesonet_ingest.py --from "2025-02-17"   # explicit start date -> now

Schedule (Windows Task Scheduler):
  Run every 15 minutes. Script is idempotent -- safe to overlap.
"""

import os
import io
import math
import logging
import argparse
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     os.getenv("MESONET_DB_HOST",     "localhost"),
    "port":     int(os.getenv("MESONET_DB_PORT", "5432")),
    "dbname":   os.getenv("MESONET_DB_NAME",     ""),
    "user":     os.getenv("MESONET_DB_USER",     ""),
    "password": os.getenv("MESONET_DB_PASSWORD", ""),
}

API_BASE   = "https://api.nysmesonet.org/data/dynserv/#############/5min/nysm"# Replace ############# with your actual API key from NYS Mesonet
DATE_FMT   = "%Y%m%dT%H%M"
MAX_WINDOW = timedelta(days=30)
OVERLAP    = timedelta(minutes=10)   # re-fetch slightly before last record for late QC updates

TABLE_NAME = "weather"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dewpoint derivation
# ---------------------------------------------------------------------------

def magnus_dewpoint(temp_c: float, rh_pct: float) -> float:
    """
    Derive dewpoint (degC) from dry bulb temperature (degC) and relative
    humidity (%) using the Magnus formula.
    Accurate to within ~0.35 degC for typical atmospheric conditions.
    """
    if pd.isna(temp_c) or pd.isna(rh_pct) or rh_pct <= 0:
        return float("nan")
    a = 17.625
    b = 243.04
    rh = rh_pct / 100.0
    gamma = (a * temp_c / (b + temp_c)) + math.log(rh)
    return (b * gamma) / (a - gamma)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

UPSERT_SQL = f"""
    INSERT INTO {TABLE_NAME}
        (time, outdoor_dry_bulb, outdoor_rel_humidity, wind_speed,
         wind_direction, solar_insolation, atm_pressure, dewpoint, is_forecast)
    VALUES %s
    ON CONFLICT (time)
    DO UPDATE SET
        outdoor_dry_bulb     = EXCLUDED.outdoor_dry_bulb,
        outdoor_rel_humidity = EXCLUDED.outdoor_rel_humidity,
        wind_speed           = EXCLUDED.wind_speed,
        wind_direction       = EXCLUDED.wind_direction,
        solar_insolation     = EXCLUDED.solar_insolation,
        atm_pressure         = EXCLUDED.atm_pressure,
        dewpoint             = EXCLUDED.dewpoint,
        is_forecast          = EXCLUDED.is_forecast
"""


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_latest_timestamp(conn) -> datetime | None:
    """Return the most recent timestamp in the weather table, or None."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(time) FROM {TABLE_NAME}")
        result = cur.fetchone()[0]
    return result  # timezone-aware datetime or None


def upsert_rows(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, UPSERT_SQL, rows)
    conn.commit()
    return len(rows)

# ---------------------------------------------------------------------------
# API fetch and parse
# ---------------------------------------------------------------------------

def fetch_window(start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch one <=30-day window from the API. Returns DataFrame or None on error."""
    url = f"{API_BASE}/{start.strftime(DATE_FMT)}/{end.strftime(DATE_FMT)}"
    log.info("  GET %s", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return df
    except requests.HTTPError as e:
        log.error("  HTTP error: %s", e)
    except Exception as e:
        log.exception("  Unexpected fetch error: %s", e)
    return None


def parse_dataframe(df: pd.DataFrame) -> list[tuple]:
    """
    Map API columns to DB columns, derive dewpoint, return list of tuples.

    API columns used:
      datetime                      -> time
      temp_2m [degC]                -> outdoor_dry_bulb
      relative_humidity [%]         -> outdoor_rel_humidity
      avg_wind_speed_prop [m/s]     -> wind_speed
      wind_direction_prop [degrees] -> wind_direction
      solar_insolation [watt/m**2]  -> solar_insolation
      station_pressure [mbar]       -> atm_pressure
      (derived via Magnus)          -> dewpoint
    """
    rename = {
        "datetime":                       "time",
        "temp_2m [degC]":                 "outdoor_dry_bulb",
        "relative_humidity [%]":          "outdoor_rel_humidity",
        "avg_wind_speed_prop [m/s]":      "wind_speed",
        "wind_direction_prop [degrees]":  "wind_direction",
        "solar_insolation [watt/m**2]":   "solar_insolation",
        "station_pressure [mbar]":        "atm_pressure",
    }
    df = df.rename(columns=rename)

    # Parse timestamp to UTC
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # Coerce numerics
    for col in ["outdoor_dry_bulb", "outdoor_rel_humidity", "wind_speed",
                "wind_direction", "solar_insolation", "atm_pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    # Derive dewpoint via Magnus formula
    df["dewpoint"] = df.apply(
        lambda row: magnus_dewpoint(row["outdoor_dry_bulb"], row["outdoor_rel_humidity"]),
        axis=1,
    )

    # Mark all Mesonet observations as not forecast
    df["is_forecast"] = False

    # Final column selection in DB order
    out_cols = ["time", "outdoor_dry_bulb", "outdoor_rel_humidity",
                "wind_speed", "wind_direction", "solar_insolation",
                "atm_pressure", "dewpoint", "is_forecast"]
    df = df[out_cols]

    # Drop rows where timestamp failed to parse
    df = df.dropna(subset=["time"])

    # Replace NaN with None for psycopg2
    df = df.where(pd.notnull(df), other=None)

    return [tuple(row) for row in df.itertuples(index=False, name=None)]

# ---------------------------------------------------------------------------
# Window chunking
# ---------------------------------------------------------------------------

def date_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a date range into <=30-day chunks."""
    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + MAX_WINDOW, end)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start_override: str | None = None):
    now = datetime.now(tz=timezone.utc)

    with get_connection() as conn:
        if start_override:
            start = datetime.fromisoformat(start_override).replace(tzinfo=timezone.utc)
            log.info("Using explicit start date: %s", start.isoformat())
        else:
            latest = get_latest_timestamp(conn)
            if latest is None:
                log.error(
                    "No data in DB and no --from date specified. "
                    "Run mesonet_bulk_load.py first, or provide --from YYYY-MM-DD."
                )
                return
            start = latest - OVERLAP
            log.info("Resuming from last DB timestamp: %s (with %s overlap)",
                     latest.isoformat(), OVERLAP)

        windows = date_windows(start, now)
        log.info("Fetching %d window(s) from %s to %s",
                 len(windows), start.isoformat(), now.isoformat())

        total = 0
        for w_start, w_end in windows:
            df = fetch_window(w_start, w_end)
            if df is None or df.empty:
                log.warning("  No data for window %s -> %s", w_start, w_end)
                continue
            rows = parse_dataframe(df)
            n = upsert_rows(conn, rows)
            total += n
            log.info("  Upserted %d rows.", n)

        log.info("Done. Total rows upserted: %d", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NYS Mesonet API -> PostgreSQL incremental ingestion (VOOR station)"
    )
    parser.add_argument(
        "--from",
        dest="start_date",
        metavar="YYYY-MM-DD",
        help="Explicit start date (UTC). Defaults to last timestamp in DB.",
    )
    args = parser.parse_args()
    run(start_override=args.start_date)
