"""
Open-Meteo Forecast Ingestion Script
======================================
Fetches 16-day hourly weather forecasts from Open-Meteo and upserts into
the PostgreSQL weather table, marking rows as is_forecast = True.

As real Mesonet observations arrive via mesonet_ingest.py, they
automatically overwrite forecast rows for past timestamps, flipping
is_forecast back to False.

Forecast variables mapped to DB schema:
  temperature_2m          -> outdoor_dry_bulb    (degC)
  relative_humidity_2m    -> outdoor_rel_humidity (percent)
  wind_speed_10m          -> wind_speed           (m/s)
  wind_direction_10m      -> wind_direction       (degrees)
  direct_radiation
  + diffuse_radiation     -> solar_insolation     (W/m^2, summed for GHI)
  surface_pressure        -> atm_pressure         (hPa = mbar)
  derived via Magnus      -> dewpoint             (degC)

API endpoint:
  https://api.open-meteo.com/v1/forecast
  No API key required.

Usage:
  python forecast_ingest.py          # normal run: upsert from now+1hr to +16 days
  python forecast_ingest.py --force  # upsert all 16 days regardless of existing rows

Schedule (Windows Task Scheduler / NSSM):
  Run every 6 hours as a separate service from mesonet_ingest.py.
  Script is idempotent -- safe to run manually at any time.
"""

import os
import io
import math
import time
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

# VOOR station coordinates
LATITUDE  = 42.65242
LONGITUDE = -73.97562

FORECAST_DAYS = 16
RUN_INTERVAL  = 6 * 3600   # 6 hours in seconds

TABLE_NAME = "weather"

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,direct_radiation,diffuse_radiation,surface_pressure"
    f"&forecast_days={FORECAST_DAYS}"
    "&timezone=UTC"
    "&wind_speed_unit=ms"
)

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
# Dewpoint derivation (Magnus formula)
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


def get_first_forecast_timestamp(conn) -> datetime | None:
    """
    Return the earliest timestamp in the DB marked is_forecast = True
    that is after the current time. This is the start of the existing
    forecast window -- we write from here forward to avoid conflicting
    with the Mesonet ingest script on recent observed rows.
    """
    now = datetime.now(tz=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MIN(time) FROM {TABLE_NAME} WHERE is_forecast = TRUE AND time > %s",
            (now,),
        )
        result = cur.fetchone()[0]
    return result


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

def fetch_forecast() -> pd.DataFrame | None:
    """Fetch 16-day hourly forecast from Open-Meteo. Returns DataFrame or None."""
    log.info("Fetching forecast from Open-Meteo...")
    try:
        resp = requests.get(API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        log.error("HTTP error fetching forecast: %s", e)
        return None
    except Exception as e:
        log.exception("Unexpected error fetching forecast: %s", e)
        return None

    hourly = data.get("hourly")
    if not hourly:
        log.error("No hourly data in API response.")
        return None

    df = pd.DataFrame(hourly)
    return df


def parse_forecast(df: pd.DataFrame, force: bool) -> list[tuple]:
    """
    Map Open-Meteo columns to DB schema, derive dewpoint, filter to
    future timestamps only (unless --force), return list of row tuples.
    """
    # Parse timestamps to UTC
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # Coerce all numeric columns
    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "direct_radiation", "diffuse_radiation",
        "surface_pressure",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sum direct + diffuse radiation for GHI
    df["solar_insolation"] = df["direct_radiation"].fillna(0) + df["diffuse_radiation"].fillna(0)

    # Derive dewpoint via Magnus formula
    df["dewpoint"] = df.apply(
        lambda row: magnus_dewpoint(row["temperature_2m"], row["relative_humidity_2m"]),
        axis=1,
    )

    # Rename to DB column names
    df = df.rename(columns={
        "temperature_2m":        "outdoor_dry_bulb",
        "relative_humidity_2m":  "outdoor_rel_humidity",
        "wind_speed_10m":        "wind_speed",
        "wind_direction_10m":    "wind_direction",
        "surface_pressure":      "atm_pressure",
    })

    # Always mark as forecast
    df["is_forecast"] = True

    # Filter to future timestamps only unless --force
    if not force:
        now = datetime.now(tz=timezone.utc)
        df = df[df["time"] > now]

    # Drop rows with unparseable timestamps
    df = df.dropna(subset=["time"])

    # Select final columns in DB order
    out_cols = [
        "time", "outdoor_dry_bulb", "outdoor_rel_humidity", "wind_speed",
        "wind_direction", "solar_insolation", "atm_pressure", "dewpoint",
        "is_forecast",
    ]
    df = df[out_cols]

    # Replace NaN with None for psycopg2
    df = df.where(pd.notnull(df), other=None)

    return [tuple(row) for row in df.itertuples(index=False, name=None)]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(force: bool = False):
    now = datetime.now(tz=timezone.utc)
    log.info("Forecast ingest run starting at %s", now.isoformat())

    with get_connection() as conn:
        df = fetch_forecast()
        if df is None or df.empty:
            log.error("No forecast data retrieved -- skipping this run.")
            return

        rows = parse_forecast(df, force=force)
        if not rows:
            log.warning("No future forecast rows to write.")
            return

        n = upsert_rows(conn, rows)
        log.info("Upserted %d forecast rows (horizon: %s to %s).",
                 n,
                 rows[0][0].isoformat() if rows else "n/a",
                 rows[-1][0].isoformat() if rows else "n/a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Open-Meteo -> PostgreSQL forecast ingestion"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upsert all 16 days including timestamps before current time.",
    )
    args = parser.parse_args()

    while True:
        run(force=args.force)
        args.force = False  # only force on first iteration if specified
        log.info("Sleeping 6 hours until next forecast update...")
        time.sleep(RUN_INTERVAL)
