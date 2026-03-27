# ep-py-digital-twin-framework

A Python-only framework for creating live EnergyPlus-based building energy digital twins with real-time sensor integration, PostgreSQL persistence, and optional OPC-UA SCADA connectivity.

## Overview

This framework implements a five-layer architecture for building energy digital twins:

1. **Retrieval Layer** -- ingests real-time sensor data from building automation systems (Seeq historian or direct OPC-UA sources)
2. **Simulator Layer** -- manages a continuously running EnergyPlus simulation, injecting sensor values as actuator overrides
3. **Persistence Layer** -- writes timestamped simulation outputs and sensor readings to a PostgreSQL database
4. **OPC-UA Layer** -- publishes digital twin state as an OPC-UA server, enabling SCADA/HMI connectivity
5. **Custom Layer** -- user-defined callbacks, unit conversions, and physics-based predictors for unmeasured quantities

The framework is designed to be building-agnostic. Configuration is driven entirely by CSV files and a single `config.ini`, with no hardcoded building-specific logic in the core codebase.

## Citation

If you use this framework in your research, please cite:

> Doyle, F.J., et al. (2026). A ZEN Framework for Building Energy Digital Twins. *paper in review*
## Prerequisites

- **EnergyPlus** >= 23.2.0 ([energyplus.net](https://energyplus.net))
- **Python** >= 3.9
- **PostgreSQL** >= 13
- A compatible building IDF model and EPW weather file
- Windows OS (NSSM service deployment; Linux adaptation is straightforward)

## Python Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list. Key dependencies include `eppy`, `opcua`, `psycopg2-binary`, `requests`, and `pandas`.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/fjdoyle2002/ep-py-digital-twin-framework.git
cd ep-py-digital-twin-framework
```

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Configure

Copy the example configuration and edit for your environment:

```bash
cp example/config.ini.example config.ini
```

Edit `config.ini` to set your database credentials, EnergyPlus path, and data source connection details.

Copy the example CSV configuration files from `example/` to your working directory and adjust as needed for your building and sensor set.

### 4. Validate configuration

```bash
python validate_config.py
```

This checks that all required configuration files are present, all referenced signals exist, and the EnergyPlus executable is reachable.

### 5. Run the digital twin

```bash
python digital_twin.py
```

For production deployment as a Windows service, see the NSSM deployment notes in `support_scripts/README_NSSM.md`.

## Configuration Files

The framework is configured through six CSV files plus a `config.ini`. The table below summarizes each:

| File | Purpose |
|---|---|
| `config.ini` | Paths, credentials, database connection, EnergyPlus executable location |
| `signals.csv` | Master list of all signals (sensors, actuators, custom) with metadata |
| `sensors.csv` | Sensor-to-EnergyPlus-actuator mappings for real-time override injection |
| `actuators.csv` | OPC-UA actuator node definitions |
| `custom.csv` | Custom signal definitions with associated predictor function references |
| `opc_devices.csv` | OPC-UA device connection parameters |
| `opc_variables.csv` | OPC-UA variable node definitions and data types |

See the paper (cited above) for a detailed description of each configuration file schema.

## Example Deployment

The `example/` directory contains a minimal working configuration using:

- A standard EnergyPlus reference building model (RefBldgMediumOfficeNew2004)
- An Albany, NY TMY3 weather file (`USA_NY_Albany.County.AP.725180_TMY3.epw`)
- Placeholder configuration CSV files demonstrating the schema

This example requires no proprietary building data and is intended as a starting template.

## Weather Data Support Scripts

The `support_scripts/` directory contains two optional ingestion scripts for populating a companion PostgreSQL weather table:

- `mesonet_ingest.py` -- queries the NYS Mesonet REST API every 15 minutes and upserts observed weather into the database
- `forecast_ingest.py` -- queries the Open-Meteo forecast API every 6 hours and writes 16-day hourly forecasts (no API key required)

Both scripts are designed to run as persistent Windows services via NSSM. See `support_scripts/README_NSSM.md` for deployment instructions.

## Repository Structure

```
ep-py-digital-twin-framework/
  custom/           User-defined callbacks, conversions, and predictors
  opcmodule/        OPC-UA server and device abstraction
  persistence/      PostgreSQL persistence layer
  retrieval/        Sensor data retrieval (Seeq and OPC-UA sources)
  simulator/        EnergyPlus process manager
  support_scripts/  Weather ingestion service scripts (optional)
  example/          Minimal example deployment configuration
  digital_twin.py   Main orchestrator
  validate_config.py Configuration validator
  requirements.txt  Python dependencies
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the following funding sources:

- NYS Energy Research and Development Authority (NYSERDA), Award 30712 (Zero Net Energy Building "ZEN Building")
- NYS Energy Research and Development Authority (NYSERDA), Award 99433 (ZEN Smart Building Partner Aligned Training Hub)
- NYS Energy Research and Development Authority (NYSERDA), Award 160485 (ZEN SMART PATH-SIM)
- New York State Department of Economic Development, Award C250184 (NYS Center for Advanced Technology in Nanoelectronics)
- National Science Foundation, Award 2315307 (Engine Development Award)

The ZEN (Zero Energy Now) building at the SUNY Polytechnic Institute campus in Albany, NY served as the primary development and validation testbed.
