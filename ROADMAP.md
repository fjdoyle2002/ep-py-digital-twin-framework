# Roadmap

This document describes planned features, improvements, and known areas for future development in the ep-py-digital-twin-framework. Contributions toward any of these items are welcome -- see [CONTRIBUTING.md](CONTRIBUTING.md) for guidance.

Items are grouped by theme rather than strict priority. Not all items have a committed timeline.

---

## Data Retrieval

### Generic PostgreSQL Retrieval Agent
The current framework supports Seeq as the primary real-time data source, which requires a commercial Seeq license. A generic PostgreSQL retrieval agent would allow sensor data to be pulled directly from any time-series database table, significantly broadening deployment options and making the framework viable without any third-party historian software. This is a high priority item for accessibility.

### Nationwide Weather Ingestion Support Script
The current `mesonet_ingest.py` support script is specific to the NYS Mesonet REST API. A replacement script targeting the NOAA Automated Surface Observing System (ASOS) network via the Iowa Environmental Mesonet (IEM) public API would provide nationwide coverage with no API key requirement. The NYS Mesonet script would be retained as a regional example.

---

## Security and Configuration

### Credential Management via Environment Variables
Database passwords, historian credentials, and other sensitive values are currently stored in `config.ini`. These should be migrated to system environment variables (e.g., `DT_DB_PASSWORD`, `DT_SEEQ_PASSWORD`) following the pattern already established in the weather ingestion support scripts. The framework should fail explicitly at startup with a clear error message if required environment variables are not set, rather than using blank or default values silently.

---

## Deployment and Operations

### Log Rotation and Configurable Log Level
Long-duration production runs currently write to a single growing log file. Log rotation (daily or by size) should be implemented to prevent unbounded disk usage. Log level should be configurable at runtime without code changes.

### Health Check and Watchdog Endpoint
A lightweight HTTP health check endpoint would allow external monitoring tools to confirm that the digital twin process is running and producing current simulation output. This is particularly useful in NSSM service deployments where silent failures can otherwise go undetected.

### Linux and macOS Deployment Path
The current deployment model relies on NSSM for Windows service management, limiting the framework to Windows environments. A Linux-compatible deployment path using `systemd` unit files, with equivalent documentation to the existing NSSM guide, would broaden the potential user base. Docker containerization is a longer-term option that would support both Linux and cloud deployment scenarios.

---

## Framework Architecture

### Decoupled Simulation and Retrieval Intervals
The EnergyPlus simulation timestep and the real-world sensor retrieval interval are currently tightly coupled. Decoupling these would allow, for example, a 15-minute EnergyPlus timestep with a 1-minute sensor polling interval, or vice versa, providing more flexibility for different building automation system configurations.

### Separated Acquisition and Conversion Pipeline Stages
The current `predictor_function` column in `opc_variables.csv` conflates data acquisition and unit conversion into a single function call. A future refactor should separate these into distinct pipeline stages -- an acquisition stage that retrieves raw values and a conversion stage that applies unit transformations. This will be a natural forcing function when ML model integration is introduced.

### ML Model Integration in the Predictor Layer
The predictor layer currently supports physics-based and rule-based predictor functions. A framework for integrating trained ML models as predictors -- with standardized interfaces for model loading, inference, and fallback behavior -- is a planned extension. This would allow data-driven predictors to supplement or replace physics-based approximations for unmeasured quantities.

---

## Testing and Code Quality

### Unit Test Suite
A minimal unit test suite covering the configuration validator, retrieval layer interfaces, and persistence layer would improve confidence in contributions and make regression testing feasible. The test suite should be runnable without a live EnergyPlus installation or database connection using mocked dependencies.

---

## Documentation

### Example Deployment with Generic PostgreSQL Source
Once the generic PostgreSQL retrieval agent is implemented, the `example/` deployment should be updated to demonstrate sensor ingestion from a local database table rather than requiring a Seeq connection. This would make the example fully self-contained with only EnergyPlus and PostgreSQL as prerequisites.
