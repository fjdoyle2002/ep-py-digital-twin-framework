# Deploying Weather Ingestion Scripts as Windows Services via NSSM

The two weather ingestion scripts (`mesonet_ingest.py` and `forecast_ingest.py`) are designed to run as persistent background services on Windows. NSSM (Non-Sucking Service Manager) provides a straightforward way to wrap a Python script as a recoverable Windows service.

## Prerequisites

- NSSM installed and available on your PATH. Download from [nssm.cc](https://nssm.cc).
- Python virtual environment created and dependencies installed (see main README).
- A PostgreSQL database with the weather table schema in place.

## Installing the Mesonet Ingestion Service

Open an Administrator Command Prompt and run:

```cmd
nssm install MesonetIngest "C:\path\to\venv\Scripts\python.exe" "C:\path\to\support_scripts\mesonet_ingest.py"
nssm set MesonetIngest AppDirectory "C:\path\to\support_scripts"
nssm set MesonetIngest AppStdout "C:\path\to\logs\mesonet_ingest.log"
nssm set MesonetIngest AppStderr "C:\path\to\logs\mesonet_ingest_err.log"
nssm set MesonetIngest Start SERVICE_AUTO_START
nssm set MesonetIngest AppRestartDelay 10000
nssm start MesonetIngest
```

## Installing the Forecast Ingestion Service

```cmd
nssm install ForecastIngest "C:\path\to\venv\Scripts\python.exe" "C:\path\to\support_scripts\forecast_ingest.py"
nssm set ForecastIngest AppDirectory "C:\path\to\support_scripts"
nssm set ForecastIngest AppStdout "C:\path\to\logs\forecast_ingest.log"
nssm set ForecastIngest AppStderr "C:\path\to\logs\forecast_ingest_err.log"
nssm set ForecastIngest Start SERVICE_AUTO_START
nssm set ForecastIngest AppRestartDelay 10000
nssm start ForecastIngest
```

## Managing the Services

```cmd
nssm status MesonetIngest       # check service status
nssm restart MesonetIngest      # restart after config changes
nssm stop MesonetIngest         # stop the service
nssm remove MesonetIngest confirm  # uninstall the service
```

The same commands apply to `ForecastIngest`.

## Recovery Behavior

NSSM will automatically restart the service after a crash with the delay specified by `AppRestartDelay` (in milliseconds). The 10-second delay above prevents rapid restart loops if there is a persistent error at startup. Check the log files first if a service fails to stay running.

## Updating the Scripts

After editing a script, simply restart the corresponding service:

```cmd
nssm restart MesonetIngest
```

No reinstallation is needed unless you change the Python executable path or working directory.
