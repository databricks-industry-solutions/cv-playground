# DL Dependency Manager - CLIP Fine-tuning Pipeline

CLIP fine-tuning pipeline demonstrating dependency management with a worked end-to-end example.

## Features

- CLIP fine-tuning with distributed training (multi-node, multi-GPU)
- Mosaic Streaming for efficient data loading
- MLflow model logging and Unity Catalog registration
- Model serving endpoint deployment
- Pinned dependencies aligned to DBR 17.3 ML GPU
- Local training/integration test with synthetic data

## Quick Start

### 1. Install Dependencies

```bash
curl -sSL https://install.python-poetry.org | python3 -

poetry install

poetry install --with dev
```

### 2. Working Locally

Run locally, currently this is an integration test really, not an effective training step.

Unit tests not all passing currently but demonstrates local unit tests.

```bash
poetry run python -m dl_dep_mgr.train --local

poetry run pytest tests/ -v
```

### 3. Configure Environment

```bash
cp dev.env.example dev.env
# Edit dev.env with your values
```

### 4. Create Unity Catalog Resources

Before first deployment, create the schema and volumes using values from your `dev.env`:

```sql
-- Replace with your CATALOG and SCHEMA values from dev.env
CREATE SCHEMA IF NOT EXISTS <CATALOG>.<SCHEMA>;
CREATE VOLUME IF NOT EXISTS <CATALOG>.<SCHEMA>.models;
CREATE VOLUME IF NOT EXISTS <CATALOG>.<SCHEMA>.logs;
```

### 5. Deploy to Databricks

```bash
./dev.sh clip_training_job
```


## Project Structure

```
dl_dep_mgr/
├── pyproject.toml              # Poetry config, DBR 17.3 aligned deps
├── poetry.lock                 # Locked dependencies
├── requirements.txt            # Exported for DAB (generated)
├── databricks.yml              # Bundle config
├── dev.env.example             # Configuration template
├── dev.env                     # Your configuration (gitignored)
├── resources/
│   └── jobs.yml                # Job definitions
├── src/dl_dep_mgr/
│   ├── train.py                # Training entry point
│   ├── deploy_serving.py       # Serving deployment
│   ├── model.py                # CLIP wrapper for MLflow
│   ├── data.py                 # Dataset utilities
│   └── config.py               # Configuration
├── tests/
├── dev.sh                      # Deploy + run + logs
└── download_logs.sh            # Download cluster logs
```

## Configuration



These are passed to `databricks bundle deploy` by `dev.sh`. You can also override via CLI:

```bash
databricks bundle deploy --var catalog=my_catalog --var schema=my_schema
```

## Python Version Alignment with DBR

**Critical**: The Python version in `pyproject.toml` must be compatible with your target DBR.

If you see this error during job execution:
```
ERROR: Package 'dl-dep-mgr' requires a different Python: X.X.X not in '...'
```

Update `pyproject.toml` to match your DBR's Python version and regenerate lock file.

## Development Workflow

### Deploy and Run with Logs

```bash
# Full workflow: build, deploy, run, download logs (logs cleaned after).
# Current version picks up the most recent logs - working on a version that outputs the log ID and always captures correctly even with logging failure.
./dev.sh clip_training_job

# Keep logs for inspection
./dev.sh clip_training_job --keep
```

### Download Logs Only

```bash
source dev.env

# Download, analyze, then clean up (default)
./download_logs.sh

# Keep logs for further inspection
./download_logs.sh --keep
```

### Run Individual Commands

```bash
# Validate bundle
databricks bundle validate

# Deploy only
databricks bundle deploy

# Run specific job
databricks bundle run clip_train_only_job
```

## Cluster Requirements 

Any of these can be changed, this is for the exact details of the current example.

For distributed training:
- 2x g5.12xlarge workers (4 GPUs each = 8 total)
- DBR 17.3.x GPU ML Runtime
- `spark.task.resource.gpu.amount = 4`

## Model Serving

After training, the model is:
1. Registered to Unity Catalog with alias "champion"
2. Deployed to a serving endpoint (scale-to-zero enabled)

Query the endpoint (endpoint name format is `clip_<catalog>_<schema>_<model_name>`):

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="clip_<CATALOG>_<SCHEMA>_clip_finetuned",
    dataframe_records=[{"texts": ["a photo of a cat", "a photo of a dog"]}]
)
print(response.predictions)
```
