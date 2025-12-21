#!/bin/bash
# Simple script to deploy, run, and download logs for a given job.
# Most interesting part is the logging and the ability to download the logs to a local volume.
# Paired with the cluster logs going to a Unity Catalog Volume, this allows for a local developer experience that is more fully featured.
# Doesn't require Databricks Connect or special setups, just a Databricks CLI and a Unity Catalog Volume with logs.
# Logging extraction in this case is focused on training logs, but could be extended to include other logs as well.
# Of course, modification may be necessary to fit your own use case.
# Usage: ./dev.sh <job_name> [--keep]

set +e

# Source configuration
if [ -f "dev.env" ]; then
    source dev.env
else
    echo "ERROR: dev.env not found. Copy dev.env.example to dev.env and configure."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <job_name> [--keep]"
    echo ""
    echo "Available jobs:"
    grep -E "^    [a-z_]+:" resources/jobs.yml | sed 's/://g' | sed 's/^    /  /' 2>/dev/null || echo "  (run from dl_dep_mgr directory)"
    exit 1
fi

JOB_NAME="$1"
KEEP_LOGS=""
if [[ "$2" == "--keep" ]]; then
    KEEP_LOGS="--keep"
fi

echo "=== Step 1: Export requirements.txt ==="
poetry export -f requirements.txt --without-hashes --output requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: poetry export failed"
fi

echo ""
echo "=== Step 2: Build wheel ==="
poetry build -f wheel
if [ $? -ne 0 ]; then
    echo "ERROR: Wheel build failed"
    exit 1
fi

echo ""
echo "=== Step 3: Deploy bundle ==="
databricks bundle deploy \
    --var "catalog=${CATALOG}" \
    --var "schema=${SCHEMA}" \
    --var "logs_volume=${LOGS_VOLUME}" \
    --var "user_name=${USER_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Bundle deploy failed"
    exit 1
fi

echo ""
echo "=== Step 4: Run job: ${JOB_NAME} ==="
databricks bundle run "${JOB_NAME}"
JOB_EXIT_CODE=$?
if [ $JOB_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "WARNING: Job failed with exit code ${JOB_EXIT_CODE}"
fi

echo ""
echo "=== Step 5: Download logs ==="

# Wait for log delivery (cluster logs can take time to appear)
VOLUME_PATH="/Volumes/${CATALOG}/${SCHEMA}/${LOGS_VOLUME}/cluster_log"
FOUND_LOGS=false

for ATTEMPT in 1 2 3 4 5 6 7 8 9 10; do
    echo "Waiting for logs (attempt $ATTEMPT/10)..."
    
    NEWEST=$(databricks fs ls "dbfs:${VOLUME_PATH}" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]' | sort -r | head -1)
    if [ -n "$NEWEST" ]; then
        if databricks fs ls "dbfs:${VOLUME_PATH}/${NEWEST}/driver" &>/dev/null; then
            FOUND_LOGS=true
            break
        fi
    fi
    sleep 10
done

if [ "$FOUND_LOGS" = true ]; then
    ./download_logs.sh $KEEP_LOGS
else
    echo "WARNING: Logs not available yet"
fi

echo ""
echo "=== Done ==="
