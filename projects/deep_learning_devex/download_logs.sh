#!/bin/bash
# Usage: ./download_logs.sh [--keep]

set -e

# Require env vars (no defaults)
: "${CATALOG:?Set CATALOG in dev.env or environment}"
: "${SCHEMA:?Set SCHEMA in dev.env or environment}"
: "${LOGS_VOLUME:?Set LOGS_VOLUME in dev.env or environment}"

VOLUME_PATH="/Volumes/${CATALOG}/${SCHEMA}/${LOGS_VOLUME}/cluster_log"
LOCAL_DIR="./logs_downloaded"
CLEAN_AFTER=true

if [[ "$1" == "--keep" ]]; then
    CLEAN_AFTER=false
fi

mkdir -p "${LOCAL_DIR}"

echo "=== Listing clusters ==="
databricks fs ls "dbfs:${VOLUME_PATH}" 2>&1 || {
    echo "No cluster logs found at ${VOLUME_PATH}"
    exit 1
}

NEWEST_CLUSTER=$(databricks fs ls "dbfs:${VOLUME_PATH}" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]' | sort -r | head -1)

if [ -z "$NEWEST_CLUSTER" ]; then
    echo "No cluster directories found"
    exit 1
fi

echo "Most recent cluster: ${NEWEST_CLUSTER}"

# Download driver logs
if databricks fs ls "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/driver" &>/dev/null; then
    mkdir -p "${LOCAL_DIR}/driver"
    echo "Downloading driver logs..."
    databricks fs cp "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/driver/stderr" "${LOCAL_DIR}/driver/stderr" 2>/dev/null || true
    databricks fs cp "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/driver/stdout" "${LOCAL_DIR}/driver/stdout" 2>/dev/null || true
fi

# Download executor logs
if databricks fs ls "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/executor" &>/dev/null; then
    mkdir -p "${LOCAL_DIR}/executor"
    echo "Downloading executor logs..."
    
    APPS=$(databricks fs ls "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/executor/" 2>/dev/null | awk '{print $NF}' | grep -E '^app-' || true)
    for APP in $APPS; do
        if [ -n "$APP" ]; then
            EXECUTORS=$(databricks fs ls "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/executor/${APP}/" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]+' || true)
            for EXEC in $EXECUTORS; do
                if [ -n "$EXEC" ]; then
                    DEST_DIR="${LOCAL_DIR}/executor/${APP}_${EXEC}"
                    mkdir -p "${DEST_DIR}"
                    databricks fs cp "dbfs:${VOLUME_PATH}/${NEWEST_CLUSTER}/executor/${APP}/${EXEC}/stderr" "${DEST_DIR}/stderr" 2>/dev/null || true
                fi
            done
        fi
    done
fi

if [ -z "$(ls -A ${LOCAL_DIR} 2>/dev/null)" ]; then
    echo "No log files found"
    exit 1
fi

echo -e "\n=== Log Files ==="
find "${LOCAL_DIR}" -type f

echo -e "\n=== ERRORS ==="
grep -rh -E "(ERROR|FATAL|Exception|Traceback|NCCL|failed|DistStoreError)" "${LOCAL_DIR}" 2>/dev/null | grep -v "^Binary" | head -60 || echo "None found"

echo -e "\n=== RANK STATUS ==="
grep -rh "\[Rank" "${LOCAL_DIR}" 2>/dev/null | head -20 || echo "None found"

echo -e "\n=== TRAINING PROGRESS ==="
grep -rh -E "(Epoch|Loss:|Batch)" "${LOCAL_DIR}" 2>/dev/null | head -20 || echo "None found"

echo -e "\n=== DRIVER STDERR (last 50 lines) ==="
if [ -f "${LOCAL_DIR}/driver/stderr" ]; then
    tail -50 "${LOCAL_DIR}/driver/stderr"
else
    echo "Not found"
fi

if [ "$CLEAN_AFTER" = true ]; then
    echo -e "\n=== Cleaning up ==="
    rm -rf "${LOCAL_DIR}"
fi

echo -e "\n=== Done ==="
