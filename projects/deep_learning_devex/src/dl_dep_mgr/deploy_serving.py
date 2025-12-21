"""Deploy CLIP model to serving endpoint."""

import argparse
import time
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    EndpointStateReady,
)

from dl_dep_mgr.config import ServingConfig


def get_model_version(config: ServingConfig) -> str:
    """Get the model version from alias or latest."""
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    try:
        # Try to get version by alias
        version_info = client.get_model_version_by_alias(config.uc_model_name, config.model_alias)
        print(f"Resolved alias '{config.model_alias}' to version {version_info.version}")
        return str(version_info.version)
    except Exception as e:
        print(f"Could not resolve alias '{config.model_alias}': {e}")
        # Fall back to latest version
        versions = client.search_model_versions(f"name='{config.uc_model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{config.uc_model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        print(f"Using latest version: {latest.version}")
        return str(latest.version)


def deploy_endpoint(config: ServingConfig, model_version: str) -> dict:
    """Deploy or update serving endpoint."""
    w = WorkspaceClient()
    endpoint_name = config.serving_endpoint_name

    print(f"=== Deploying Model Serving Endpoint ===")
    print(f"Model: {config.uc_model_name}")
    print(f"Version: {model_version}")
    print(f"Endpoint: {endpoint_name}")

    # Check if endpoint exists
    endpoint_exists = False
    try:
        existing = w.serving_endpoints.get(endpoint_name)
        endpoint_exists = True
        print(
            f"Existing endpoint found. State: {existing.state.ready if existing.state else 'Unknown'}"
        )
    except Exception:
        print("No existing endpoint found. Creating new one.")

    # Define served entity config
    served_entity = ServedEntityInput(
        entity_name=config.uc_model_name,
        entity_version=model_version,
        scale_to_zero_enabled=config.scale_to_zero,
        workload_size=config.workload_size,
    )

    if endpoint_exists:
        # Update existing endpoint
        print("Updating endpoint configuration...")
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity],
        )
    else:
        # Create new endpoint
        print("Creating new endpoint...")
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )

    # Wait for endpoint to be ready
    print("Waiting for endpoint to be ready...")
    max_wait_seconds = 600
    poll_interval = 15
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise TimeoutError(f"Endpoint did not become ready within {max_wait_seconds}s")

        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state

        if state and state.ready == EndpointStateReady.READY:
            print(f"Endpoint is READY!")
            break

        config_update = state.config_update if state else None
        print(
            f"  Status: ready={state.ready if state else 'Unknown'}, config_update={config_update}"
        )

        if config_update == "UPDATE_FAILED":
            raise RuntimeError("Endpoint update failed")

        time.sleep(poll_interval)

    # Return deployment info
    final_endpoint = w.serving_endpoints.get(endpoint_name)
    return {
        "endpoint_name": endpoint_name,
        "model_name": config.uc_model_name,
        "model_version": model_version,
        "state": str(final_endpoint.state.ready) if final_endpoint.state else "Unknown",
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy CLIP model to serving endpoint")
    # UC paths - REQUIRED, defaults in databricks.yml
    parser.add_argument("--catalog", type=str, required=True, help="UC catalog name")
    parser.add_argument("--schema", type=str, required=True, help="UC schema name")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    # Serving config - have sensible defaults
    parser.add_argument("--model-alias", type=str, default="champion", help="Model alias")
    parser.add_argument("--endpoint-name", type=str, default=None, help="Override endpoint name")
    parser.add_argument(
        "--scale-to-zero", action="store_true", default=True, help="Enable scale to zero"
    )
    parser.add_argument("--no-scale-to-zero", action="store_false", dest="scale_to_zero")
    parser.add_argument("--workload-size", type=str, default="Small", help="Workload size")
    parser.add_argument("--local", action="store_true", help="Skip deployment (local testing)")

    args = parser.parse_args()

    if args.local:
        print("=== Local Mode: Skipping deployment ===")
        print("In production, this would deploy to a serving endpoint.")
        return

    config = ServingConfig(
        catalog=args.catalog,
        schema=args.schema,
        model_name=args.model_name,
        model_alias=args.model_alias,
        endpoint_name=args.endpoint_name,
        scale_to_zero=args.scale_to_zero,
        workload_size=args.workload_size,
    )

    # Get model version
    model_version = get_model_version(config)

    # Deploy endpoint
    result = deploy_endpoint(config, model_version)

    print("\n=== Deployment Complete ===")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
