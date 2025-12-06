"""
Fetch model input schemas from Databricks Model Serving endpoints.

This queries the Databricks API to get the exact input schema
that each model expects, helping debug schema mismatch errors.

Usage:
    python get_model_schemas.py
"""

import os
import sys
import json
import requests

# Get token from environment or command line
if len(sys.argv) > 1:
    DATABRICKS_TOKEN = sys.argv[1]
else:
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

WORKSPACE_URL = "https://3866123326870389.9.gcp.databricks.com"

# Endpoint names (without the /invocations suffix)
ENDPOINTS = [
    "01-opponent-modeling-preflop",
    "01-opponent-modeling-flop",
    "01-opponent-modeling-turn",
    "01-opponent-modeling-river",
    "02-profit-modeling-preflop",
    "02-profit-modeling-flop",
    "02-profit-modeling-turn",
    "02-profit-modeling-river",
    "03-policy-training-preflop",
    "03-policy-training-flop",
    "03-policy-training-turn",
    "03-policy-training-river",
]


def get_endpoint_schema(endpoint_name: str) -> dict:
    """Fetch the input schema for a serving endpoint."""
    if not DATABRICKS_TOKEN:
        return {"error": "DATABRICKS_TOKEN not set"}

    # API to get serving endpoint details
    url = f"{WORKSPACE_URL}/api/2.0/serving-endpoints/{endpoint_name}"

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()

            # Extract schema information
            config = data.get("config", {})
            served_entities = config.get("served_entities", [])

            schema_info = {
                "name": data.get("name"),
                "state": data.get("state", {}).get("ready"),
            }

            if served_entities:
                entity = served_entities[0]
                schema_info["entity_name"] = entity.get("entity_name")
                schema_info["entity_version"] = entity.get("entity_version")

                # Try to get the model signature
                model_name = entity.get("entity_name")
                model_version = entity.get("entity_version")

                if model_name and model_version:
                    # Query the model version for signature
                    model_url = f"{WORKSPACE_URL}/api/2.0/mlflow/model-versions/get"
                    model_params = {"name": model_name, "version": model_version}
                    model_response = requests.get(
                        model_url,
                        headers=headers,
                        params=model_params,
                        timeout=30
                    )

                    if model_response.status_code == 200:
                        model_data = model_response.json()
                        model_version_info = model_data.get("model_version", {})

                        # Check for signature in run data
                        run_id = model_version_info.get("run_id")
                        if run_id:
                            schema_info["run_id"] = run_id

                            # Get artifact to find MLmodel file with signature
                            artifacts_url = f"{WORKSPACE_URL}/api/2.0/mlflow/artifacts/list"
                            artifacts_params = {"run_id": run_id, "path": "model"}
                            artifacts_response = requests.get(
                                artifacts_url,
                                headers=headers,
                                params=artifacts_params,
                                timeout=30
                            )

                            if artifacts_response.status_code == 200:
                                # Try to get MLmodel file which contains signature
                                mlmodel_url = f"{WORKSPACE_URL}/api/2.0/mlflow/artifacts/get"
                                mlmodel_params = {"run_id": run_id, "path": "model/MLmodel"}
                                mlmodel_response = requests.get(
                                    mlmodel_url,
                                    headers=headers,
                                    params=mlmodel_params,
                                    timeout=30
                                )

                                if mlmodel_response.status_code == 200:
                                    schema_info["mlmodel_raw"] = mlmodel_response.text[:2000]

            return schema_info
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "message": response.text[:500]
            }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 80)
    print("Databricks Model Schema Fetcher")
    print("=" * 80)

    if not DATABRICKS_TOKEN:
        print("\n[ERROR] DATABRICKS_TOKEN not set!")
        print("   Set it with: set DATABRICKS_TOKEN=your_token_here")
        print("   Or pass as argument: python get_model_schemas.py YOUR_TOKEN")
        return

    print(f"\n[OK] Token configured (length: {len(DATABRICKS_TOKEN)})")
    print(f"Workspace: {WORKSPACE_URL}")
    print("\nFetching endpoint schemas...\n")

    all_schemas = {}

    for endpoint in ENDPOINTS:
        print(f"\n--- {endpoint} ---")
        schema = get_endpoint_schema(endpoint)
        all_schemas[endpoint] = schema

        if "error" in schema:
            print(f"  [ERROR] {schema.get('error')}")
            if "message" in schema:
                print(f"  {schema['message'][:200]}")
        else:
            print(f"  State: {schema.get('state')}")
            print(f"  Entity: {schema.get('entity_name')} v{schema.get('entity_version')}")
            if "run_id" in schema:
                print(f"  Run ID: {schema.get('run_id')}")
            if "mlmodel_raw" in schema:
                print(f"  MLmodel snippet: {schema['mlmodel_raw'][:500]}...")

    # Save full results to file
    output_file = "model_schemas.json"
    with open(output_file, "w") as f:
        json.dump(all_schemas, f, indent=2)
    print(f"\n\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
