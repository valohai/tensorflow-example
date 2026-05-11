import json
import os
from pathlib import Path

import requests
# https://app.valohai.com/api/v0/pipelines/

VALOHAI_API_URL = os.environ.get("VALOHAI_API_URL", "https://app.valohai.com/api/v0")

# TODO: Adjust these for the target Valohai project/environment if needed.
PROJECT_ID = os.environ.get("VALOHAI_PROJECT_ID", "018f72b2-ae36-d10b-e14d-a31e0509c619")
ENVIRONMENT_ID = os.environ.get("VALOHAI_ENVIRONMENT_ID", "0167d05d-a1d7-cc02-8256-6455a6ecfa56")

INPUTS_JSON_PATH = Path("/valohai/config/inputs.json")
OUTPUTS_DIR = Path("/valohai/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("VALOHAI_API_TOKEN") or os.environ.get("VH_API_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "Missing API token. Please expose VALOHAI_API_TOKEN or VH_API_TOKEN "
        "as an environment variable in the execution."
    )


def load_dataset_uris() -> list[str]:
    """Read Valohai's generated inputs.json and extract all dataset input URIs."""
    if not INPUTS_JSON_PATH.exists():
        raise FileNotFoundError(f"Could not find {INPUTS_JSON_PATH}")

    with INPUTS_JSON_PATH.open("r", encoding="utf-8") as f:
        inputs_config = json.load(f)

    dataset_input = inputs_config.get("dataset")
    if not dataset_input:
        raise RuntimeError("No 'dataset' input found in /valohai/config/inputs.json")

    files = dataset_input.get("files", [])
    if not files:
        raise RuntimeError("The 'dataset' input does not contain any files")

    uris = []

    for file_info in files:
        # Prefer the original Valohai input URI.
        # This can be azure://..., datum://..., dataset://..., etc.
        uri = file_info.get("uri") or file_info.get("storage_uri")

        if not uri:
            raise RuntimeError(f"Could not find URI for input file: {file_info}")

        uris.append(uri)

    return uris


def build_training_pipeline_payload(dataset_uri: str) -> dict:
    """Build one child training-pipeline payload for one dataset URI."""
    return {
        "edges": [
            {
                "source_node": "preprocess",
                "source_key": "preprocessed_mnist.npz",
                "source_type": "output",
                "target_node": "train",
                "target_type": "input",
                "target_key": "dataset",
            },
            {
                "source_node": "train",
                "source_key": "model*",
                "source_type": "output",
                "target_node": "evaluate",
                "target_type": "input",
                "target_key": "model",
            },
        ],
        "nodes": [
            {
                "name": "preprocess",
                "type": "execution",
                "on_error": "stop-all",
                "edge_merge_mode": "replace",
                "template": {
                    "environment": ENVIRONMENT_ID,
                    "commit": "main",
                    "step": "preprocess-dataset",
                    "image": "python:3.9",
                    "command": "pip install numpy valohai-utils\npython ./preprocess_dataset.py",
                    "inputs": {
                        "dataset": [
                            dataset_uri
                        ]
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                    "priority": 0,
                },
            },
            {
                "name": "train",
                "type": "execution",
                "on_error": "stop-all",
                "edge_merge_mode": "replace",
                "template": {
                    "environment": ENVIRONMENT_ID,
                    "commit": "main",
                    "step": "train-model",
                    "image": "tensorflow/tensorflow:2.6.0",
                    "command": "pip install valohai-utils\npython ./train_model.py {parameters}",
                    # Dataset comes from the preprocess -> train edge, so this stays empty.
                    "inputs": {
                        "dataset": []
                    },
                    "parameters": {
                        "epochs": {
                            "style": "single",
                            "rules": {
                                "value": 5
                            },
                        },
                        "learning_rate": {
                            "style": "single",
                            "rules": {
                                "value": 0.001
                            },
                        },
                    },
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                    "priority": 0,
                },
            },
            {
                "name": "evaluate",
                "type": "execution",
                "on_error": "stop-all",
                "edge_merge_mode": "replace",
                "template": {
                    "environment": ENVIRONMENT_ID,
                    "commit": "main",
                    "step": "batch-inference",
                    "image": "tensorflow/tensorflow:2.6.0",
                    "command": "pip install pillow valohai-utils\npython ./batch_inference.py",
                    "inputs": {
                        "model": [],
                        "images": [
                            "https://valohaidemo.blob.core.windows.net/mnist/four-inverted.png",
                            "https://valohaidemo.blob.core.windows.net/mnist/five-inverted.png",
                            "https://valohaidemo.blob.core.windows.net/mnist/five-normal.jpg",
                        ],
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                    "priority": 0,
                },
            },
        ],
        "project": PROJECT_ID,
        "tags": ["batch-created"],
        "parameters": {
            "epochs_pipeline": {
                "config": {
                    "name": "epochs_pipeline",
                    "default": [
                        5,
                        2
                    ],
                    "targets": [
                        "train.parameter.epochs"
                    ],
                },
                "expression": {
                    "style": "single",
                    "rules": {
                        "value": [
                            5,
                            2
                        ]
                    },
                },
            }
        },
        "title": f"training-pipeline - {dataset_uri}",
    }


def create_pipeline(dataset_uri: str) -> dict:
    response = requests.request(
        url="https://app.valohai.com/api/v0/pipelines/",
        method="POST",
        headers={"Authorization": f"Token {TOKEN}"},
        json=build_training_pipeline_payload(dataset_uri),
    )
    # response = requests.post(
    #     # f"{VALOHAI_API_URL}/pipelines/",
    #     "https://app.valohai.com/api/v0/pipelines/",
    #     headers={
    #         "Authorization": f"Token {TOKEN}",
    #         "Content-Type": "application/json",
    #     },
    #     json=build_training_pipeline_payload(dataset_uri),
    #     timeout=60,
    # )

    if response.status_code == 400:
        raise RuntimeError(response.json())

    response.raise_for_status()
    return response.json()


def main() -> None:
    dataset_uris = load_dataset_uris()

    created_pipelines = []

    for dataset_uri in dataset_uris:
        print(f"Creating training pipeline for dataset URI: {dataset_uri}")
        pipeline = create_pipeline(dataset_uri)

        created_pipelines.append(
            {
                "dataset_uri": dataset_uri,
                "pipeline_id": pipeline["id"],
                "pipeline_url": pipeline.get("url"),
                "display_url": pipeline.get("urls", {}).get("display"),
                "status": pipeline.get("status"),
            }
        )

    pipeline_ids = [item["pipeline_id"] for item in created_pipelines]

    output = {
        "pipeline_ids": pipeline_ids,
        "created_pipelines": created_pipelines,
    }

    output_path = OUTPUTS_DIR / "pipeline_ids.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Valohai metadata
    print(json.dumps({"pipeline_ids": pipeline_ids}))

    print(f"Saved created pipeline IDs to {output_path}")


if __name__ == "__main__":
    main()
