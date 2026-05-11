import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


VALOHAI_API_URL = os.environ.get("VALOHAI_API_URL", "https://app.valohai.com/api/v0")

INPUT_DIR = Path("/valohai/inputs/pipeline_ids")
OUTPUTS_DIR = Path("/valohai/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("VALOHAI_API_TOKEN") or os.environ.get("VH_API_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "Missing API token. Please expose VALOHAI_API_TOKEN or VH_API_TOKEN "
        "as an environment variable in the execution."
    )


def get_wait_for_completion() -> bool:
    value = os.environ.get("VH_PARAMETER_WAIT_FOR_COMPLETION", "true")
    return value.lower() in {"1", "true", "yes"}


def find_pipeline_ids_file() -> Path:
    candidates = list(INPUT_DIR.glob("*.json"))

    if not candidates:
        raise FileNotFoundError(f"No JSON file found under {INPUT_DIR}")

    if len(candidates) > 1:
        print(f"Multiple JSON files found, using first one: {candidates[0]}")

    return candidates[0]


def load_pipeline_ids() -> list[str]:
    path = find_pipeline_ids_file()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pipeline_ids = data.get("pipeline_ids")

    if not pipeline_ids:
        raise RuntimeError(f"No pipeline_ids found in {path}")

    return pipeline_ids


def api_get(path: str) -> dict | list:
    response = requests.get(
        f"{VALOHAI_API_URL}{path}",
        headers={
            "Authorization": f"Token {TOKEN}",
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def get_pipeline(pipeline_id: str) -> dict:
    return api_get(f"/pipelines/{pipeline_id}/")


def wait_until_pipeline_finished(pipeline_id: str) -> dict:
    terminal_statuses = {
        "complete",
        "error",
        "stopped",
        "stopping",
    }

    while True:
        pipeline = get_pipeline(pipeline_id)
        status = pipeline.get("status")

        print(f"Pipeline {pipeline_id} status: {status}")

        if status in terminal_statuses:
            return pipeline

        time.sleep(30)


def extract_execution_ids(pipeline: dict) -> list[str]:
    execution_ids = []

    nodes = pipeline.get("nodes", [])

    # Some API examples show nodes as a dict, but in practice this may be a list.
    if isinstance(nodes, dict):
        nodes = [nodes]

    for node in nodes:
        execution = node.get("execution")

        if not execution:
            continue

        execution_id = execution.get("id")

        if execution_id:
            execution_ids.append(execution_id)

    return execution_ids


def get_execution_outputs(execution_id: str) -> list[dict]:
    data = api_get(f"/executions/{execution_id}/outputs/?include=download_url")

    if isinstance(data, dict):
        # In case the endpoint is paginated.
        if "results" in data:
            return data["results"]

        # Fallback if API returns object-like response.
        if "outputs" in data:
            return data["outputs"]

    if isinstance(data, list):
        return data

    raise RuntimeError(f"Unexpected outputs response for execution {execution_id}: {data}")


def safe_filename_from_output(output: dict, execution_id: str, index: int) -> str:
    name = (
        output.get("name")
        or output.get("filename")
        or output.get("file_name")
        or output.get("path")
    )

    if name:
        name = Path(name).name
    else:
        uri = output.get("uri") or output.get("storage_uri") or output.get("download_url")
        if uri:
            parsed = urlparse(uri)
            name = Path(parsed.path).name

    if not name:
        name = f"output_{index}"

    return f"{execution_id}_{name}"


def download_output(output: dict, destination: Path) -> None:
    download_url = output.get("download_url")

    if not download_url:
        raise RuntimeError(f"Output does not include download_url: {output}")

    with requests.get(download_url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    wait_for_completion = get_wait_for_completion()
    pipeline_ids = load_pipeline_ids()

    summary = {
        "pipelines": [],
    }

    for pipeline_id in pipeline_ids:
        print(f"Processing pipeline {pipeline_id}")

        if wait_for_completion:
            pipeline = wait_until_pipeline_finished(pipeline_id)
        else:
            pipeline = get_pipeline(pipeline_id)

        execution_ids = extract_execution_ids(pipeline)

        pipeline_summary = {
            "pipeline_id": pipeline_id,
            "pipeline_status": pipeline.get("status"),
            "execution_ids": execution_ids,
            "downloaded_outputs": [],
        }

        for execution_id in execution_ids:
            outputs = get_execution_outputs(execution_id)

            for index, output in enumerate(outputs):
                if not output.get("download_url"):
                    print(f"Skipping output without download_url: {output}")
                    continue

                filename = safe_filename_from_output(output, execution_id, index)
                destination = OUTPUTS_DIR / filename

                print(f"Downloading output from execution {execution_id} to {destination}")
                download_output(output, destination)

                pipeline_summary["downloaded_outputs"].append(
                    {
                        "execution_id": execution_id,
                        "filename": filename,
                        "path": str(destination),
                        "source": output.get("uri") or output.get("storage_uri"),
                    }
                )

        summary["pipelines"].append(pipeline_summary)

    summary_path = OUTPUTS_DIR / "batch_results_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "collected_pipeline_count": len(summary["pipelines"]),
        "downloaded_output_count": sum(
            len(item["downloaded_outputs"])
            for item in summary["pipelines"]
        ),
    }))

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
