# Valohai Batch Wrapper Pipeline

This folder contains a parent Valohai pipeline that creates multiple child `training-pipeline` runs from multiple uploaded `dataset` inputs.

## Files

- `valohai.yaml`
  - Adds a new `batch-training-pipeline`.
- `create_training_pipelines.py`
  - Reads `/valohai/config/inputs.json`.
  - Extracts all URIs from the `dataset` input.
  - Creates one `training-pipeline` per dataset URI through the Valohai API.
  - Saves created pipeline IDs to `/valohai/outputs/pipeline_ids.json`.
- `collect_training_results.py`
  - Reads `pipeline_ids.json`.
  - Pulls each created pipeline with `/api/v0/pipelines/{id}/`.
  - Extracts execution IDs from the pipeline nodes.
  - Pulls execution outputs with `/api/v0/executions/{id}/outputs/?include=download_url`.
  - Downloads outputs into `/valohai/outputs`.

## Required environment variables

Set one of these in the Valohai execution environment:

```bash
VALOHAI_API_TOKEN=<token>
```

or:

```bash
VH_API_TOKEN=<token>
```

Optional overrides:

```bash
VALOHAI_PROJECT_ID=<project-id>
VALOHAI_ENVIRONMENT_ID=<environment-id>
VALOHAI_API_URL=https://app.valohai.com/api/v0
```

## Other Notes

- All child pipeline node commits are set to `main`.
- The parent collector step currently waits for child pipelines by default.
- If you do not want the parent pipeline to wait, set `wait_for_completion` to `false`.
- The YAML edge syntax may need to be adjusted if the repo uses a different Valohai YAML style.
