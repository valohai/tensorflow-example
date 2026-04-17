import requests

resp = requests.request(
    url="https://app.valohai.com/api/v0/executions/",
    method="POST",
    headers={"Authorization": "Token YOUR_TOKEN_HERE"},
    json={
        "project": "01902f1e-bd6c-25f3-a7f0-05255caff855",
        "environment": "01764236-1f69-fea3-392a-be679bf067b3",
        "commit": "8243de97bf8aefbed1a269c000a60139763f9ee9",
        "step": "train-model",
        "image": "docker.io/noorai/dynamic-pipelines-demo:0.1",
        "command": "pip install debugpy\npython ./train_model.py {parameters}",
        "inputs": {
            "dataset": [
                "dataset://{parameter:dataset_name}_train/latest"
            ]
        },
        "parameters": {
            "epochs": {
                "style": "single",
                "rules": {
                    "value": 25
                }
            },
            "learning_rate": {
                "style": "single",
                "rules": {
                    "value": 0.001
                }
            },
            "batch_size": {
                "style": "single",
                "rules": {
                    "value": 64
                }
            },
            "dataset_name": {
                "style": "single",
                "rules": {
                    "value": "all_harbors"
                }
            },
            "debug": {
                "style": "single",
                "rules": {
                    "value": False
                }
            }
        },
        "runtime_config": {},
        "inherit_environment_variables": True,
        "tags": [],
        "time_limit": 0,
        "environment_variables": {},
        "priority": 0
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()