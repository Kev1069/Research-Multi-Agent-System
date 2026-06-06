import yaml
from pathlib import Path

def load_prompt(name: str) -> str:
    path = Path(__file__).parent / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["prompt"]