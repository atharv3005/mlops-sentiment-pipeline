import re
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_params(path: str = "params.yaml") -> dict:
    with open(ROOT / path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace. Kept identical between
    training and serving so the model never sees train/serve skew."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
