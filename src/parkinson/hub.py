"""Load Skore Hub credentials from the repo-level .skore config file."""

import json
from pathlib import Path

from parkinson import PROJECT_ROOT


def load_skore_credentials() -> dict:
    """Read .skore and return the config dict (api_key, workspace, hub_url, …)."""
    skore_file = PROJECT_ROOT / ".skore"
    with open(skore_file) as f:
        cfg = json.load(f)
    # Export the API key so skore.login(mode="hub") picks it up automatically.
    import os
    os.environ.setdefault("SKORE_API_KEY", cfg["api_key"])
    return cfg
