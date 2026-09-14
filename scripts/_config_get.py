"""Read one dotted key out of config.yaml. Used by download_models.sh."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _config_get.py <dotted.key>", file=sys.stderr)
        return 2

    config_path = REPO_ROOT / "config.yaml"
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        # The caller substitutes a default; a missing config is not fatal here.
        print("")
        return 0

    for key in sys.argv[1].split("."):
        if not isinstance(data, dict):
            print("")
            return 0
        data = data.get(key)

    print(data if data is not None else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
