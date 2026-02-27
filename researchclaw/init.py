from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_config
from .storage import StorageManager


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    config_path = next((a for a in args if not a.startswith("--")), "config.yaml")

    checks: dict[str, str] = {}
    success = True
    try:
        cfg = load_config(config_path)
        checks["config"] = f"ok - loaded {config_path}"
    except Exception as e:
        checks["config"] = f"ERROR: {e}"
        success = False
        cfg = None

    if cfg is not None:
        try:
            storage = StorageManager(cfg.base_dir)
            storage.ensure_layout()
            checks["layout"] = f"ok - initialized {Path(cfg.base_dir).resolve()}"
        except Exception as e:
            checks["layout"] = f"ERROR: {e}"
            success = False

    payload = {"success": success, "checks": checks}
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for k, v in checks.items():
            print(f"{k}: {v}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
