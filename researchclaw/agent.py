from __future__ import annotations

import sys

from .config import load_config
from .orchestrator import ResearchClaw


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    agent = ResearchClaw(config)
    agent.run()


if __name__ == "__main__":
    main()
