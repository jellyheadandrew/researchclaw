# ResearchClaw

Experiment 24/7 while you sleep. Research assistant for projects.

> ⚠️ **Alpha Release** — ResearchClaw is in active development. Core loop works, but expect rough edges. Bug reports and feedback welcome: [GitHub Issues](https://github.com/jellyheadandrew/researchclaw/issues)

## Alpha Status

This project is currently alpha and intended for fast iteration.

Known limitations:
- New-experiment loop behavior is still being hardened.
- Abort handling is not yet uniform across every state.
- Autopilot transition audit logging is still being refined.

## Install

```bash
pipx install researchclaw
```

## Usage

```bash
researchclaw .
researchclaw /path/to/project
researchclaw status
```

## What it does NOT do

- Not a paper writing tool — it runs experiments, not LaTeX

---

Built by [Sookwan Han](https://jellyheadandrew.github.io) · ICCV 2023 Oral · ECCV 2024 Oral · CVPR 2025
