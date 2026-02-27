# ResearchClaw V2

ResearchClaw V2 is a chat-first research orchestrator with:
- 11-state FSM
- least-privilege state policies
- experiment/eval loop retries with max iteration cutoffs
- non-interruptible report generation
- trial summaries in `EXPERIMENT_LOGS.md`
- project assimilation and auto-push in `UPDATE_AND_PUSH`

## States
`DECIDE`, `PLAN`, `EXPERIMENT_IMPLEMENT`, `EXPERIMENT_EXECUTE`, `EVAL_IMPLEMENT`, `EVAL_EXECUTE`, `REPORT_SUMMARY`, `VIEW_SUMMARY`, `UPDATE_AND_PUSH`, `SETTINGS`, `RESEARCH`

## Quick Start
```bash
./onboard
./gateway
```

## Runtime Layout
Under `base_dir` (default `./workspace`):
- `projects/{project_name}`
- `sandbox/{YYYYMMDD}/trial_{N:03}/codes|outputs|eval_codes|run.sh|eval.sh`
- `results/{YYYYMMDD}/trial_{N:03}/REPORT.md`
- `references/{YYYYMM}/BRAINSTORM_{DD}.md`
- `EXPERIMENT_LOGS.md`

## Core Commands
Global:
- `/status`
- `/abort [reason]`
- `/autopilot-start`
- `/autopilot-stop`
- `/exit`

DECIDE:
- `/plan`
- `/view_summary`
- `/update_and_push`
- `/settings`
- `/research`

PLAN:
- `/plan project <number|name|scratch>`
- `/plan show`
- `/plan approve`

EXPERIMENT_IMPLEMENT / EVAL_IMPLEMENT:
- `/write <path>` then content lines then `/endwrite`
- `/exp run` (experiment)
- `/eval run` (eval)

SETTINGS:
- `/settings show`
- `/settings set <key> <value>`
- `/settings set-project <project> <field> <value>`

RESEARCH:
- `/research cadence <ask|disabled|hourly|6h|daily>`
- `/research brainstorm <text>`
- `/research download <url>`

## Validation
```bash
python -m researchclaw.init config.yaml
python -m pytest
```
