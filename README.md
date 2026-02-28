# ResearchClaw

> Your experiments run while you sleep.
> You approve from your phone.

ResearchClaw is a chat-first research orchestrator that automates the full experiment lifecycle:

1. **Plan** — Brainstorm and refine experiment plans with AI assistance
2. **Implement** — AI auto-generates experiment code from the approved plan
3. **Execute** — Runs experiments and evaluations automatically
4. **Report** — Generates comprehensive analysis reports
5. **Iterate** — Learns from prior trials to guide future experiments

## Key Features
- 11-state FSM with least-privilege access policies
- Fully autonomous experiment loop (implement → execute → fix → retry)
- Ralph loop: fresh AI context per retry for unbiased debugging
- Non-interruptible report generation with EXPERIMENT_LOGS.md tracking
- Intelligent project assimilation and auto-push to GitHub
- Conversational UX — just describe what you want in natural language
- Telegram support for remote experiment management
- Autopilot mode for fully autonomous research cycles

## States
`DECIDE`, `PLAN`, `EXPERIMENT_IMPLEMENT`, `EXPERIMENT_EXECUTE`, `EVAL_IMPLEMENT`, `EVAL_EXECUTE`, `REPORT_SUMMARY`, `VIEW_SUMMARY`, `UPDATE_AND_PUSH`, `SETTINGS`, `RESEARCH`

## Quick Start
```bash
./onboard
./gateway
```

## How It Works

1. At **DECIDE**, tell the agent what you'd like to do (plan experiment, review results, etc.)
2. At **PLAN**, describe your experiment idea. The agent searches for relevant research trends, considers past trial history, and iterates with you on a plan. Say "approve" when ready.
3. After approval, the agent **automatically**: generates code → runs the experiment → fixes failures → generates eval code → runs evaluation → writes a report → returns to DECIDE.
4. The user never needs to write code manually. The AI handles all implementation.

## Runtime Layout
Under `base_dir` (default `./workspace`):
- `projects/{project_name}` — Git-connected codebases
- `sandbox/{YYYYMMDD}/trial_{N:03}/` — Experiment sandboxes (codes, outputs, eval_codes, run.sh, eval.sh)
- `results/{YYYYMMDD}/trial_{N:03}/` — Trial results and REPORT.md
- `references/{YYYYMM}/BRAINSTORM_{DD}.md` — Research brainstorm notes
- `EXPERIMENT_LOGS.md` — Cumulative trial summaries

## Configuration

Edit `config.yaml`:

```yaml
base_dir: "./workspace"

messenger:
  type: stdio        # or: telegram
  # bot_token: ""    # required for telegram
  # chat_id: ""      # required for telegram

planner:
  use_claude: true
  claude_cli_path: claude
  model: claude-sonnet-4-6
  web_search: true
```

## Slash Commands (shortcuts)
Global: `/status`, `/abort`, `/autopilot-start`, `/autopilot-stop`, `/exit`

Most interactions are conversational — just type naturally. Slash commands are available as shortcuts.

## Validation
```bash
python -m researchclaw.init config.yaml
python -m pytest
```

## Telegram Setup
1. Create a bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID
3. Set `messenger.type: telegram`, `messenger.bot_token`, and `messenger.chat_id` in config.yaml
4. Install: `pip install 'researchclaw[telegram]'`
