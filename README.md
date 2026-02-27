# ResearchClaw

AI-powered research collaborator for ML experiment workflows.

ResearchClaw runs on your GPU server and automates the trial lifecycle: it proposes code changes in isolated sandboxes, runs experiments, monitors GPU utilization and process health, generates reports, and manages git merges — all communicated via Telegram, Slack, or your terminal. Nothing in your codebase is ever changed without your explicit approval.

## Features

- **Sandboxed experiments** — all code changes happen in isolated copies of your codebase; the original is never modified without approval
- **Multi-messenger** — Telegram (recommended for VMs, no public IP needed), Slack (native Socket Mode), or stdio for local development
- **Multi-LLM** — Claude CLI (recommended), Anthropic API, with stubs for OpenAI and Ollama
- **Automatic environment management** — copy-on-write Python venvs so each trial can install packages without side effects
- **GPU monitoring** — real-time experiment tracking with `nvidia-smi`, hang detection, NaN detection
- **Full audit trail** — REPORT.md for every trial, git commits for approved changes, append-only trial ledger

## Quick Start

```bash
git clone https://github.com/<your-org>/ResearchClaw.git
cd ResearchClaw
./onboard          # interactive setup wizard
./gateway           # start the agent
```

The `onboard` script handles everything:
1. Creates a Python virtual environment and installs dependencies
2. Optionally discovers your Telegram chat ID
3. Walks you through configuration (messenger, LLM provider, base directory)
4. Validates the setup end-to-end

Then start the agent:

```bash
./gateway                  # foreground (Ctrl-C to stop)
./gateway --detach         # background via tmux
./gateway --status         # check if running
./gateway --stop           # stop background agent
```

## Requirements

- **Python 3.10+**
- **Git**
- One of:
  - [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`, then `claude login`)
  - Anthropic API key (set in `.env`)
- *(Optional)* NVIDIA GPU + drivers — for GPU monitoring during experiments
- *(Optional)* tmux — for running the agent in the background on remote servers

## How It Works

```
You: "start a new trial: reduce learning rate"
        |
        v
  +--------------+     +---------------+
  |  Copy code   |---->|   Sandbox     |  (isolated working copy)
  |  to sandbox  |     |  trial_001/   |
  +--------------+     +-------+-------+
                                |
                    Agent proposes changes
                    You approve/reject each
                                |
                       +--------v--------+
                       | Run experiment  |  (subprocess, monitored)
                       +--------+--------+
                                |
                    Watcher monitors: GPU, hangs, NaN
                                |
                       +--------v--------+
                       |   REPORT.md     |  (auto-generated summary)
                       +--------+--------+
                                |
              +-----------------+-----------------+
              v                 v                 v
          Approve           Reject            Continue
       (merge + commit)    (discard)      (keep iterating)
```

## Directory Layout

After setup, your workspace looks like this:

| Path | Purpose |
|---|---|
| `config.yaml` | Non-secret configuration (messenger, LLM, paths) |
| `.env` | Secrets — API keys, bot tokens (gitignored) |
| `.env.example` | Template showing all available secrets |
| `onboard` | First-time setup wizard |
| `gateway` | Start / stop / check the agent |
| `setup_tui.py` | Interactive TUI configuration wizard |
| `researchclaw/` | Python package (agent, LLM, messenger, etc.) |
| `<base_dir>/github_codes/` | Your research code (read-only during trials) |
| `<base_dir>/sandbox/` | Trial working copies (auto-created) |
| `<base_dir>/experiment_reports/` | Logs, metrics, and REPORT.md per trial |
| `<base_dir>/reference/` | Papers, docs, notes for agent context (you manage this) |

> `base_dir` defaults to `./workspace`. You can set it to any path during setup.

## Configuration

The interactive setup wizard (`./onboard`) is the easiest way to configure ResearchClaw. For manual configuration:

1. Copy `.env.example` to `.env` and fill in your secrets
2. Edit `config.yaml` — set `base_dir`, messenger type, and LLM provider
3. Run `python -m researchclaw.init config.yaml` to validate

See [TUTORIALS.md](TUTORIALS.md) for detailed walkthroughs:
- **Scenario 1 — stdio** — local development, no external services
- **Scenario 2 — Telegram** — recommended for remote/VM deployments
- **Scenario 3 — Slack** — team workspace integration

## Development

```bash
# Set up dev environment
./onboard

# Run tests
python -m pytest researchclaw/tests/ -v

# Validate configuration
python -m researchclaw.init config.yaml

# Find your Telegram chat ID
python -m researchclaw.get_chat_id
```

## Architecture

| Module | Purpose |
|---|---|
| `agent.py` | Main loop and state machine (IDLE, RESEARCH, EXECUTE, AWAITING_APPROVAL) |
| `llm.py` | Pluggable LLM providers (Claude CLI, Anthropic API, OpenAI, Ollama) |
| `messenger.py` | Communication layer (Telegram, Slack Socket Mode, stdio) |
| `sandbox_manager.py` | Trial lifecycle — create, finalize, reactivate sandboxes |
| `env_manager.py` | Copy-on-write Python environment management |
| `runner.py` | Subprocess execution with environment activation |
| `watcher.py` | Process monitoring — GPU, hangs, NaN, status updates |
| `access_control.py` | Three-tier path security (always-writable, trial-scoped, read-only) |
| `git_manager.py` | Git operations — diff, merge, commit, push (always with approval) |
| `summarizer.py` | Auto-generated REPORT.md with metrics and LLM analysis |
| `config.py` | Configuration loading from `config.yaml` + `.env` |
| `init.py` | Setup validator — checks deps, dirs, connectivity |

## Security Model

- **Path validation**: All file I/O goes through `PathValidator`, which enforces a three-tier access model (always-writable agent memory, trial-scoped sandbox writes, read-only codebase)
- **Shell command validation**: Mutative git operations are blocked at the shell level; all commands are shown and confirmed before execution
- **Sandbox isolation**: Experiments run in copies of your code, never the original
- **Explicit approval**: Every code change, command execution, and merge requires your confirmation

## License

Apache 2.0 — see [LICENSE](LICENSE).
