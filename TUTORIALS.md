# ResearchClaw Setup Tutorials

Three scenarios are covered, from simplest to most involved:

- **[Scenario 1 — stdio](#scenario-1--stdio-local-no-external-services)** — no accounts, no credentials; type in your terminal
- **[Scenario 2 — Telegram](#scenario-2--telegram-recommended-for-remotevm)** — chat from your phone or desktop app; recommended for GPU servers
- **[Scenario 3 — Slack](#scenario-3--slack-via-openclaw-advanced)** — team workspace integration via OpenClaw

---

## Before You Start

These steps apply to every scenario.

> **Path placeholders used in this guide:**
> - `<PROJECT_DIR>` — where you cloned ResearchClaw (the directory containing `config.yaml`)
> - `<BASE_DIR>` — the `base_dir` value from `config.yaml` (default: `./workspace`)

### 1. Understand the directory layout

ResearchClaw uses two separate locations:

| Path | Purpose |
|---|---|
| `<PROJECT_DIR>/` | The agent code and its `config.yaml` |
| `<BASE_DIR>/` | Your research working area (`base_dir` in config) |
| `<BASE_DIR>/github_codes/` | Your research code lives here |
| `<BASE_DIR>/reference/` | Context files you add manually: papers, API docs, notes (auto-created, read-only to agent) |
| `<BASE_DIR>/MEMORY.md` | Agent's persistent memory — written by the agent, survives restarts (auto-created) |
| `<BASE_DIR>/sandbox/` | Isolated trial copies (auto-created) |
| `<BASE_DIR>/experiment_reports/` | Logs and reports (auto-created) |

### 2. Put some code in `github_codes/`

The agent needs a codebase to experiment on. Clone or copy a repository there:

```bash
# Option A: clone your project
git clone https://github.com/you/your-project.git <BASE_DIR>/github_codes/

# Option B: copy an existing local project
cp -r /path/to/your/project <BASE_DIR>/github_codes/

# Option C: for a quick smoke-test, create a minimal script
mkdir -p <BASE_DIR>/github_codes
cat > <BASE_DIR>/github_codes/train.py << 'EOF'
import time, random
print("Training started")
for epoch in range(5):
    loss = 1.0 / (epoch + 1) + random.uniform(0, 0.05)
    print(f"Epoch {epoch+1}: loss={loss:.4f}")
    time.sleep(1)
print("Done")
EOF
```

> **Note:** During a trial, `github_codes/` is read-only for the agent — all changes happen
> in the sandbox. On approval, the agent merges the sandbox changes back into `github_codes/`,
> commits them, and prompts you to push to your remote. Nothing in `github_codes/` is ever
> changed without your explicit approval.
>
> **Note:** `reference/` is also read-only. Drop anything you want the agent to use as context:
> papers (PDF or text), API documentation, configuration guides, notes, external codebases.
> The agent reads from it proactively when interpreting your directions. You manage its contents
> manually — the agent never writes there.

### 3. Activate the virtual environment

```bash
source <PROJECT_DIR>/researchclaw/.venv/bin/activate
```

Verify it worked — your prompt should show `(.venv)`.

### 4. Choose an LLM provider

Open `config.yaml` and look at the `llm:` section. Two options are supported:

**Option A — Claude CLI (default, recommended)**

Requires the `claude` CLI to be installed and logged in. No API key needed.

```bash
# Check if it's available
claude --version
# If missing: install from https://claude.ai/code, then run: claude login
```

Config (already set in `config.yaml`):
```yaml
llm:
  provider: claude_cli
  model: claude-sonnet-4-6
```

**Option B — Anthropic API key**

If you'd rather use a direct API key, edit `config.yaml`:
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
```

Then create `<PROJECT_DIR>/.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Scenario 1 — stdio (Local, No External Services)

**When to use:** Developing locally, testing the agent on your own machine, or when you don't want to set up a bot.

**How it works:** The agent reads from `stdin` and prints to `stdout` in your terminal. You type your messages; the agent responds inline.

**Limitation:** You must keep the terminal open. If the connection drops (SSH disconnect, etc.), the agent stops.

---

### Step 1 — Verify config.yaml

Open `<PROJECT_DIR>/config.yaml`. The messenger section should already read:

```yaml
messenger:
  type: stdio
```

No further changes needed.

### Step 2 — (Optional) Run the validator

This checks that all dependencies and directories are in order:

```bash
python -m researchclaw.init <PROJECT_DIR>/config.yaml
```

Expected output:
```
  ✓  config           ok — loaded from config.yaml
  ✓  directories      ok — created: sandbox, experiment_reports
  ✓  claude_settings  ok — wrote .claude/settings.json
  ✓  git              ok — git version 2.x.x
  ✓  claude_cli       ok — claude 1.x.x
  ✓  nvidia_smi       ok — Tesla T4, 15 GB  (or: not available — CPU-only mode)
  ✓  messenger        ok — sent test message via stdio
  ✓  llm              ok — LLM responded

All checks passed.  Start the agent with:
  python -m researchclaw.agent <PROJECT_DIR>/config.yaml
```

If any check fails, the error message tells you exactly what to fix.

### Step 3 — Start the agent

```bash
python -m researchclaw.agent <PROJECT_DIR>/config.yaml
```

You will immediately see:

```
[ResearchClaw] ResearchClaw online. No active trial. Say 'start a new trial' to begin.
→
```

The `→` is the input prompt. Type your message and press Enter.

### Step 4 — Run a trial

Here is a complete example session:

```
→  start a new trial
[ResearchClaw] Creating new trial sandbox...
[ResearchClaw] Started trial_20260225_001 for 2026-02-25.
               Copied codebase to <BASE_DIR>/sandbox/20260225/trial_001.
               Goal: (not specified)

               What should I try? I'll read the codebase and propose changes.

→  Try reducing the learning rate in train.py from 1e-3 to 5e-4

[ResearchClaw] I'll read train.py and propose the change.

               CODE_CHANGE: train.py
               ─────────────────────
               [full new content of train.py shown here]
               ─────────────────────
               Approve this change? [Y/N]

→  y

[ResearchClaw] Change applied to sandbox.

               RUN_COMMAND: python train.py
               Run this command? [Y/N]

→  y

[ResearchClaw] Experiment started (PID 14231). Watching for output...

               [a few seconds later]

               ✅ trial_001 finished
               ├ Runtime: 0m 06s
               ├ GPU: n/a (CPU-only)

               Final log:
               ```
               Training started
               Epoch 1: loss=1.0312
               Epoch 2: loss=0.5241
               ...
               Done
               ```

               REPORT.md generated for trial_001.

               What would you like to do?
               (a) approve — merge to main
               (b) reject — discard trial
               (c) continue — keep iterating

→  approve

[ResearchClaw] Merging trial_001 to github_codes/ and committing...
               ✅ trial_001 approved and merged.
               Commit: `a1b2c3d`
               Push to remote now? [Y/N]

→  y

[ResearchClaw] Pushing to remote...
               ✅ Pushed.
               ok

               Start a new trial when ready.
→
```

### Stopping the agent

Press `Ctrl+C`. The agent shuts down cleanly; any in-progress trial is preserved and will be resumed on next start.

---

### Troubleshooting — stdio

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: researchclaw` | venv not active | Run `source <PROJECT_DIR>/researchclaw/.venv/bin/activate` |
| `FileNotFoundError: github_codes/ not found` | Empty base_dir | Clone or copy a project into `<BASE_DIR>/github_codes/` |
| `EnvironmentError: claude CLI not found` | `claude` not in PATH | Install from https://claude.ai/code and run `claude login` |
| `EnvironmentError: ANTHROPIC_API_KEY not set` | Missing .env | Create `.env` with the API key (see Before You Start) |
| Agent says `No active trial` to everything | You haven't started a trial | Type `start a new trial` |
| Agent exits after startup | config.yaml parse error | Run the validator and check the error |

---

## Scenario 2 — Telegram (Recommended for Remote/VM)

**When to use:** Your agent runs on a GPU server (cloud VM, university cluster, etc.) and you want to message it from anywhere — phone, desktop, or another machine.

**How it works:** You create a Telegram bot, give its token to ResearchClaw, and chat with it like any other Telegram contact. The agent uses long-polling — no public IP or port forwarding needed.

---

### Step 1 — Create a Telegram bot

1. Open Telegram and search for **@BotFather** (the official bot-creation bot).
2. Send: `/newbot`
3. When asked for a name, type anything (e.g. `ResearchClaw`).
4. When asked for a username, it must end in `bot` (e.g. `researchclaw_myname_bot`).
5. BotFather replies with your **bot token**. It looks like:
   ```
   7123456789:AAFaBcDeFgHiJkLmNoPqRsTuVwXyZ_abc
   ```
   Keep this secret — it's the password to your bot.

### Step 2 — Find your chat ID

Your bot needs to know who to talk to. Every Telegram user and group has a numeric ID.

**A. Send a message to your bot first**

Open your new bot in Telegram and send it any message (e.g. `hello`). The bot won't reply yet — that's fine. This step is needed so the next script can find your ID.

**B. Run the helper script**

```bash
source <PROJECT_DIR>/researchclaw/.venv/bin/activate

# Option 1: if you already have TELEGRAM_BOT_TOKEN in .env
python -m researchclaw.get_chat_id

# Option 2: pass the token directly
python -m researchclaw.get_chat_id --token 7123456789:AAFaBcD...
```

Expected output:
```
Fetching recent messages from the bot API...

  chat_id: 123456789   (private, @yourusername)

Copy the chat_id for your account into config.yaml:
  messenger:
    telegram_chat_id: "123456789"
```

If you see `No messages found`, go back and send a message to the bot first.

**Group chat IDs** (negative numbers like `-100987654321`) work too — just add the bot to the group and send a message there.

### Step 3 — Create the `.env` file

Create `<PROJECT_DIR>/.env` with the bot token:

```bash
cat > <PROJECT_DIR>/.env << 'EOF'
TELEGRAM_BOT_TOKEN=7123456789:AAFaBcDeFgHiJkLmNoPqRsTuVwXyZ_abc
EOF
chmod 600 <PROJECT_DIR>/.env
```

> `.env` is listed in `.gitignore` — it will never be committed.

### Step 4 — Update config.yaml

Edit `<PROJECT_DIR>/config.yaml`. Change the `messenger:` section to:

```yaml
messenger:
  type: telegram
  telegram_chat_id: "123456789"        # your ID from Step 2
  telegram_bot_token_env: "TELEGRAM_BOT_TOKEN"   # env var name (leave as-is)
  telegram_poll_timeout: 30
  telegram_poll_interval: 1.0
```

Leave all other sections untouched.

### Step 5 — Validate

```bash
python -m researchclaw.init <PROJECT_DIR>/config.yaml
```

If Telegram is configured correctly, you will receive a test message on your phone:
```
ResearchClaw initializer: connection test successful. You can ignore this message.
```

And the terminal shows:
```
  ✓  messenger        ok — sent test message via telegram
```

If the messenger check fails, see the troubleshooting table below.

### Step 6 — Start the agent

```bash
python -m researchclaw.agent <PROJECT_DIR>/config.yaml
```

Your phone receives:
```
ResearchClaw online. No active trial. Say 'start a new trial' to begin.
```

From this point, every message you send to the bot is forwarded to the agent, and every agent reply appears in your chat.

### Step 7 — Run a trial

The interaction is identical to Scenario 1, but happens in Telegram:

```
You:   start a new trial
Bot:   Creating new trial sandbox...
       Started trial_20260225_001 for 2026-02-25.
       What should I try?

You:   Try reducing the learning rate from 1e-3 to 5e-4

Bot:   I'll propose the change...
       [shows diff]
       Approve this change? [Y/N]

You:   y

Bot:   Change applied. RUN_COMMAND: python train.py
       Run this command? [Y/N]

You:   y

Bot:   Experiment started (PID 14231). Watching for output...
       [later]
       ✅ trial_001 finished
       REPORT.md generated.
       (a) approve  (b) reject  (c) continue

You:   approve

Bot:   ✅ trial_001 approved and merged.
       Commit: `a1b2c3d`
       Push to remote now? [Y/N]

You:   y

Bot:   Pushing to remote...
       ✅ Pushed.
       ok

       Start a new trial when ready.
```

### Keeping the agent alive on a remote server

If you're SSH'd into a server, use `tmux` or `screen` so the agent keeps running after you disconnect:

```bash
tmux new -s research
source <PROJECT_DIR>/researchclaw/.venv/bin/activate
python -m researchclaw.agent <PROJECT_DIR>/config.yaml
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t research
```

---

### Troubleshooting — Telegram

| Symptom | Cause | Fix |
|---|---|---|
| `EnvironmentError: TELEGRAM_BOT_TOKEN is not set` | Missing or mis-named .env | Create `.env` with `TELEGRAM_BOT_TOKEN=...` |
| `ValueError: telegram.chat_id is not set` | Missing `telegram_chat_id` in config.yaml | Add your chat ID (see Step 2) |
| Validator fails with `401 Unauthorized` | Wrong bot token | Re-copy the token from @BotFather |
| `No messages found` in `get_chat_id` | Haven't messaged the bot yet | Open the bot in Telegram and send any message, then re-run |
| Messages sent but bot doesn't reply | Wrong `chat_id` | Re-run `get_chat_id` and update config |
| Bot replies to others too | Bot has no privacy filtering | ResearchClaw filters by `chat_id`; only your ID gets through |
| Bot stops responding after SSH disconnect | Agent process died | Use `tmux` (see above) |
| `pyTelegramBotAPI is not installed` | Missing dependency | Run `pip install pyTelegramBotAPI` inside the venv |

---

## Scenario 3 — Slack via OpenClaw (Advanced)

**When to use:** Your team uses Slack and wants the agent to post in a shared channel so everyone can see experiment updates.

**Important caveat:** The Slack integration in ResearchClaw uses **OpenClaw**, which is a separate service. The connector in `researchclaw/messenger.py` is a reference implementation — you may need to adapt it to the actual OpenClaw API version you're running. The setup steps below cover what ResearchClaw needs; OpenClaw's own installation is out of scope here.

---

### Step 1 — Set up OpenClaw

OpenClaw handles the Slack OAuth, token storage, and message routing. Install and configure it separately by following the OpenClaw documentation. Once OpenClaw is running, it provides a Python client (`import openclaw`) that ResearchClaw uses internally.

> If `import openclaw` fails at startup, ResearchClaw automatically falls back to `stdio` and logs:
> ```
> WARNING — OpenClaw not found — falling back to stdio messenger
> ```
> You can verify which messenger is actually running with `python -m researchclaw.init`.

### Step 2 — Update config.yaml

Edit `<PROJECT_DIR>/config.yaml`:

```yaml
messenger:
  type: slack
  slack_channel: "#research-agent"    # the Slack channel to use
```

Create the channel in Slack first and invite the OpenClaw bot to it if required.

No secrets go in `.env` for Slack — OpenClaw manages authentication externally.

### Step 3 — Validate

```bash
python -m researchclaw.init <PROJECT_DIR>/config.yaml
```

If OpenClaw is properly installed and configured, the messenger check passes and a test message appears in `#research-agent`:
```
  ✓  messenger        ok — sent test message via slack
```

If the check shows `ERROR: No module named 'openclaw'`, OpenClaw is not installed in the active Python environment. Install it inside the venv.

### Step 4 — Start the agent

```bash
python -m researchclaw.agent <PROJECT_DIR>/config.yaml
```

A startup message appears in `#research-agent`:
```
ResearchClaw online. No active trial. Say 'start a new trial' to begin.
```

Interact exactly as in the other scenarios — type in the Slack channel and the agent replies there.

---

### Troubleshooting — Slack

| Symptom | Cause | Fix |
|---|---|---|
| `WARNING — OpenClaw not found — falling back to stdio` | `openclaw` package not installed | Install OpenClaw in the venv and re-run |
| Messages not appearing in channel | Bot not added to channel | Invite the OpenClaw bot to `#research-agent` in Slack |
| Init check says `ERROR: ...` | OpenClaw misconfigured | Check OpenClaw's own logs and configuration |
| Agent starts but never receives messages | Wrong `slack_channel` in config | Make sure the channel name matches exactly (including `#`) |

---

## Quick Reference

### Commands the agent understands

| What you type | What happens |
|---|---|
| `start a new trial` | Creates a sandbox, copies `github_codes/`, begins a trial |
| `start a new trial: reduce batch size` | Same, with a goal description |
| `status` | Shows trial name, status, and last git summary |
| `approve` or `merge` | Merges trial changes into `github_codes/`, commits, then prompts to push |
| `reject` | Discards the trial (sandbox kept for reference) |
| `push to github` | Pushes the committed changes to the remote |
| Anything else | Forwarded to the LLM for research conversation |

### Approval responses

When the agent proposes a code change or command, it asks `[Y/N]`. Type:

| Your reply | Meaning |
|---|---|
| `y`, `yes` | Approve |
| `n`, `no` | Reject |

When the experiment finishes and a report is ready, type:

| Your reply | Meaning |
|---|---|
| `approve` or `a` | Merge to `github_codes/` |
| `reject` or `b` | Discard trial |
| `continue` or `c` | Keep iterating in this trial |

### Key file locations

| File | Purpose |
|---|---|
| `<PROJECT_DIR>/config.yaml` | All non-secret configuration |
| `<PROJECT_DIR>/.env` | Secrets (API keys, bot tokens) |
| `<BASE_DIR>/github_codes/` | Your research code (read-only during trials; updated on approval) |
| `<BASE_DIR>/reference/` | Papers, docs, notes you add for context (read-only to agent) |
| `<BASE_DIR>/MEMORY.md` | Agent's persistent memory (written by agent, survives restarts) |
| `<BASE_DIR>/sandbox/` | Trial working copies (auto-created) |
| `<BASE_DIR>/experiment_reports/` | Logs and REPORT.md files (auto-created) |
| `<BASE_DIR>/.trials.jsonl` | Trial ledger (auto-created) |
