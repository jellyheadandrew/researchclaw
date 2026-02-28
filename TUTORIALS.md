# Tutorials

## End-to-End Trial (Autonomous)
1. Run `./gateway`
2. Say "plan a new experiment" (or `/plan`)
3. Select a project or say "scratch"
4. Describe your experiment idea in natural language
5. Iterate with the agent on the plan via chat
6. Say "approve" when the plan looks good
7. Watch the agent automatically:
   - Generate experiment code from the plan
   - Execute the experiment (run.sh)
   - Auto-fix any failures (Ralph loop)
   - Generate evaluation code
   - Execute the evaluation (eval.sh)
   - Generate a comprehensive report
8. Review results via "show me past results" (or `/view_summary`)

## Autopilot Mode
1. Say `/autopilot-start` and confirm
2. The agent runs experiment cycles autonomously: plan → implement → execute → evaluate → report → repeat
3. Stop with `/autopilot-stop`
4. Safeguards: auto-stops after max consecutive trials or failures (configurable in SETTINGS)

## Update and Push
1. Say "push results to a project" (or `/update_and_push`)
2. Choose an existing project or add a new one (clone or init)
3. Select the trial to assimilate
4. The agent intelligently merges trial code into the project and pushes to GitHub

## Research Mode
1. Say "explore research ideas" (or `/research`)
2. Chat naturally about research directions
3. Search papers: `/research search <query>`
4. Brainstorm: `/research brainstorm <topic>`
5. Download references: `/research download <url>`
6. Configure periodic idea nudges: `/research cadence hourly`

## Settings
1. Say "settings" (or `/settings`)
2. View current values: `/settings show`
3. Change a value: `/settings set experiment_max_iterations 5`
4. Explain fields: `/settings explain`

## Telegram Setup
1. Create a bot via @BotFather on Telegram
2. Get your chat ID (send a message to the bot, then check `https://api.telegram.org/bot<token>/getUpdates`)
3. Update config.yaml:
   ```yaml
   messenger:
     type: telegram
     bot_token: "YOUR_BOT_TOKEN"
     chat_id: "YOUR_CHAT_ID"
   ```
4. Install telegram dependency: `pip install 'researchclaw[telegram]'`
5. Run `./gateway` — messages now flow through Telegram
