# Tutorials

## Minimal End-to-End Trial
1. Run `./gateway`
2. Enter `/plan`
3. Select project: `/plan project scratch`
4. Provide planning notes in plain text
5. Approve: `/plan approve`
6. Write experiment script:
   - `/write run.sh`
   - add commands
   - `/endwrite`
7. Run experiment: `/exp run`
8. Write eval script and eval code (if needed)
9. Run eval: `/eval run`
10. Wait for report and review via `/view_summary`

## Update and Push
1. Enter `/update_and_push`
2. Choose existing project or add clone/init
3. Select trial
4. Assimilation and push run automatically
