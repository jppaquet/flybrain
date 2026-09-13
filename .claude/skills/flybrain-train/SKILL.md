---
name: flybrain-train
description: Launch, watch and evaluate a flybrain driving training run (examples/drive_es.py) - check resources, pick the settings, start it in the background (streamed to the 3D page if the server runs), evaluate the result against the hand-written baseline on common seeds, and report. Use when the user asks to train, relaunch, continue or compare runs of the fly driving the kart.
---

# Train the fly to drive

Background: `TRAINING.md` (the problem, the knobs, the results so far, the leads) and
`API.md`. Runs live in `runs/drive_es/<name>/` (`config.json`, `log.csv`, `best.json`,
`last.json`), ignored by git. Work from the repository root.

1. **Prerequisites.** `.venv/bin/python` and `malecns_graph.npz` must exist (otherwise
   `./setup.sh`). Each worker process needs ~1.5 GB of RAM and one core: check with
   `sysctl -n hw.memsize hw.ncpu` (macOS) or `nproc; free -g` (Linux), and use
   `--workers` = min(cores − 2, free GB / 1.5).

2. **Settings.** Use what the user asked for; otherwise these defaults:
   `--steps 400 --gens 30 --pop 16 --member-seeds 2 --eval-seeds 3`. Name the run after
   what it changes, e.g. `--run-dir runs/drive_es/steps400-ms2`. To continue a run:
   `--init runs/drive_es/<run>/last.json`. Before starting, give the user the estimated
   duration: episodes per generation = pop × member_seeds + eval_seeds; an episode of
   `steps` × 50 ms costs about 0.85 × its simulated time in CPU; divide by the workers.

3. **Watching.** `curl -s -m 3 http://127.0.0.1:8765/api/live/session` — if the server
   answers, add `--watch` and tell the user to open http://127.0.0.1:8765/fly: each
   generation's mean policy drives there, is recorded in `runs/recordings/`, and can be
   replayed or filmed (Recordings section). If it does not answer and the user wants to
   watch, start `./run.sh` in the background first.

4. **Launch** in the background (run_in_background):
   `.venv/bin/python -u examples/drive_es.py <args> 2>&1 | grep -v "^connectome"`.
   Do not poll in a loop: the task notifies when it ends. Read
   `runs/drive_es/<run>/log.csv` if the user asks for progress.

5. **Evaluate** on the horizon it was trained for, on common seeds, next to the baseline
   and to earlier runs:
   `.venv/bin/python examples/drive_eval.py runs/drive_es/<run>/best.json runs/drive_es/<run>/last.json [earlier runs' best.json] --baseline --steps <steps> --seeds 5 --workers <workers>`.

6. **Report**: the learning curve from `log.csv` (generation where the mean policy first
   reached half the baseline, best return), the evaluation table (return ± sd, laps, off
   road), how it compares with earlier runs, and one or two next steps from the "Leads"
   of `TRAINING.md`. Offer to add the result to "Results so far" in `TRAINING.md`; do not
   edit it unasked.
