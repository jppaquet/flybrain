---
name: flybrain-test
description: Test flybrain after a change - syntax of every JS and Python file, backend smoke tests (model, flyenv, HTTP API, recordings, replay), and the end-to-end test of the 3D page in headless Chrome (every mode, following a streamed run, replay, video) with screenshots to look at. Use after editing code and before committing.
---

# Test flybrain

Work from the repository root. Every step prints PASS / FAIL lines; the scripts exit with
the number of failures.

1. **Syntax** (seconds):
   ```sh
   for f in static/*.js static/fly/*.js; do node --input-type=module --check < "$f" || echo "FAIL $f"; done
   .venv/bin/python -m py_compile *.py examples/*.py tests/*.py && echo "python ok"
   ```

2. **Backend smoke tests** (~1 min; needs `./setup.sh` done):
   `.venv/bin/python tests/smoke.py` — the "always through synapses" rule, DriveEnv, the
   baseline controller, the HTTP API of a test server on a spare port, recording a
   streamed run, reading it back and replaying it.

3. **3D page end to end** (~2 min; needs Node ≥ 22 and Chrome, `CHROME=/path` to choose):
   `node tests/e2e.mjs`. It starts its own server on a spare port, so a running
   `./run.sh` is not disturbed, and removes the recordings it creates. Then look at the
   screenshots with Read — `tests/out/keyboard.png`, `switches.png`, `drive.png`,
   `watch.png` — and check that the fly, the console, the kart and the 3D brain are drawn.

4. **Report** what passed and what failed, with the failing lines. Software rendering in
   headless Chrome is slow: a step-3 failure that depends on timing deserves one re-run
   before digging in (say so if it passes the second time).

For a backend-only change, steps 1 and 2 are enough; run all three for anything under
`static/`, `live.py` or `dashboard.py`.
