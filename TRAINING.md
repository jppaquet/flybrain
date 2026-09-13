# Training the fly to drive

How to reproduce the training runs, what can be changed, what has been tried, and where
to look for improvements. The programming interface is described in [API.md](API.md).

## The problem, as the agent sees it

The fly stands in a kart on a circular road (radius 24, half width 4). Only its motor
neurons move the kart (`world.CONTROLS`): the front legs' coxa promotors push the wheel
rim, the right hind leg's extensors press the accelerator, the left hind leg's flexors
pull the brake. A spring recentres the wheel, so following the curve means pushing all
the time.

| | |
|---|---|
| **Actions** | Poisson rates for four groups of neurons: DNg16 right (accelerator), DNpe008 left (brake), DNg12_e right (wheel left), DNg12_e left (wheel right). `drive_es.py` produces two commands — speed and steering, in −1..1 — that `flyenv.paired_action` turns into those rates (≤ 150 Hz, never two opposite inputs at once) |
| **Observation** | speed, wheel angle, offset from the centre line, heading error, and the throttle, brake and wheel push that the legs actually produce (7 values, ≈ −1..1) |
| **Reward** | distance gained along the road per step, minus 0.02 × offset / half width; −5 and end of the episode when the kart leaves the road |
| **Step, episode** | 50 ms per step; 200 steps (10 s) or 400 (20 s) per episode |
| **Cost** | one step takes about 42 ms of CPU: one worker process runs about as fast as real time and needs about 1.5 GB of RAM (it loads the connectome) |

The agent is a controller outside the fly: it reads the kart's state and chooses which
neurons to drive. The fly does not see the road; its nervous system turns the commands
into leg movements, and the connectome decides how well that works (crosstalk between
legs, runaway when both front-leg DNs are driven together, noise).

## Reproduce

```sh
# the score to beat: a hand-written feedback controller over the same four inputs
.venv/bin/python examples/drive_baseline.py --steps 200

# train from scratch: 30 generations of 16 episodes of 10 s on 8 workers (~10 min)
.venv/bin/python examples/drive_es.py --gens 30 --pop 16 --workers 8 --steps 200 --run-dir runs/drive_es/my-run

# watch it train: start ./run.sh, open http://127.0.0.1:8765/fly, and add --watch
.venv/bin/python examples/drive_es.py --steps 200 --workers 8 --watch

# compare runs, and the baseline, on the same evaluation seeds
.venv/bin/python examples/drive_eval.py runs/drive_es/*/best.json --baseline --steps 200 --seeds 5

# continue a run from its last policy
.venv/bin/python examples/drive_es.py --init runs/drive_es/my-run/last.json --run-dir runs/drive_es/my-run-2

# drive the kart of the live page with a trained policy
.venv/bin/python examples/live_policy.py runs/drive_es/my-run/best.json
```

Each run writes `runs/drive_es/<name>/` (ignored by git):

| File | Content |
|---|---|
| `config.json` | every argument, the git commit (with "+ uncommitted changes" if the tree was modified), start time, simulation parameters, inputs, observation, track |
| `log.csv` | one row per generation: population mean, max and standard deviation of the return, return and distance of the mean policy, best so far, seconds |
| `best.json` | the best mean policy so far (load it with `flyenv.load_policy`) |
| `last.json` | the policy after the last generation, to continue with `--init` |

With `--watch`, the mean policy of every generation is also streamed to the server, which
records it in `runs/recordings/` (one file per generation, a few hundred kB). On `/fly`,
*Recordings → Replay* plays any of them again, and *Record a video of every watched run
or replay* saves one WebM video per generation (the page must stay open and visible).

A run is reproducible: `--seed` fixes the perturbations and the episode seeds, and an
episode is deterministic given its seed (the Poisson inputs come from a seeded numpy
generator). The number of workers does not change the results. Different machines or
numpy versions may differ in the last digits.

## What can be changed

| What | Where |
|---|---|
| generations, population, perturbation size, step size, episode length, workers, seed | `drive_es.py` arguments |
| episodes per perturbation, episodes to pick the best policy | `drive_es.py --member-seeds` (default 1), `--eval-seeds` (default 3) |
| which neurons the agent drives | `DriveEnv(inputs=[dict(name, query, field, side), ...])`, default `live.KEYSETS["car"]` |
| maximum rate, step length, episode length | `DriveEnv(max_hz, step_ms, max_steps)`, `drive_es.MAX_HZ` |
| the reward | `DriveEnv.step` |
| the road | `world.Track(radius, half_width)`, `DriveEnv(track=...)` |
| the kart's physics | `world.Kart` (acceleration, braking, drag, top speed, steering, wheel spring) |
| which motor neurons move the kart, and how strongly | `world.CONTROLS` |
| the fly's network | `DriveEnv(params=dict(w_scale=..., dt=..., std_u=...))`, defaults `flyenv.LIVE_PARAMS` |

## Results so far

Evaluated with `drive_eval.py` on the same seeds (from 1000):

| Policy | 10 s episodes | Off road | 20 s episodes | Notes |
|---|---|---|---|---|
| `drive_baseline.py` (hand-written) | 98.7 ± 1.7 (0.66 lap) | 0/5 | 231.7 ± 1.9 (1.55 laps) | never leaves the road |
| ES v1: four sigmoid outputs | 9.5 ± 0.5 (0.07 lap) | 0/3 | — | crawls and never steers: from rates of ~12 Hz, the perturbations changed them by ±5 Hz, too little to move the wheel (DNg12_e needs ~100 Hz) |
| ES v2: two paired commands (`paired-v1`) | 84.4 ± 13.3 (0.58 lap) | 2/5 | 84.2 ± 19.9 (0.60 lap) | leaves the road on 2 of 5 new seeds over 10 s, and on every 20 s episode after ~9.4 s |

Both ES runs: `drive_es.py --gens 30 --pop 16 --workers 8 --steps 200` (10 s episodes,
about 10 minutes), with a single evaluation episode per generation for the mean policy
(the default is now `--eval-seeds 3`, so new runs will give different numbers). What v2
shows:

- **The paired commands make the problem learnable**: the mean policy went from not
  moving at all to ~90 in 8 generations, and scored 101.6 on its training seed at
  generation 29 — the baseline's level.
- **Winner's curse**: `best.json` was picked on one episode, where it was lucky (101.6);
  on new seeds it scores 84. Picking it on the mean of several episodes (`--eval-seeds`)
  fixes the selection; `--member-seeds` does the same for the ranking of perturbations.
- **Horizon**: trained on 10 s episodes, the policy never had to hold the curve longer;
  on 20 s episodes it leaves the road right after 10 s. Train on the horizon you will
  evaluate on.

## Leads for improvement

Roughly ordered by expected gain for the effort:

1. **Train on the horizon you want** (`--steps 400` for 20 s): the v2 policy leaves the
   road right after the 10 s it was trained on.
2. **Less noisy selection.** `--eval-seeds` (default 3) picks `best.json` on the mean of
   several episodes; `--member-seeds 2` or `3` scores each perturbation on several seeds
   (2–3× the cost, a cleaner gradient); a larger `--pop` also helps.
3. **Curriculum.** Start on a wider road (`half_width` 8) or with a lower top speed,
   then tighten: early episodes last longer and give a signal about steering.
3. **Reward shaping.** Add a penalty on the heading error or a small bonus per step on
   the road; or reward lap time once laps are completed.
4. **Richer policy.** The wheel has dynamics (a spring, integration): add the previous
   commands or the wheel's rate of change to the observation, or use a small MLP
   (7 → 16 → 2); ES handles either without change.
5. **Other optimisers.** With 16 parameters, CMA-ES is a natural fit; PPO or SAC need a
   Gymnasium wrapper and new, pinned dependencies. ES stays the easiest to parallelise.
6. **Faster simulation.** The simulator is the bottleneck (165,384 neurons, 10.4 M
   synapses, 0.2 ms steps). Options: more workers; simulating only the neurons within a
   few synapses of the inputs and readouts (much faster, but a different model: compare
   before trusting it); a coarser time step (changes the dynamics).
7. **A runaway guard in flyenv.** The live page resets the network when keys run it
   away; `flyenv` does not, so an agent can push the network into its self-sustained
   state (for instance with both front-leg DNs). Detecting it and ending the episode
   with a penalty would teach the agent to avoid it.
8. **Other inputs.** Drive neurons further upstream (premotor or sensory neurons)
   instead of the leg descending neurons, or give the fly vision by encoding the road
   into visual neurons (LC, LPLC…) — a research question: whether the connectome carries
   a useful signal to the legs is not known in advance.
9. **Other networks.** `w_scale`, synaptic depression and the sign rule change the fly
   itself; a policy can be evaluated under several settings to see how robust it is.
