#!/usr/bin/env python3
"""
Learn to drive from scratch with an evolution strategy, evaluated in parallel.

The policy is linear with two commands: tanh(W @ [observation, 1]) gives a speed command
(+ accelerator, - brake) and a steering command (+ right, - left), which
flyenv.paired_action turns into the rates of the four Drive inputs. Opposite inputs are
never driven together - they would cancel out, and both front-leg DNs at once run the
network away - and near zero a small change of the parameters changes the rates by tens
of Hz, so the search can find the steering. W starts at zero: the kart does not move.

Each generation samples antithetic perturbations of W, runs one episode per perturbation
in worker processes (each loads the connectome once, about 1.5 GB of RAM), ranks the
returns and moves W along the ranked perturbations (Adam). All the members of a
generation share one seed - the same Poisson noise - so their returns differ only by
their policies. No dependency beyond numpy.

    .venv/bin/python examples/drive_es.py                          # 4 workers, 40 generations
    .venv/bin/python examples/drive_es.py --workers 8 --steps 200 --watch
    .venv/bin/python examples/drive_es.py --init runs/drive_es/<run>/last.json   # continue a run

Every run gets a directory (default runs/drive_es/<date-time>, --run-dir to name it):
config.json (arguments, git commit, inputs, track), log.csv (one row per generation),
best.json (best mean policy so far) and last.json (policy after the last generation).
Compare runs with examples/drive_eval.py; the score to beat is examples/drive_baseline.py
(return ~97 in 200 steps, ~230 in 400). --watch streams the mean policy of every
generation to the server: /fly shows it drive.
"""
import argparse, csv, datetime, json, multiprocessing as mp, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import flyenv, live, world                            # no connectome here: it loads in the workers

MAX_HZ = 150.0
SPECS = [dict(name=d["label"], query=d["query"], field=d["field"], side=d.get("side")) for d in live.KEYSETS["car"]]
_env, _watch_env, _cfg = None, None, None


def policy(W, obs):
    return flyenv.paired_action(np.tanh(W @ np.append(obs, 1.0)), MAX_HZ)


def init_worker(steps, watch):
    global _env, _cfg
    _cfg = (steps, watch)
    _env = flyenv.DriveEnv(max_steps=steps)


def episode(task):
    """(W, seed, label) -> (return, distance). A label streams this episode to the server."""
    global _watch_env
    W, seed, label = task
    env = _env
    if label:
        if _watch_env is None:                        # shares the worker's connectome: cheap
            _watch_env = flyenv.DriveEnv(max_steps=_cfg[0],
                                         viewer=flyenv.Viewer(_cfg[1], label=label, world="kart", quiet=True))
        env = _watch_env
        env.brain.viewer.label = label
        env.brain.viewer.reconnect()                   # a new session per generation: the page follows it
    obs, info = env.reset(seed=seed)
    ret = 0.0
    for _ in range(env.max_steps):
        obs, r, terminated, truncated, info = env.step(policy(W, obs))
        ret += r
        if terminated or truncated: break
    if label: env.brain.viewer.flush()               # the page gets the end of the episode too
    return ret, info["distance"]


def centered_ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def git_commit():
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return rev + (" + uncommitted changes" if dirty else "")
    except OSError:
        return None


def save_policy(path, W, **meta):
    with open(path, "w") as f:
        json.dump(dict(kind="paired", W=W.tolist(), max_hz=MAX_HZ, observation=flyenv.DriveEnv.observation_names,
                       commands=["speed (+ accelerator, - brake)", "steering (+ right, - left)"],
                       inputs=SPECS, **meta), f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gens", type=int, default=40, help="generations")
    ap.add_argument("--pop", type=int, default=16, help="episodes per generation (even: antithetic pairs)")
    ap.add_argument("--sigma", type=float, default=0.5, help="size of the parameter perturbations")
    ap.add_argument("--lr", type=float, default=0.1, help="Adam step size")
    ap.add_argument("--steps", type=int, default=400, help="steps of 50 ms per episode")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=0, help="seed of the perturbations and of the episode seeds")
    ap.add_argument("--member-seeds", type=int, default=1,
                    help="episodes per perturbation, each with its own seed (more: a less noisy ranking, more cost)")
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="episodes of the mean policy per generation; best.json is chosen on their mean")
    ap.add_argument("--init", default=None, help="start from a saved paired policy (a run's last.json or best.json)")
    ap.add_argument("--run-dir", default=None, help="output directory (default runs/drive_es/<date-time>)")
    ap.add_argument("--watch", nargs="?", const="http://127.0.0.1:8765", default=None, metavar="URL",
                    help="stream each generation's mean policy to the server so that /fly shows it")
    a = ap.parse_args()

    run = a.run_dir or os.path.join("runs", "drive_es", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    W = np.zeros((2, len(flyenv.DriveEnv.observation_names) + 1))   # from scratch: no speed, no steering
    if a.init:
        with open(a.init) as f: p = json.load(f)
        if p.get("kind") != "paired": sys.exit(f"{a.init}: not a paired policy (drive_es.py)")
        W = np.array(p["W"], dtype=float)
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump(dict(vars(a), run_dir=run, git=git_commit(), started=datetime.datetime.now().isoformat(timespec="seconds"),
                       max_hz=MAX_HZ, live_params=flyenv.LIVE_PARAMS, track=world.Track().to_dict(),
                       step_ms=50, inputs=SPECS, observation=flyenv.DriveEnv.observation_names), f, indent=1)
    log = open(os.path.join(run, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["generation", "pop_mean", "pop_max", "pop_std", "mean_policy_return", "mean_policy_distance", "best", "seconds"])

    m, v, b1, b2 = np.zeros_like(W), np.zeros_like(W), 0.9, 0.999
    rng, pairs, best = np.random.default_rng(a.seed), max(1, a.pop // 2), -np.inf
    print(f"run {run} · {a.workers} workers · {2 * pairs} episodes of {a.steps * 0.05:.0f} s per generation · inputs:",
          *[s["name"] for s in SPECS], sep="\n  ")
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(a.steps, a.watch)) as pool:
        for g in range(1, a.gens + 1):
            t0 = time.time()
            seeds = [int(s) for s in rng.integers(1 << 30, size=max(a.member_seeds, a.eval_seeds))]
            eps = rng.standard_normal((pairs,) + W.shape); eps = np.concatenate([eps, -eps])
            label = f"ES generation {g} · mean policy" if a.watch else None
            tasks = [(W + a.sigma * e, s, None) for e in eps for s in seeds[:a.member_seeds]]
            tasks += [(W, s, label if j == 0 else None) for j, s in enumerate(seeds[:a.eval_seeds])]
            res = pool.map(episode, tasks, chunksize=1)
            n = len(eps) * a.member_seeds
            R = np.array([r for r, _ in res[:n]]).reshape(len(eps), a.member_seeds).mean(1)
            mean_ret, mean_dist = (float(np.mean([x[k] for x in res[n:]])) for k in (0, 1))
            if mean_ret > best:                        # the policy that was just evaluated, on eval_seeds episodes
                best = mean_ret
                save_policy(os.path.join(run, "best.json"), W, generation=g, ret=mean_ret, seeds=seeds[:a.eval_seeds])
            grad = np.tensordot(centered_ranks(R), eps, axes=1) / (len(eps) * a.sigma)
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
            W = W + a.lr * (m / (1 - b1 ** g)) / (np.sqrt(v / (1 - b2 ** g)) + 1e-8)
            save_policy(os.path.join(run, "last.json"), W, generation=g)
            dt = time.time() - t0
            wr.writerow([g, round(R.mean(), 3), round(R.max(), 3), round(R.std(), 3), round(mean_ret, 3),
                         round(mean_dist, 3), round(best, 3), round(dt, 1)]); log.flush()
            print(f"gen {g:3d}: population mean {R.mean():7.1f} max {R.max():7.1f} · mean policy {mean_ret:7.1f} "
                  f"({mean_dist:6.1f} along the road) · best {best:7.1f} · {dt:.0f} s", flush=True)
    log.close()
    print(f"-> {run}: best.json (return {best:.1f}), last.json, log.csv, config.json\n"
          f"   compare: examples/drive_eval.py {run}/best.json --baseline\n"
          f"   on the page: examples/live_policy.py {run}/best.json")


if __name__ == "__main__":
    main()
