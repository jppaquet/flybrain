#!/usr/bin/env python3
"""
Evaluate saved driving policies - and the hand-written baseline - on the same seeds, in
parallel, to compare runs fairly: the Poisson noise of the inputs changes from seed to
seed, so a single episode says little.

    .venv/bin/python examples/drive_eval.py runs/drive_es/*/best.json --baseline
    .venv/bin/python examples/drive_eval.py drive_policy.json --seeds 10 --steps 400 --workers 8

Policies: JSON saved by drive_es.py ("paired") or drive_hillclimb.py. For each one: mean
and standard deviation of the return, laps, how often the kart left the road, and how long
it lasted. Evaluation seeds start at --seed-base (1000), apart from training seeds.
"""
import argparse, math, multiprocessing as mp, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import flyenv

_env = None


def init_worker(steps):
    global _env
    _env = flyenv.DriveEnv(max_steps=steps)


def episode(task):
    import drive_baseline
    name, seed = task
    env, pol = _env, None if name == "baseline" else flyenv.load_policy(name)
    obs, info = env.reset(seed=seed)
    ret, n, off = 0.0, 0, False
    for n in range(1, env.max_steps + 1):
        k = env.driver.kart
        act = drive_baseline.controller(k.state(), k.track) if pol is None else pol(obs)
        obs, r, off, truncated, info = env.step(act)
        ret += r
        if off or truncated: break
    return name, ret, info["distance"], off, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="*", help="policy JSON files")
    ap.add_argument("--baseline", action="store_true", help="also evaluate examples/drive_baseline.py")
    ap.add_argument("--seeds", type=int, default=5, help="episodes per policy")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=400, help="steps of 50 ms per episode")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    names = list(a.policies) + (["baseline"] if a.baseline else [])
    if not names: sys.exit("nothing to evaluate: give policy files and/or --baseline")
    tasks = [(nm, a.seed_base + s) for nm in names for s in range(a.seeds)]
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(a.steps,)) as pool:
        res = pool.map(episode, tasks, chunksize=1)
    lap = 2 * math.pi * flyenv.world.Track().R
    rows = []
    for nm in names:
        r = [x for x in res if x[0] == nm]
        R = np.array([x[1] for x in r])
        rows.append((R.mean(), nm, R.std(), np.mean([x[2] for x in r]) / lap, sum(x[3] for x in r), np.mean([x[4] for x in r])))
    print(f"{a.seeds} seeds from {a.seed_base}, {a.steps} steps ({a.steps * 0.05:.0f} s) per episode\n")
    print(f"{'return':>16}  {'laps':>5}  {'off road':>8}  {'lasted':>7}  policy")
    for mean, nm, std, laps, offs, n in sorted(rows, reverse=True):
        print(f"{mean:8.1f} ± {std:5.1f}  {laps:5.2f}  {offs:>4d}/{a.seeds:<3d}  {n * 0.05:5.1f} s  {nm}")


if __name__ == "__main__":
    main()
