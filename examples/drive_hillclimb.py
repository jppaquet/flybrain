#!/usr/bin/env python3
"""
Learn to drive the kart through the fly's neurons, with a tiny hill-climbing search.

The policy is linear: the DriveEnv observation (speed, wheel, offset, heading error,
pedal and wheel-push levels) -> the Poisson rate of each input group (sigmoid x max Hz).
The default inputs are the Drive keys of the 3D page: the descending neurons that move
each front leg (steering wheel) and each hind leg (pedals). Only the leg motor neurons
move the kart. No dependency beyond numpy; an episode runs about as fast as real time.

    .venv/bin/python examples/drive_hillclimb.py --iters 20 --steps 200
    .venv/bin/python examples/live_policy.py drive_policy.json     # then watch it on /fly
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flyenv


def policy(W, obs, high):
    return high / (1.0 + np.exp(-(W @ np.append(obs, 1.0))))


def episode(env, W, steps, seed=0):
    obs, info = env.reset(seed=seed)
    total, n = 0.0, 0
    for n in range(1, steps + 1):
        obs, r, terminated, truncated, info = env.step(policy(W, obs, env.action_high))
        total += r
        if terminated or truncated: break
    return total, info["distance"], n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20, help="candidate policies to try")
    ap.add_argument("--steps", type=int, default=200, help="steps of 50 ms per episode")
    ap.add_argument("--sigma", type=float, default=1.0, help="size of the random perturbations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="drive_policy.json")
    a = ap.parse_args()

    env = flyenv.DriveEnv(max_steps=a.steps)
    print("inputs:", *env.action_names, sep="\n  ")
    rng = np.random.default_rng(a.seed)
    W = np.zeros((len(env.action_names), len(env.observation_names) + 1))
    W[:, -1] = -2.0                                   # start with every input almost off
    best, dist, n = episode(env, W, a.steps)
    print(f"start    : return {best:7.2f} · distance {dist:6.1f} · {n} steps")
    for it in range(1, a.iters + 1):
        t0 = time.time()
        cand = W + a.sigma * rng.standard_normal(W.shape)
        ret, dist, n = episode(env, cand, a.steps)
        if ret > best: W, best = cand, ret
        print(f"iter {it:3d} : return {ret:7.2f} (best {best:7.2f}) · distance {dist:6.1f} · {n} steps · {time.time() - t0:.0f} s")
    with open(a.out, "w") as f:
        json.dump(dict(W=W.tolist(), max_hz=float(env.action_high[0]), observation=env.observation_names,
                       inputs=[dict(g.spec, name=g.name) for g in env.inputs]), f, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
