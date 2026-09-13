#!/usr/bin/env python3
"""
A hand-written reference driver: a feedback controller that reads the kart's state and
sets the Poisson rates of the four Drive inputs - the same interface as a learned policy.

It plays the part of the agent, not of the fly. What it drives are descending neurons
(DNg12_e left and right for the steering wheel, DNg16 right for the accelerator); the
fly's leg motor neurons, downstream in the nerve cord, turn the wheel and press the
pedal. It shows that the circle can be driven, and gives the score to beat: about 1.4 to
1.5 laps in 20 s, within ~1 of the centre line (the road's half width is 4).

    .venv/bin/python examples/drive_baseline.py                 # 3 seeds, 20 s each
    .venv/bin/python examples/drive_baseline.py --watch         # and watch it on /fly

Controller: the wheel angle the circle needs, atan(wheelbase / radius), corrected by the
offset from the centre line and by the heading error; the gap between that target and the
actual wheel drives the DN of the left or the right front leg, proportionally.
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flyenv, world


def controller(s, track, gas=150.0, K=800.0, a=0.3, b=1.5):
    """Kart state -> rates (Hz) for [accelerator, brake, wheel left, wheel right]."""
    w0 = -math.atan(world.Kart.WHEELBASE / track.R) / world.Kart.MAX_STEER
    target = w0 - a * s["offset"] / track.W + b * s["heading_error"]
    e = target - s["wheel"]                           # < 0: the wheel must turn further left
    return np.array([gas, 0.0, min(250.0, max(0.0, -e * K)), min(250.0, max(0.0, e * K))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=400, help="steps of 50 ms per episode")
    ap.add_argument("--seeds", type=int, default=3, help="episodes, one per seed (Poisson noise)")
    ap.add_argument("--watch", nargs="?", const="http://127.0.0.1:8765", default=None, metavar="URL",
                    help="stream the episodes to the server so that /fly shows them")
    a = ap.parse_args()

    env = flyenv.DriveEnv(max_steps=a.steps, viewer=a.watch)
    track, rets = env.driver.kart.track, []
    lap = 2 * math.pi * track.R
    for seed in range(a.seeds):
        env.reset(seed=seed)
        ret, offs, n, info, term = 0.0, [], 0, {}, False
        for n in range(1, a.steps + 1):
            s = env.driver.kart.state()
            _, r, term, trunc, info = env.step(controller(s, track))
            ret += r; offs.append(abs(s["offset"]))
            if term or trunc: break
        rets.append(ret)
        print(f"seed {seed}: {info['distance'] / lap:.2f} lap in {n * env.step_ms / 1000:.1f} s · "
              f"|offset| max {max(offs):.2f} · {'off the road' if term else 'on the road'} · return {ret:.1f}")
    print(f"mean return {np.mean(rets):.1f} over {a.seeds} seeds: the score a learned policy should beat")


if __name__ == "__main__":
    main()
