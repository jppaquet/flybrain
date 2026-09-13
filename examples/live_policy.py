#!/usr/bin/env python3
"""
Drive the kart of the live 3D page with a policy, over HTTP (e.g. one saved by
drive_hillclimb.py).

Start the server (./run.sh), open http://127.0.0.1:8765/fly and choose Drive, then:

    .venv/bin/python examples/live_policy.py drive_policy.json

The script joins the page's session (GET /api/live/session), reads the kart state from
the frames (GET /api/live/frames), and drives the policy's input groups every 50 ms
(POST /api/live/stim, each drive lasting 150 ms so that it stops if the script does).
The page shows the fly driving. Ctrl+C stops.
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flyenv, world


def call(url, body=None):
    req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", help="JSON from drive_hillclimb.py")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    p = json.load(open(a.policy))
    W, high, U = np.array(p["W"]), float(p["max_hz"]), a.url.rstrip("/")

    cur = call(f"{U}/api/live/session")
    if cur["alive"]:
        sid = cur["id"]
        print(f"joined session {sid} (open /fly and choose Drive to watch)")
    else:
        sid = call(f"{U}/api/live/start", dict(params=dict(model="shiu", dt=0.2, w_scale=0.5, quench_ms=1000, world="kart")))["id"]
        print(f"started session {sid} (the page shows its own session: start Drive there to watch)")
    call(f"{U}/api/live/world", dict(id=sid, world="kart", reset=True))
    keys, nxt = world.Kart.FRAME_KEYS, call(f"{U}/api/live/frames?id={sid}&since=0")["next"]
    try:
        while True:
            d = call(f"{U}/api/live/frames?id={sid}&since={nxt}")
            nxt = d["next"]
            kart = next((f[7] for f in reversed(d["frames"]) if len(f) > 7 and f[7]), None)
            if kart:
                s = dict(zip(keys, kart))
                rates = high / (1.0 + np.exp(-(W @ np.append(flyenv.drive_observation(s), 1.0))))
                for spec, hz in zip(p["inputs"], rates):
                    call(f"{U}/api/live/stim", dict(id=sid, name=spec["name"], query=spec["query"], field=spec["field"],
                                                    side=spec["side"], hz=float(hz), ms=150))
                print(f"\rspeed {s['speed']:5.1f} · offset {s['offset']:5.1f} · distance {s['distance']:7.1f}", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
