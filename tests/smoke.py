#!/usr/bin/env python3
"""
Backend smoke tests, no browser: the model, flyenv, and the HTTP API of a test server
started on a spare port (a running ./run.sh is not disturbed). About a minute; run after
./setup.sh.

    .venv/bin/python tests/smoke.py

Each check prints PASS or FAIL; the exit code is the number of failures.
"""
import json, os, socket, subprocess, sys, tempfile, time, urllib.request
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import flyenv, live

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{' · ' + str(detail) if detail != '' else ''}", flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, body=None):
    req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def model_checks():
    brain = flyenv.FlyBrain()
    r = brain.step({brain.group("^LAL018$", side="L"): 150}, ms=300)
    check("LAL018 left drives DNa01/02 left through a synapse", r["turnL"] > 20, f"turnL {r['turnL']:.0f} Hz")
    brain.reset()
    r = brain.step({brain.group("^DNp09$"): 150}, ms=100)
    check("driven neurons are never read (DNp09 driven: fwd 0)", r["fwd"] == 0, f"fwd {r['fwd']:.0f} Hz")

    env = flyenv.DriveEnv(max_steps=60)
    env.reset(seed=0)
    for _ in range(40): _, _, _, _, info = env.step(flyenv.paired_action([1, 0]))
    check("DriveEnv: the accelerator neurons move the kart", info["kart"]["speed"] > 3, f"speed {info['kart']['speed']:.1f}")

    import drive_baseline
    env.reset(seed=1)
    off = False
    for _ in range(60):
        k = env.driver.kart
        _, _, off, _, info = env.step(drive_baseline.controller(k.state(), k.track))
        if off: break
    check("the baseline controller stays on the road for 3 s", not off and info["distance"] > 10, f"distance {info['distance']:.1f}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(kind="paired", W=[[0] * 7 + [2.0], [0] * 8], max_hz=150), f)
    a = flyenv.load_policy(f.name)(np.zeros(7))
    os.unlink(f.name)
    check("load_policy: a paired policy gives 4 rates", a.shape == (4,) and a[0] > 100 and a[1] == 0, np.round(a, 1).tolist())


def api_checks():
    port = free_port()
    U = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "dashboard.py"), "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    made = []
    try:
        for _ in range(120):
            try: call(U + "/api/live/session"); break
            except OSError: time.sleep(0.5)
        info = call(U + "/api/live/start", dict(params=dict(model="shiu", dt=0.2, w_scale=0.5, quench_ms=1000, world="kart")))
        sid, nkeys = info["id"], len(info["channels"])
        check("a live session starts with the kart", info.get("world") == "kart" and nkeys > 50, f"{nkeys} channels")
        end = time.time() + 1.5
        while time.time() < end:
            call(U + "/api/live/drive", dict(id=sid, keys=["up"], keyset="car")); time.sleep(0.1)
        kart = [f[7] for f in call(f"{U}/api/live/frames?id={sid}&since=0")["frames"] if f[7]]
        check("the Drive keys move the kart over HTTP", kart and kart[-1][3] > 1, f"speed {kart[-1][3]:.1f}" if kart else "no kart")
        r = call(U + "/api/live/stim", dict(id=sid, name="test", query="^LAL018$", side="L", hz=150, ms=200))
        check("any group can be driven over HTTP", r["n"] == 1, r)

        v = call(U + "/api/view/start", dict(label="smoke test", world="kart"))
        made.append(v.get("recording"))
        frames = [[10.0 * (i + 1), [0.0] * nkeys, 0, 0, [], "smoke" if i == 0 else None, None,
                   [0, 0.1 * i, 0, 1, 0, 0.5, 0, 0, 0.1 * i, 0, 0]] for i in range(50)]
        call(U + "/api/view/push", dict(id=v["id"], frames=frames, groups={"g": [1, 2]}))
        s = call(U + "/api/live/session")
        check("a streamed session becomes the current one", s["kind"] == "remote" and s["label"] == "smoke test", f"{s['kind']} {s['label']}")
        names = [x["name"] for x in call(U + "/api/recordings")]
        check("the streamed run is recorded", v.get("recording") in names, v.get("recording"))
        head, groups, fr = live.read_recording(v["recording"])
        check("the recording reads back", len(fr) == 50 and groups.get("g") == [1, 2] and head["label"] == "smoke test", f"{len(fr)} frames")
        rp = call(U + "/api/recordings/replay", dict(name=v["recording"]))
        time.sleep(1.0)
        n = len(call(f"{U}/api/live/frames?id={rp['id']}&since=0")["frames"])
        check("a recording replays as a new session", rp["remote"].startswith("Replay") and n > 20, f"{n} frames")
    finally:
        srv.terminate(); srv.wait(10)
        for name in filter(None, made):
            try: os.unlink(os.path.join(live.RECORD_DIR, name))
            except OSError: pass


if __name__ == "__main__":
    t0 = time.time()
    model_checks()
    api_checks()
    fails = results.count(False)
    print(f"\n{len(results) - fails}/{len(results)} passed in {time.time() - t0:.0f} s")
    sys.exit(fails)
