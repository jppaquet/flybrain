#!/usr/bin/env python3
"""
dashboard - local server to simulate the MaleCNS connectome from the browser.

    .venv/bin/python dashboard.py [--port 8765]

then open http://127.0.0.1:8765 . No dependency beyond numpy/pyarrow.
"""
import argparse, json, os, struct, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import Counter
import numpy as np
import engine, live

STATIC = os.path.join(engine.HERE, "static")
NT_NAMES = ["acetylcholine", "gaba", "glutamate", "histamine", "dopamine",
            "serotonin", "octopamine", "tyramine", "unclear", ""]
SIDES = ["L", "R", "M", ""]

PRESETS = {
    "stim": [
        ("LC4 — looming detection", "^LC4$", "type"),
        ("LPLC2 — looming", "^LPLC2$", "type"),
        ("ORN DA1 — cVA pheromone", "^ORN_DA1$", "type"),
        ("ORN VA1v — pheromone", "^ORN_VA1v$", "type"),
        ("Johnston's organ B — sound", "^JO-B", "type"),
        ("Johnston's organ C/E — wind", "^JO-(C|E)", "type"),
        ("Labellar GRNs (LB)", "^LB[0-9]", "type"),
        ("Giant fiber DNp01", "^DNp01$", "type"),
        ("MDN — backward walking", "^MDN$", "type"),
        ("pIP10 — courtship song", "^pIP10$", "type"),
        ("DNa02 — turning", "^DNa02$", "type"),
        ("s-LNv — clock", "^s-LNv$", "type"),
    ],
    "readout": [
        ("Giant fiber DNp01", "^DNp01$", "type"),
        ("Descending neurons", "^descending", "superclass"),
        ("VNC motor neurons", "^vnc_motor", "superclass"),
        ("MN9 — proboscis", "^MN9$", "type"),
        ("Brain motor neurons", "^cb_motor", "superclass"),
        ("Kenyon cells", "^Kenyon", "class"),
    ],
}


def pack(header, arrays):
    """[u32 header size][JSON header][arrays aligned on 8 bytes]."""
    specs, blobs, off = [], [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        b = a.tobytes()
        specs.append(dict(name=name, dtype=a.dtype.str.lstrip("<|"), offset=off,
                          length=int(a.size)))
        pad = (-len(b)) % 8
        blobs.append(b + b"\0" * pad); off += len(b) + pad
    h = json.dumps(dict(header, arrays=specs)).encode()
    h += b" " * ((-(len(h) + 4)) % 8)
    return struct.pack("<I", len(h)) + h + b"".join(blobs)


class Jobs:
    def __init__(self, C):
        self.C, self.jobs, self.lock = C, {}, threading.Lock()

    def start(self, spec):
        C = self.C
        q = engine.resolve_params(spec.get("params", {}))
        stims = []
        for s in spec.get("stims", []):
            idx = C.find(s.get("query", ""), s.get("field", "any"))
            if not len(idx): continue
            stims.append(dict(idx=idx, hz=float(s.get("hz", 150)),
                              t_on=float(s.get("t_on", 0)),
                              t_off=float(s.get("t_off", q["t_ms"])), label=s.get("query")))
        if not stims: raise ValueError("no neuron stimulated: check the patterns")
        sil = spec.get("silence") or {}
        silence = C.find(sil.get("query", ""), sil.get("field", "any"))
        readouts = []
        for r in spec.get("readouts", [])[:4]:
            idx = C.find(r.get("query", ""), r.get("field", "any"))
            if len(idx): readouts.append(dict(idx=idx, label=r.get("label") or r.get("query")))
        with self.lock:
            for j in self.jobs.values():
                if j["state"] == "running": j["cancel"].set()
            jid = uuid.uuid4().hex[:10]
            job = dict(id=jid, state="running", progress=0.0, spikes=0, t0=time.time(),
                       cancel=threading.Event(), result=None, error=None)
            self.jobs = {k: v for k, v in self.jobs.items() if v["state"] == "running"}
            self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job, q, stims, silence, readouts),
                         daemon=True).start()
        return job

    def _run(self, job, q, stims, silence, readouts):
        C = self.C
        def prog(f, n): job["progress"], job["spikes"] = f, n
        try:
            t0 = time.time()
            runs = engine.simulate(C, stims, q, silence, prog, job["cancel"])
            wall = time.time() - t0
            stim_idx = np.unique(np.concatenate([s["idx"] for s in stims]))
            header, arrays = engine.summarize(C, runs, q, stim_idx, readouts)
            header.update(
                params=q, wall_s=wall,
                stims=[dict(label=s["label"], n=int(len(s["idx"])), hz=s["hz"],
                            t_on=s["t_on"], t_off=s["t_off"]) for s in stims],
                readouts=[dict(label=r["label"], n=int(len(r["idx"]))) for r in readouts],
                n_silenced=int(len(silence)), n_stim=int(len(stim_idx)),
                groups=engine.GROUPS)
            arrays["stim_idx"] = stim_idx.astype(np.int32)
            arrays["silenced"] = silence.astype(np.int32)
            job["result"] = pack(header, arrays)
            job["state"], job["progress"] = "done", 1.0
            print(f"simulation {job['id']}: {header['total_spikes']:,} spikes, {wall:.1f} s")
        except engine.Cancelled:
            job["state"] = "cancelled"
        except Exception as e:
            traceback.print_exc()
            job["state"], job["error"] = "error", str(e)


def build_meta(C):
    nt_code = {n: i for i, n in enumerate(NT_NAMES)}
    types = sorted(set(C.type))
    tid = {t: i for i, t in enumerate(types)}
    header = dict(
        N=C.N, n_edges=int(len(C.indices)), n_synapses=int(C.nsyn.sum()),
        groups=engine.GROUPS, nt_names=NT_NAMES, sides=SIDES, types=types,
        defaults=engine.DEFAULTS,
        presets={k: [dict(label=a, query=b, field=c) for a, b, c in v]
                 for k, v in PRESETS.items()},
        group_sizes=np.bincount(C.group, minlength=len(engine.GROUPS)).tolist(),
    )
    arrays = dict(
        pos=(C.pos * 0.008).astype(np.float32).ravel(),       # 8 nm voxels -> µm
        pos_est=C.pos_est.astype(np.uint8),
        group=C.group,
        nt=np.array([nt_code.get(n, len(NT_NAMES) - 2) for n in C.nt], np.uint8),
        side=np.array([SIDES.index(s) if s in SIDES else 3 for s in C.side], np.uint8),
        type_id=np.array([tid[t] for t in C.type], np.int32),
        out_deg=np.diff(C.indptr).astype(np.int32),
    )
    return pack(header, arrays)


def neuron_detail(C, i, top=25):
    ptr, order = C.csc()
    o = slice(C.indptr[i], C.indptr[i + 1])
    outs = [(int(p), int(n)) for p, n in zip(C.indices[o], C.nsyn[o])]
    ins_e = order[ptr[i]:ptr[i + 1]]
    ins = [(int(p), int(n)) for p, n in zip(C.pre[ins_e], C.nsyn[ins_e])]
    def fmt(lst):
        lst.sort(key=lambda x: -x[1])
        return [dict(idx=p, n=n, type=C.type[p], instance=C.inst[p],
                     nt=C.nt[p], sign=int(C.sign[p])) for p, n in lst[:top]]
    return dict(
        idx=i, body=int(C.body[i]), type=C.type[i], instance=C.inst[i],
        superclass=C.superclass[i], klass=C.klass[i], subclass=C.subclass[i],
        side=C.side[i], nt=C.nt[i], group=engine.GROUPS[C.group[i]],
        pos_est=bool(C.pos_est[i]),
        n_in=len(ins), n_out=len(outs),
        syn_in=int(sum(n for _, n in ins)), syn_out=int(sum(n for _, n in outs)),
        inputs=fmt(ins), outputs=fmt(outs))


def search(C, q, field):
    idx = C.find(q, field)
    by_type = Counter(C.type[i] or C.inst[i] or "?" for i in idx).most_common(12)
    return dict(count=int(len(idx)), by_type=by_type,
                by_group=np.bincount(C.group[idx], minlength=len(engine.GROUPS)).tolist())


class Handler(BaseHTTPRequestHandler):
    C = jobs = meta = live = None

    def log_message(self, fmt, *a): pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); p = u.path; qs = parse_qs(u.query)
        try:
            if p in ("/", "/fly") or p.startswith("/static/"):
                rel = {"/": "index.html", "/fly": "fly.html"}.get(p) or p[len("/static/"):]
                root = os.path.realpath(STATIC)
                f = os.path.realpath(os.path.join(root, rel))
                if not f.startswith(root + os.sep) or not os.path.isfile(f):
                    return self.send(404, {"error": "not found"})
                ct = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
                      ".css": "text/css", ".json": "application/json",
                      }.get(os.path.splitext(f)[1], "application/octet-stream")
                with open(f, "rb") as fh: return self.send(200, fh.read(), ct + "; charset=utf-8")
            if p == "/api/meta":
                return self.send(200, self.meta, "application/octet-stream")
            if p == "/api/search":
                return self.send(200, search(self.C, qs.get("q", [""])[0],
                                             qs.get("field", ["any"])[0]))
            if p == "/api/live/session":
                return self.send(200, self.live.current())
            if p == "/api/recordings":
                return self.send(200, self.live.recordings())
            if p == "/api/live/info":
                try: return self.send(200, self.live.info_of(qs.get("id", ["0"])[0]))
                except KeyError as e: return self.send(409, {"error": str(e)})
            if p == "/api/live/frames":
                try: s = self.live.get(qs.get("id", ["0"])[0])
                except KeyError as e: return self.send(409, {"error": str(e)})
                return self.send(200, s.read(int(qs.get("since", ["0"])[0])))
            if p.startswith("/api/neuron/"):
                i = int(p.rsplit("/", 1)[1])
                if not 0 <= i < self.C.N: return self.send(404, {"error": "idx out of range"})
                return self.send(200, neuron_detail(self.C, i))
            if p.startswith("/api/job/"):
                parts = p.split("/")
                job = self.jobs.jobs.get(parts[3])
                if job is None: return self.send(404, {"error": "unknown job"})
                if len(parts) > 4 and parts[4] == "result":
                    if job["state"] != "done": return self.send(409, {"error": job["state"]})
                    return self.send(200, job["result"], "application/octet-stream")
                return self.send(200, dict(state=job["state"], progress=job["progress"],
                                           spikes=job["spikes"], error=job["error"],
                                           elapsed=time.time() - job["t0"]))
            self.send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc(); self.send(400, {"error": str(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if p == "/api/run":
                job = self.jobs.start(body)
                return self.send(200, {"job": job["id"]})
            # another process (flyenv's Viewer) streams its frames: the page renders them
            if p == "/api/view/start":
                return self.send(200, self.live.remote_start(str(body.get("label") or "remote"), body.get("world")))
            if p == "/api/recordings/replay":
                try: return self.send(200, self.live.replay(body["name"]))
                except KeyError as e: return self.send(404, {"error": str(e)})
            if p == "/api/view/push":
                try: return self.send(200, self.live.push(body["id"], body.get("frames") or [], body.get("groups") or {}))
                except KeyError as e: return self.send(409, {"error": str(e)})
            if p.startswith("/api/live/"):
                act = p.rsplit("/", 1)[1]
                try:
                    if act == "start": return self.send(200, self.live.start(body.get("params", {})))
                    if act == "event": self.live.event(body["id"], body["key"])
                    elif act == "audio": self.live.audio(body["id"], body.get("level", 0), bool(body.get("hit")))
                    elif act == "drive": self.live.drive(body["id"], set(body.get("keys", [])), body.get("keyset", "walk"))
                    elif act == "reach": self.live.reach(body["id"], body["side"])
                    elif act == "world": return self.send(200, self.live.world(body["id"], body.get("world"), bool(body.get("reset"))))
                    elif act == "stim": return self.send(200, self.live.stim(
                        body["id"], str(body.get("name", "api")), body["query"], body.get("field", "type"),
                        body.get("side"), float(body.get("hz", 0)), body.get("ms")))
                    elif act == "stop": self.live.get(body["id"]).stop()
                    else: return self.send(404, {"error": "not found"})
                except (KeyError, StopIteration) as e:
                    return self.send(409, {"error": f"unknown session or event: {e}"})
                return self.send(200, {"ok": True})
            if p.startswith("/api/job/") and p.endswith("/cancel"):
                job = self.jobs.jobs.get(p.split("/")[3])
                if job: job["cancel"].set()
                return self.send(200, {"ok": True})
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(400, {"error": str(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    C = engine.Connectome()
    Handler.C, Handler.jobs, Handler.meta, Handler.live = C, Jobs(C), build_meta(C), live.Live(C)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dashboard ready: http://{a.host}:{a.port}   ·   3D fly: http://{a.host}:{a.port}/fly")
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
