"""
live - continuous simulation of the connectome, driving the 3D fly.

Like Eon Systems (LIF brain -> simulated body), but the readout goes all the way down to
the VNC motor neurons, which MaleCNS contains. A session runs in a thread: the dynamics
advance in FRAME_MS slices with the current stimuli (short events, or continuous inputs
such as sound -> Johnston's organ). For each slice we compute the firing rate of "channels":
  - a few descending neurons, which drive a procedural walking controller
    (the LIF has no rhythm generator: the same limit as Eon's);
  - groups of motor neurons (legs by muscle, wings, neck, abdomen, proboscis, antennae),
    translated directly into posture.
Every BRAIN_EVERY slices, the frame also lists the neurons that fired (for the 3D brain).
"""
import base64, re, threading, time
import numpy as np
import engine

FRAME_MS = 10.0
BRAIN_EVERY = 4        # one brain frame every 40 ms of simulated time
BRAIN_CAP = 4000       # neurons sent per brain frame (random sample beyond that)

LEGS = [("fl", "T1"), ("ml", "T2"), ("hl", "T3")]
LEG_FUNCS = [  # (key, label, pattern on the motor neuron type)
    ("pro", "coxa promotors", r"Sternal anterior rotator|promotor"),
    ("rem", "coxa remotors", r"Sternal posterior rotator|remotor"),
    ("trx", "trochanter extensors", r"^Tr extensor|^Sternotrochanter|^Tergotr|^TTMn$|^STTMm$"),
    ("trf", "trochanter flexors", r"Tr flexor"),
    ("tix", "tibia extensors", r"^Ti extensor"),
    ("tif", "tibia flexors", r"Ti flexor"),
    ("tad", "tarsus depressors", r"^Ta depressor|^ltm"),
    ("tal", "tarsus levators", r"^Ta levator"),
]

# Available stimuli: sensory, or "optogenetic" (direct activation of a DN)
EVENTS = [
    dict(key="loomL", label="Looming, left", kind="sens", query=r"^LC4$", field="type", side="L", hz=150, ms=300),
    dict(key="loomR", label="Looming, right", kind="sens", query=r"^LC4$", field="type", side="R", hz=150, ms=300),
    # GRNs that recruit MN9 in this connectome at w_scale 0.5 (screened over the 60 gustatory types)
    dict(key="sugar", label="Taste (LB3c + taste pegs)", kind="sens", query=r"^(LB3c|claw_tpGRN)$", field="type", hz=150, ms=800),
    # Johnston's organ: JO-A/B pick up sound (vibrations), JO-C/E wind and gravity
    dict(key="sound", label="Sound (JO-B)", kind="sens", query=r"^JO-B", field="type", hz=200, ms=500),
    dict(key="wind", label="Wind (JO-C/E)", kind="sens", query=r"^JO-(C|E)", field="type", hz=150, ms=500),
    dict(key="cva", label="cVA odour (ORN DA1)", kind="sens", query=r"^ORN_DA1$", field="type", hz=150, ms=800),
    dict(key="gf", label="Giant fiber", kind="opto", query=r"^DNp01$", field="type", hz=200, ms=100),
    dict(key="p09", label="DNp09: walk", kind="opto", query=r"^DNp09$", field="type", hz=150, ms=1500),
    dict(key="mdn", label="MDN: walk backward", kind="opto", query=r"^MDN$", field="type", hz=150, ms=1500),
    dict(key="a02L", label="DNa02 left", kind="opto", query=r"^DNa0[12]$", field="type", side="L", hz=150, ms=1000),
    dict(key="a02R", label="DNa02 right", kind="opto", query=r"^DNa0[12]$", field="type", side="R", hz=150, ms=1000),
    dict(key="pip10", label="pIP10: song", kind="opto", query=r"^pIP10$", field="type", hz=150, ms=1500),
    dict(key="mn9", label="MN9: proboscis", kind="opto", query=r"^MN9$", field="type", hz=150, ms=800),
]
# Sound from the page -> JO-B (at w_scale 0.5: a graded, bounded response that dies out in
# ~40 ms). Not JO-A: added to JO-B, it inhibits the output. Not JO-C/E: above ~30 Hz they
# push the network into the self-sustained state. A background drive that follows the
# volume, and a short burst on every onset (kick drum): the motor output then follows the beat.
AUDIO_INPUT = dict(query=r"^JO-B", field="type", hz=30.0, hit_hz=300.0, hit_ms=80.0)


def select(C, query, field="type", side=None):
    idx = C.find(query, field)
    if side: idx = idx[np.array([C.side[i] == side for i in idx], dtype=bool)] if len(idx) else idx
    return idx


def channels(C):
    """Channels read at every slice. Cached on the Connectome object."""
    if getattr(C, "_live_channels", None) is not None: return C._live_channels
    M = [i for i in range(C.N) if C.superclass[i] in ("vnc_motor", "cb_motor")]
    ty, sub, side = C.type, C.subclass, C.side
    motor = lambda pred: np.array([i for i in M if pred(i)], dtype=np.int32)
    out = []
    def add(key, label, group, idx):
        if len(idx): out.append(dict(key=key, label=label, group=group, idx=np.asarray(idx, np.int32)))
    add("jo", "JO-B (sound)", "hearing", select(C, AUDIO_INPUT["query"], AUDIO_INPUT["field"]))
    add("esc", "DNp02/06/11 (escape)", "hearing", select(C, r"^DNp(02|06|11)$"))
    add("fwd", "DNp09 (walking)", "locomotion", select(C, r"^DNp09$"))
    add("back", "MDN (backward)", "locomotion", select(C, r"^MDN$"))
    for s in "LR":
        add(f"turn{s}", f"DNa01/02 {s}", "locomotion", select(C, r"^DNa0[12]$", side=s))
    add("gf", "Giant fiber", "jump", select(C, r"^DNp01$"))
    add("ttmn", "TTMn (jump)", "jump", select(C, r"^TTMn$"))
    add("mn9", "MN9 (rostrum)", "proboscis", select(C, r"^MN9$"))
    add("pm", "Other proboscis MNs", "proboscis", motor(lambda i: sub[i] == "pm" and ty[i] != "MN9"))
    for s in "LR":
        add(f"wpow{s}", f"Flight DLM/DVM {s}", "wings", motor(lambda i: sub[i] == "wm" and re.match(r"^D[LV]Mn", ty[i]) and side[i] == s))
        add(f"wstr{s}", f"Wing steering {s}", "wings", motor(lambda i: sub[i] == "wm" and not re.match(r"^D[LV]Mn", ty[i]) and side[i] == s))
        add(f"hal{s}", f"Haltere {s}", "wings", motor(lambda i: sub[i] == "hm" and side[i] == s))
        add(f"neck{s}", f"Neck {s}", "head", motor(lambda i: sub[i] == "nm" and side[i] == s))
        add(f"ant{s}", f"Antenna {s}", "head", motor(lambda i: sub[i] == "am" and side[i] == s))
        add(f"abd{s}", f"Abdomen {s}", "abdomen", motor(lambda i: sub[i] == "ad" and side[i] == s))
        for code, leg in LEGS:
            for f, flabel, rx in LEG_FUNCS:
                add(f"{leg}{s}_{f}", f"{leg}{s} {flabel}", "legs",
                    motor(lambda i: sub[i] == code and side[i] == s and re.search(rx, ty[i])))
    C._live_channels = out
    return out


class Session:
    def __init__(self, C, params):
        self.C, self.q = C, engine.resolve_params(params)
        self.sim = engine.Stepper(C, self.q)
        self.chans = channels(C)
        self.frames, self.base = [], 0
        self.inputs = {}                    # name -> dict(idx, hz, until)
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.last_poll, self.speed, self.error = time.time(), 0.0, None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stimulate(self, name, idx, hz, ms=None):
        with self.lock:
            if hz <= 0 or not len(idx): self.inputs.pop(name, None); return
            until = None if ms is None else self.sim.t_ms + ms
            self.inputs[name] = dict(idx=idx, hz=float(hz), until=until)

    def _run(self):
        n = max(1, int(round(FRAME_MS / self.q["dt"])))
        w0, s0 = time.time(), self.sim.t_ms
        quench_ms, last_input = self.q.get("quench_ms", 0.0), 0.0
        acc, rng = [], np.random.default_rng(0)
        try:
            while not self.stop_ev.is_set():
                if time.time() - self.last_poll > 20: break        # nobody is watching any more
                with self.lock:
                    now = self.sim.t_ms
                    for k in [k for k, e in self.inputs.items() if e["until"] is not None and e["until"] <= now]:
                        del self.inputs[k]
                    drive = [(e["idx"], e["hz"]) for e in self.inputs.values()]
                    names = sorted(self.inputs)
                    if names: last_input = now
                t = time.time()
                fired = self.sim.step(n, drive)
                counts = np.bincount(fired, minlength=self.C.N)
                k = 1000.0 / FRAME_MS
                rates = [round(float(counts[c["idx"]].sum()) * k / len(c["idx"]), 1) for c in self.chans]
                nact, note = int(np.count_nonzero(counts)), None
                # safeguard: the uniform LIF sometimes falls into a self-sustained state
                # (mostly in the central complex) that never dies out on its own
                if quench_ms and not names and nact > 20 and self.sim.t_ms - last_input > quench_ms:
                    self.sim.quench(); last_input = self.sim.t_ms
                    note = (f"Safeguard: self-sustained activity switched off ({nact} neurons still active "
                            f"{quench_ms / 1000:.1f} s after the last stimulus)")
                # neurons that fired over the last BRAIN_EVERY slices, as base64 int32
                acc.append(fired); brain = None
                if len(acc) >= BRAIN_EVERY:
                    u = np.unique(np.concatenate(acc)); acc = []
                    if len(u) > BRAIN_CAP: u = np.sort(rng.choice(u, BRAIN_CAP, replace=False))
                    brain = base64.b64encode(u.astype("<i4").tobytes()).decode()
                frame = [round(self.sim.t_ms, 1), rates, int(len(fired)), nact, names, note, brain]
                with self.lock:
                    self.frames.append(frame)
                    if len(self.frames) > 6000:
                        self.frames = self.frames[3000:]; self.base += 3000
                dt = time.time() - t
                self.speed = 0.9 * self.speed + 0.1 * (FRAME_MS / 1000.0 / max(dt, 1e-6))
                ahead = (self.sim.t_ms - s0) / 1000.0 - (time.time() - w0)
                if ahead > 0: time.sleep(ahead)                     # never faster than real time
                else: w0, s0 = time.time(), self.sim.t_ms           # running late: do not catch up
        except Exception as e:
            self.error = str(e)
            raise

    def read(self, since, limit=300):
        self.last_poll = time.time()
        with self.lock:
            i = min(max(0, since - self.base), len(self.frames))
            out = self.frames[i:i + limit]
            return dict(frames=out, next=self.base + i + len(out), speed=round(min(self.speed, 1.0), 3),
                        t_ms=round(self.sim.t_ms, 1), inputs=sorted(self.inputs), error=self.error,
                        alive=self.thread.is_alive())

    def stop(self):
        self.stop_ev.set()


class Live:
    """One session at a time (local server, single user)."""
    def __init__(self, C):
        self.C, self.session, self.sid = C, None, 0
        self._ev, self._jo = None, None

    def event_idx(self):
        """Neurons driven by each event (regex over 165k neurons: computed once)."""
        if self._ev is None:
            self._ev = {e["key"]: select(self.C, e["query"], e["field"], e.get("side")) for e in EVENTS}
        return self._ev

    def audio_idx(self):
        if self._jo is None: self._jo = select(self.C, AUDIO_INPUT["query"], AUDIO_INPUT["field"])
        return self._jo

    def start(self, params):
        if self.session: self.session.stop()
        self.sid += 1
        self.session = Session(self.C, params)
        ev = self.event_idx()
        return dict(id=self.sid, frame_ms=FRAME_MS, brain_ms=FRAME_MS * BRAIN_EVERY, params=self.session.q,
                    channels=[dict(key=c["key"], label=c["label"], group=c["group"], n=int(len(c["idx"])))
                              for c in self.session.chans],
                    events=[dict(e, n=int(len(ev[e["key"]])), idx=ev[e["key"]].tolist()) for e in EVENTS],
                    audio_idx=self.audio_idx().tolist())

    def get(self, sid):
        if self.session is None or int(sid) != self.sid: raise KeyError("unknown or replaced session")
        return self.session

    def event(self, sid, key):
        e = next(e for e in EVENTS if e["key"] == key)
        self.get(sid).stimulate(f"ev:{key}", self.event_idx()[key], e["hz"], e["ms"])

    def audio(self, sid, level, hit=False):
        """Sound picked up by the page -> JO-B: background proportional to the level (0..1), burst on an onset."""
        s, A, idx = self.get(sid), AUDIO_INPUT, self.audio_idx()
        level = min(max(float(level), 0.0), 1.0)
        if hit: s.stimulate("audio:hit", idx, A["hit_hz"], A["hit_ms"])
        else: s.stimulate("audio", idx, A["hz"] * level, 400)
