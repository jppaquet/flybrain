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
import base64, collections, gzip, json, os, re, threading, time
import numpy as np
import engine, world

RECORD_DIR = os.path.join(engine.HERE, "runs", "recordings")   # every streamed run is kept here

FRAME_MS = 10.0
BRAIN_EVERY = 4        # one brain frame every 40 ms of simulated time
BRAIN_CAP = 4000       # neurons sent per brain frame (random sample beyond that)
RUNAWAY_N = 200        # while only keys drive the network: reset if more neurons than this fire
RUNAWAY_FRAMES = 30    # per 10 ms on average over 30 frames (key driving: < 120; runaway: 230 to 1,700)

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
    # both eyes: TTMn fires within ~15 ms, the reliable way to make the fly jump
    dict(key="loom", label="Looming, both eyes", kind="sens", query=r"^LC4$", field="type", hz=150, ms=300),
    # GRNs that recruit MN9 in this connectome at w_scale 0.5 (screened over the 60 gustatory types)
    dict(key="sugar", label="Taste (LB3c + taste pegs)", kind="sens", query=r"^(LB3c|claw_tpGRN)$", field="type", hz=150, ms=800),
    # Johnston's organ: JO-A/B pick up sound (vibrations), JO-C/E wind and gravity
    dict(key="sound", label="Sound (JO-B)", kind="sens", query=r"^JO-B", field="type", hz=200, ms=500),
    dict(key="wind", label="Wind (JO-C/E)", kind="sens", query=r"^JO-(C|E)", field="type", hz=150, ms=500),
    dict(key="cva", label="cVA odour (ORN DA1)", kind="sens", query=r"^ORN_DA1$", field="type", hz=150, ms=800),
    # direct activations whose effect is read downstream (giant fiber -> TTMn, pIP10 -> wing
    # motor neurons). Activating DNp09, MDN, DNa02 or MN9 directly is not offered: the body
    # would read back the very neurons being driven (see "always through synapses" below)
    dict(key="gf", label="Giant fiber", kind="opto", query=r"^DNp01$", field="type", hz=200, ms=100),
    dict(key="pip10", label="pIP10: song", kind="opto", query=r"^pIP10$", field="type", hz=150, ms=1500),
]
# Sound from the page -> JO-B (at w_scale 0.5: a graded, bounded response that dies out in
# ~40 ms). Not JO-A: added to JO-B, it inhibits the output. Not JO-C/E: above ~30 Hz they
# push the network into the self-sustained state. A background drive that follows the
# volume, and a short burst on every onset (kick drum): the motor output then follows the beat.
AUDIO_INPUT = dict(query=r"^JO-B", field="type", hz=30.0, hit_hz=300.0, hit_ms=80.0)
# Held keys -> neurons upstream of what the body reads, so that every command crosses
# synapses before it moves anything (the page resends held keys every ~100 ms).
KEYSETS = {
    # Keyboard mode: the body reads locomotion from DNp09, MDN and DNa01/02, so the keys
    # drive their strongest clean presynaptic partners (screen of their inputs): at 150 Hz
    # ICL012m brings DNp09 to ~32 Hz, DNpe023 brings MDN to ~63 Hz and LAL018 brings its
    # DNa01/02 to ~45 Hz, with at most a few hundred neurons active and no runaway.
    "walk": [
        dict(key="up", label="ICL012m → DNp09: walk forward", query=r"^ICL012m$", field="type", hz=150),
        dict(key="down", label="DNpe023 → MDN: walk backward", query=r"^DNpe023$", field="type", hz=150),
        dict(key="left", label="LAL018 left → DNa01/02 left: turn left", query=r"^LAL018$", field="type", side="L", hz=150),
        dict(key="right", label="LAL018 right → DNa01/02 right: turn right", query=r"^LAL018$", field="type", side="R", hz=150),
    ],
    # Drive mode: the kart reads leg motor neurons (world.CONTROLS), so the keys drive the
    # descending neurons that move each leg most specifically (screens of all DN types).
    # No DN extends the left hind leg cleanly: the brake is a lever the leg pulls (flexors).
    "car": [
        dict(key="up", label="DNg16 right → right hind leg extends: accelerator", query=r"^DNg16$", field="type", side="R", hz=150),
        dict(key="down", label="DNpe008 left → left hind leg flexes: brake", query=r"^DNpe008$", field="type", side="L", hz=150),
        dict(key="left", label="DNg12_e right → right front leg pushes the rim: turn left", query=r"^DNg12_e$", field="type", side="R", hz=150),
        dict(key="right", label="DNg12_e left → left front leg pushes the rim: turn right", query=r"^DNg12_e$", field="type", side="L", hz=150),
    ],
}
# Switchboard: a number key makes the fly press a switch with a front leg. The key drives
# DNg12_e on that side: in a screen of all 472 left DN types, the most specific front-leg
# descending neuron (alone it recruits ~20-30 neurons, mostly the coxa promotors, which
# swing the leg forward, 30-60 Hz within 50 ms at 250 Hz, and on the left the tibia
# extensors). The page reads the promotors back to move the leg, and the switch flips only
# if they fire enough. One leg at a time: both DNg12_e together tip the network into its
# self-sustained state (~3,500 neurons).
REACH = [dict(side=s, label=f"DNg12_e {'left' if s == 'L' else 'right'}: reach with the front leg",
              query=r"^DNg12_e$", field="type", hz=250, ms=400, ref=12, press=16,   # flips at 16 Hz (peaks: 30-60)
              read=[f"T1{s}_pro"], read_label=f"T1{s} coxa promotors")
         for s in "LR"]


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
        self.keys = [c["key"] for c in self.chans]
        self.kind, self.label, self.groups = "sim", None, {}     # groups: inputs named by scripts
        self.driver = world.Driver() if params.get("world") == "kart" else None
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

    def set_world(self, name, reset=False):
        """name "kart": the legs' motor neurons drive world.Driver; None: no world."""
        with self.lock:
            if name == "kart":
                if self.driver is None: self.driver = world.Driver()
                elif reset: self.driver.reset()
            else:
                self.driver = None

    def _run(self):
        n = max(1, int(round(FRAME_MS / self.q["dt"])))
        w0, s0 = time.time(), self.sim.t_ms
        quench_ms, last_input = self.q.get("quench_ms", 0.0), 0.0
        acc, rng, recent = [], np.random.default_rng(0), collections.deque(maxlen=RUNAWAY_FRAMES)
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
                # always through synapses: the body never reads a neuron that a stimulus
                # drives directly, only what the network makes of it downstream
                read = counts
                if drive:
                    read = counts.copy()
                    for idx, _ in drive: read[idx] = 0
                k = 1000.0 / FRAME_MS
                rates = [round(float(read[c["idx"]].sum()) * k / len(c["idx"]), 1) for c in self.chans]
                nact, note = int(np.count_nonzero(counts)), None
                # safeguard: the uniform LIF sometimes falls into a self-sustained state
                # (mostly in the central complex) that never dies out on its own
                if quench_ms and not names and nact > 20 and self.sim.t_ms - last_input > quench_ms:
                    self.sim.quench(); last_input = self.sim.t_ms
                    note = (f"Safeguard: self-sustained activity switched off ({nact} neurons still active "
                            f"{quench_ms / 1000:.1f} s after the last stimulus)")
                # while driving with the keys, the same state can start under the drive itself
                # (other inputs, e.g. the looming jump, legitimately recruit more: not counted)
                if names and all(k.startswith("key:") for k in names): recent.append(nact)
                else: recent.clear()
                if len(recent) == RUNAWAY_FRAMES and sum(recent) / RUNAWAY_FRAMES > RUNAWAY_N:
                    self.sim.quench(); recent.clear()
                    note = f"Safeguard: runaway while driving switched off ({nact:,} neurons firing per 10 ms)"
                # neurons that fired over the last BRAIN_EVERY slices, as base64 int32
                acc.append(fired); brain = None
                if len(acc) >= BRAIN_EVERY:
                    u = np.unique(np.concatenate(acc)); acc = []
                    if len(u) > BRAIN_CAP: u = np.sort(rng.choice(u, BRAIN_CAP, replace=False))
                    brain = base64.b64encode(u.astype("<i4").tobytes()).decode()
                kart, drv = None, self.driver
                if drv is not None:                                  # the legs drive the kart
                    drv.frame(dict(zip(self.keys, rates)), FRAME_MS)
                    kart = drv.kart.frame()
                frame = [round(self.sim.t_ms, 1), rates, int(len(fired)), nact, names, note, brain, kart]
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

    @property
    def alive(self):
        return self.thread.is_alive() and not self.stop_ev.is_set()

    @property
    def world_name(self):
        return "kart" if self.driver is not None else None


class Recorder:
    """Keeps a streamed run as runs/recordings/<date-time>-<label>.jsonl.gz: a header line,
    then the frames as they arrived (and {"groups": ...} lines for the input groups), so
    that the page can replay it later - and film it. Flushed on every push, so a file being
    written, or cut short, is still readable."""
    def __init__(self, label, world_name):
        os.makedirs(RECORD_DIR, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")[:60] or "run"
        base = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
        self.name, k = f"{base}.jsonl.gz", 1
        while os.path.exists(os.path.join(RECORD_DIR, self.name)):
            k += 1; self.name = f"{base}-{k}.jsonl.gz"
        self.f = gzip.open(os.path.join(RECORD_DIR, self.name), "wt", encoding="utf-8")
        self.f.write(json.dumps(dict(label=label, world=world_name, frame_ms=FRAME_MS,
                                     started=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")

    def write(self, frames, groups):
        if groups: self.f.write(json.dumps(dict(groups=groups), separators=(",", ":")) + "\n")
        for fr in frames: self.f.write(json.dumps(fr, separators=(",", ":")) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_recording(name):
    """-> (header, groups, frames) of a recording; tolerant of a file still being written."""
    path = os.path.join(RECORD_DIR, os.path.basename(str(name)))
    if not path.endswith(".jsonl.gz") or not os.path.isfile(path): raise KeyError(f"unknown recording {name}")
    head, groups, frames = {}, {}, []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                if i == 0: head = x
                elif isinstance(x, dict): groups.update(x.get("groups", {}))
                else: frames.append(x)
    except (EOFError, OSError, ValueError):
        pass                                       # still being written, or cut short: keep what was read
    return head, groups, frames


def list_recordings(limit=300):
    out = []
    if not os.path.isdir(RECORD_DIR): return out
    for fn in sorted(os.listdir(RECORD_DIR), reverse=True):
        if not fn.endswith(".jsonl.gz"): continue
        p = os.path.join(RECORD_DIR, fn)
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f: head = json.loads(f.readline())
        except (EOFError, OSError, ValueError):
            head = {}
        out.append(dict(name=fn, label=head.get("label", fn), world=head.get("world"),
                        started=head.get("started"), bytes=os.path.getsize(p)))
        if len(out) >= limit: break
    return out


class Remote:
    """A session simulated in another process (flyenv with a Viewer), which streams its
    frames here in the same format: the page renders them exactly like its own simulation.
    Nothing is simulated in the server, and it takes no input (they belong to the process).
    Streamed runs are recorded (Recorder); replays are not."""
    kind = "remote"

    def __init__(self, label, world_name=None, record=True):
        self.label, self.world_name, self.groups = label, world_name, {}
        self.frames, self.base, self.t_ms = [], 0, 0.0
        self.lock, self.last_push, self.stopped = threading.Lock(), time.time(), False
        self.rec = Recorder(label, world_name) if record else None

    @property
    def alive(self):
        return not self.stopped and time.time() - self.last_push < 15

    def push(self, frames, groups):
        frames = [f for f in frames if isinstance(f, list) and len(f) >= 8]
        with self.lock:
            self.groups.update(groups)
            self.frames.extend(frames)
            if self.frames: self.t_ms = self.frames[-1][0]
            if len(self.frames) > 6000:
                self.frames = self.frames[3000:]; self.base += 3000
            self.last_push = time.time()
            if self.rec: self.rec.write(frames, groups)

    def read(self, since, limit=300):
        with self.lock:
            i = min(max(0, since - self.base), len(self.frames))
            out = self.frames[i:i + limit]
            return dict(frames=out, next=self.base + i + len(out), speed=1.0, t_ms=round(self.t_ms, 1),
                        inputs=out[-1][4] if out else [], error=None, alive=self.alive)

    def stimulate(self, *args, **kwargs):
        raise KeyError(f"this session is simulated by {self.label}: drive it from there")

    set_world = stimulate

    def stop(self):
        with self.lock:
            self.stopped = True
            if self.rec: self.rec.close(); self.rec = None


class Live:
    """One session at a time (local server, single user)."""
    def __init__(self, C):
        self.C, self.session, self.sid, self.info = C, None, 0, None
        self._ev, self._jo, self._reach, self._keys, self._api = None, None, None, {}, {}

    def event_idx(self):
        """Neurons driven by each event (regex over 165k neurons: computed once)."""
        if self._ev is None:
            self._ev = {e["key"]: select(self.C, e["query"], e["field"], e.get("side")) for e in EVENTS}
        return self._ev

    def keyset_idx(self, name):
        if name not in self._keys:
            self._keys[name] = {d["key"]: select(self.C, d["query"], d["field"], d.get("side")) for d in KEYSETS[name]}
        return self._keys[name]

    def reach_idx(self):
        if self._reach is None:
            self._reach = {r["side"]: select(self.C, r["query"], r["field"], r["side"]) for r in REACH}
        return self._reach

    def audio_idx(self):
        if self._jo is None: self._jo = select(self.C, AUDIO_INPUT["query"], AUDIO_INPUT["field"])
        return self._jo

    def _info(self, **extra):
        """What a page needs to render a session: channels, inputs, key sets, track…"""
        ev, rc = self.event_idx(), self.reach_idx()
        return dict(id=self.sid, frame_ms=FRAME_MS, brain_ms=FRAME_MS * BRAIN_EVERY,
                    channels=[dict(key=c["key"], label=c["label"], group=c["group"], n=int(len(c["idx"])))
                              for c in channels(self.C)],
                    events=[dict(e, n=int(len(ev[e["key"]])), idx=ev[e["key"]].tolist()) for e in EVENTS],
                    keysets={name: [dict(d, n=int(len(self.keyset_idx(name)[d["key"]])),
                                         idx=self.keyset_idx(name)[d["key"]].tolist()) for d in ks]
                             for name, ks in KEYSETS.items()},
                    controls=world.CONTROLS, track=world.Track().to_dict(), kart_keys=world.Kart.FRAME_KEYS,
                    reach=[dict(r, n=int(len(rc[r["side"]])), idx=rc[r["side"]].tolist()) for r in REACH],
                    audio_idx=self.audio_idx().tolist(), **extra)

    def start(self, params):
        if self.session: self.session.stop()
        self.sid += 1
        self.session = Session(self.C, params)
        self.info = self._info(params=self.session.q, remote=None, world=self.session.world_name)
        return self.info

    def remote_start(self, label, world_name=None, record=True):
        """Another process (flyenv's Viewer) streams its own simulation: it becomes the
        current session, which the page follows, and it is recorded."""
        if self.session: self.session.stop()
        self.sid += 1
        self.session = Remote(label, world_name, record)
        self.info = self._info(params=None, remote=label, world=world_name)
        return dict(self.info, recording=self.session.rec.name if self.session.rec else None)

    def recordings(self):
        return list_recordings()

    def replay(self, name):
        """Plays a recording again, at its own pace, as a new streamed session: the page
        follows it like a live run (and can film it)."""
        head, groups, frames = read_recording(name)
        if not frames: raise KeyError(f"{name}: no frame recorded")
        info = self.remote_start(f"Replay · {head.get('label', name)}", head.get("world"), record=False)
        s = self.session
        s.push([], groups)
        threading.Thread(target=self._play, args=(s, frames), daemon=True).start()
        return info

    def _play(self, s, frames):
        t0, first = time.time(), frames[0][0]
        for i in range(0, len(frames), 5):                     # 50 ms of frames at a time
            if s.stopped: return                               # replaced by another session
            ahead = (frames[i][0] - first) / 1000.0 - (time.time() - t0)
            if ahead > 0: time.sleep(ahead)
            s.push(frames[i:i + 5], {})

    def push(self, sid, frames, groups):
        s = self.get(sid)
        if not isinstance(s, Remote): raise KeyError("this session is not a streamed one")
        s.push(frames, groups)
        return dict(ok=True, n=len(frames))

    def get(self, sid):
        if self.session is None or int(sid) != self.sid: raise KeyError("unknown or replaced session")
        return self.session

    def info_of(self, sid):
        """The start information of the current session, to follow it from a page."""
        s = self.get(sid)
        return dict(self.info, id=self.sid, groups=s.groups, world=s.world_name, remote=s.label)

    def current(self):
        """The running session, so that a page or a script can follow or join it."""
        s = self.session
        return dict(id=self.sid if s else None, alive=bool(s and s.alive), kind=s.kind if s else None,
                    label=s.label if s else None, world=s.world_name if s else None, frame_ms=FRAME_MS)

    def event(self, sid, key):
        e = next(e for e in EVENTS if e["key"] == key)
        self.get(sid).stimulate(f"ev:{key}", self.event_idx()[key], e["hz"], e["ms"])

    def audio(self, sid, level, hit=False):
        """Sound picked up by the page -> JO-B: background proportional to the level (0..1), burst on an onset."""
        s, A, idx = self.get(sid), AUDIO_INPUT, self.audio_idx()
        level = min(max(float(level), 0.0), 1.0)
        if hit: s.stimulate("audio:hit", idx, A["hit_hz"], A["hit_ms"])
        else: s.stimulate("audio", idx, A["hz"] * level, 400)

    def drive(self, sid, keys, keyset="walk"):
        """Keys held on the page -> the neurons of KEYSETS[keyset]. The page resends the
        held keys every ~100 ms; each input expires after 300 ms, so a page that goes away
        releases its keys by itself."""
        s, idx = self.get(sid), self.keyset_idx(keyset)
        for d in KEYSETS[keyset]:
            s.stimulate(f"key:{keyset}:{d['key']}", idx[d["key"]], d["hz"] if d["key"] in keys else 0, 300)

    def world(self, sid, name, reset=False):
        """Turns the kart on ("kart") or off (None) in the running session."""
        self.get(sid).set_world(name, reset)
        return dict(world=name, track=world.Track().to_dict() if name == "kart" else None,
                    kart_keys=world.Kart.FRAME_KEYS)

    def stim(self, sid, name, query, field="type", side=None, hz=0.0, ms=None):
        """Any group of neurons, for scripts and agents: Poisson drive at `hz` for `ms`
        (None: until set again; hz 0 stops it). The body never reads the driven neurons."""
        s, k = self.get(sid), (query, field, side)
        if k not in self._api: self._api[k] = select(self.C, query, field, side)
        idx = self._api[k]
        s.stimulate(f"api:{name}", idx, float(hz), None if ms is None else float(ms))
        s.groups[f"api:{name}"] = idx.tolist()                 # lets the page tint them in the 3D brain
        return dict(name=name, n=int(len(idx)))

    def reach(self, sid, side):
        """Number key on the switchboard -> a burst to the front-leg descending neuron of that side."""
        r = next(r for r in REACH if r["side"] == side)
        self.get(sid).stimulate(f"reach:{side}", self.reach_idx()[side], r["hz"], r["ms"])
