"""
flyenv - the fly as an end user, for machine learning (no dependency beyond numpy).

Everything goes through synapses. An agent acts only by driving groups of neurons with
Poisson spikes (sensory neurons, descending neurons, any group), and observes only what
the network makes of it: channel rates are read with the driven neurons masked out, the
same rule as the live 3D page. The simulation steps synchronously, as fast as the CPU
allows (roughly real time on a laptop), with no wall clock.

The environments follow the Gymnasium API without importing it:

    import numpy as np, flyenv
    env = flyenv.DriveEnv()                        # loads the connectome once (~1 s)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(len(env.action_names)))
    env.action_names, env.action_high, env.observation_names

Lower level, any experiment:

    brain = flyenv.FlyBrain()
    lal = brain.group("^LAL018$", side="L")
    rates = brain.step({lal: 150}, ms=100)         # {channel key: Hz} over those 100 ms
    rates["turnL"]                                 # DNa01/02 left, driven through synapses
"""
import base64, json, math, queue, threading, time, urllib.error, urllib.request
import numpy as np
import engine, live, world

LIVE_PARAMS = dict(model="shiu", dt=0.2, w_scale=0.5)   # the live page's defaults
_C = None


def connectome():
    """The MaleCNS connectome, loaded once per process and shared by every environment."""
    global _C
    if _C is None: _C = engine.Connectome()
    return _C


class Group:
    """A named group of neurons (indices into the connectome)."""
    def __init__(self, name, idx, spec):
        self.name, self.idx, self.spec = name, np.asarray(idx, np.int32), spec

    def __repr__(self):
        return f"Group({self.name!r}, {len(self.idx)} neurons)"


class Viewer:
    """Streams a flyenv simulation to the running server (./run.sh), so that the 3D page
    (/fly) renders it live - the fly, its 3D brain and, with DriveEnv, the kart - while
    your code trains. The page follows the server's current session by itself.

        env = flyenv.DriveEnv(viewer=True)                 # or viewer="http://host:8765"
        env.brain.viewer.note("best policy so far")        # shown in the page's log

    Frames leave in batches from a background thread and never slow the simulation down;
    they are dropped if the server is unreachable. If someone takes the server back (Start
    simulation on the page, or another script), streaming stops until reconnect().
    realtime=True slows the simulation down to real time, to watch a policy calmly."""
    BRAIN_EVERY, BRAIN_CAP = live.BRAIN_EVERY, live.BRAIN_CAP

    def __init__(self, url="http://127.0.0.1:8765", label="flyenv", world=None, realtime=False, quiet=False):
        self.url, self.label, self.world, self.realtime, self.quiet = url.rstrip("/"), label, world, realtime, quiet
        self.id, self.t_ms, self.acc, self.notes, self.wall0 = None, 0.0, [], [], None
        self.groups, self.pending = {}, {}             # input groups (name -> indices), and those to send
        self.q, self.active, self.warned, self.busy = queue.Queue(maxsize=3000), True, False, False
        threading.Thread(target=self._run, daemon=True).start()

    def note(self, text):
        """A message shown in the page's event log (one per frame, in order)."""
        self.notes.append(text)

    def reconnect(self):
        """Resume streaming in a new session after the server was taken over."""
        self.id, self.active = None, True

    def flush(self, timeout=3.0):
        """Waits until the frames and notes already produced have left (end of a script)."""
        end = time.time() + timeout
        while (not self.q.empty() or self.notes or self.busy) and self.active and time.time() < end:
            time.sleep(0.05)

    def frame(self, rates, spikes, inputs, kart=None, groups=None):
        """One 10 ms frame: channel rates (list, live.channels order, driven neurons masked),
        neurons that fired, names of the active inputs, kart frame or None."""
        if not self.active: return
        self.t_ms += live.FRAME_MS                     # the page's own clock: never goes back
        for name, idx in (groups or {}).items():
            if name not in self.groups:
                self.groups[name] = self.pending[name] = [int(i) for i in idx]
        self.acc.append(spikes)
        b64 = None
        if len(self.acc) >= self.BRAIN_EVERY:
            u = np.unique(np.concatenate(self.acc)); self.acc = []
            if len(u) > self.BRAIN_CAP: u = np.sort(np.random.default_rng(0).choice(u, self.BRAIN_CAP, replace=False))
            b64 = base64.b64encode(u.astype("<i4").tobytes()).decode()
        f = [round(self.t_ms, 1), rates, int(len(spikes)), int(len(spikes)), list(inputs),
             self.notes.pop(0) if self.notes else None, b64, kart]
        try: self.q.put_nowait(f)
        except queue.Full: pass                        # the server is too slow: drop, never block
        if self.realtime:
            now = time.time()
            if self.wall0 is None: self.wall0 = now - self.t_ms / 1000
            ahead = self.t_ms / 1000 - (now - self.wall0)
            if ahead > 0: time.sleep(ahead)
            elif ahead < -0.5: self.wall0 = now - self.t_ms / 1000

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def _run(self):
        last = None
        while True:
            try:
                batch = [self.q.get(timeout=0.3)]
            except queue.Empty:                        # idle (e.g. an episode just ended): pending notes
                if not (self.notes and last and self.active): continue
                self.t_ms += live.FRAME_MS             # ride on a copy of the last frame
                batch = [[round(self.t_ms, 1)] + last[1:5] + [self.notes.pop(0), None] + last[7:]]
            self.busy = True
            time.sleep(0.1)                            # ~10 requests per second at most
            while True:
                try: batch.append(self.q.get_nowait())
                except queue.Empty: break
            last = batch[-1]
            if not self.active: self.busy = False; continue
            try:
                if self.id is None:
                    self.id = self._post("/api/view/start", dict(label=self.label, world=self.world))["id"]
                    self.pending = dict(self.groups)
                groups, self.pending = self.pending, {}
                self._post("/api/view/push", dict(id=self.id, frames=batch, groups=groups))
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    self.active = False
                    if not self.quiet:
                        print("flyenv viewer: the server's session was taken over; streaming stopped "
                              "(viewer.reconnect() resumes)", flush=True)
            except Exception as e:
                if not self.warned and not self.quiet:
                    print(f"flyenv viewer: {self.url} unreachable ({e}); frames are dropped", flush=True)
                    self.warned = True
                time.sleep(2)
            finally:
                self.busy = False


def _viewer(v, label, world_name):
    """viewer argument of the environments: None/False, True, a URL, or a Viewer."""
    if not v: return None
    if isinstance(v, Viewer): return v
    return Viewer(url=v if isinstance(v, str) else "http://127.0.0.1:8765", label=label, world=world_name)


class FlyBrain:
    """Leaky integrate-and-fire simulation of the whole CNS, advanced in 10 ms frames.

    Channels are the live page's readouts (live.channels): descending neurons for
    locomotion, motor neurons of every leg muscle group, wings, neck, antennae, abdomen,
    proboscis. `channel_keys` lists them. viewer=True streams every frame to the 3D page."""
    FRAME_MS = live.FRAME_MS

    def __init__(self, params=None, seed=0, viewer=None):
        self.C = connectome()
        self.q = engine.resolve_params(dict(LIVE_PARAMS, **(params or {})))
        self.chans = live.channels(self.C)
        self.channel_keys = [c["key"] for c in self.chans]
        self.viewer = _viewer(viewer, "FlyBrain", None)
        self.world_frame = None                        # callable -> kart frame, set by DriveEnv
        self.reset(seed)

    def reset(self, seed=0):
        """Network at rest, clock at 0."""
        self.sim = engine.Stepper(self.C, self.q, seed=seed)
        self.t_ms, self.n_active, self.spikes = 0.0, 0, np.zeros(0, np.int32)

    def group(self, query, field="type", side=None, name=None):
        """Neurons matching a regex on a field (type, instance, class, superclass, any, body, idx)."""
        idx = live.select(self.C, query, field, side)
        if not len(idx): raise ValueError(f"no neuron matches {query!r} (field {field}, side {side})")
        return Group(name or (f"{query} {side}" if side else query), idx, dict(query=query, field=field, side=side))

    def step(self, drive=None, ms=None):
        """Advances `ms` (rounded to 10 ms frames; default one frame) with the Poisson drive
        {Group or index array: Hz}. Returns {channel key: mean rate in Hz} over that time,
        never counting the spikes of driven neurons. Sets .spikes (neurons that fired) and
        .n_active."""
        frames = max(1, int(round((ms or self.FRAME_MS) / self.FRAME_MS)))
        n = max(1, int(round(self.FRAME_MS / self.q["dt"])))
        active = [(g, hz) for g, hz in (drive or {}).items() if hz > 0]
        pairs = [(g.idx if isinstance(g, Group) else np.asarray(g, np.int32), float(hz)) for g, hz in active]
        tot = np.zeros(self.C.N, np.int64)
        for _ in range(frames):
            c = np.bincount(self.sim.step(n, pairs), minlength=self.C.N)
            tot += c
            self.t_ms += self.FRAME_MS
            if self.viewer is not None: self._emit(c, pairs, active)
        self.spikes = np.flatnonzero(tot).astype(np.int32)
        self.n_active = int(len(self.spikes))
        for idx, _ in pairs: tot[idx] = 0                   # always through synapses
        sec = frames * self.FRAME_MS / 1000.0
        return {c["key"]: float(tot[c["idx"]].sum()) / len(c["idx"]) / sec for c in self.chans}

    def _emit(self, c, pairs, active):
        """One frame to the viewer, in the live page's format."""
        spikes = np.flatnonzero(c).astype(np.int32)
        if pairs:
            c = c.copy()
            for idx, _ in pairs: c[idx] = 0
        k = 1000.0 / self.FRAME_MS
        rates = [round(float(c[ch["idx"]].sum()) * k / len(ch["idx"]), 1) for ch in self.chans]
        names = [g.name if isinstance(g, Group) else "input" for g, _ in active]
        self.viewer.frame(rates, spikes, names, self.world_frame() if self.world_frame else None,
                          {g.name: g.idx for g, _ in active if isinstance(g, Group)})


def drive_observation(s, track=None):
    """Kart state (dict with world.Kart.FRAME_KEYS, e.g. world.Kart.state() or a live
    frame zipped with its keys) -> the DriveEnv observation vector. Use it to run a policy
    trained offline on the live page."""
    W = (track or world.Track()).W
    return np.array([s["speed"] / world.Kart.VMAX, s["wheel"], s["offset"] / W, s["heading_error"] / math.pi,
                     s["throttle"], s["brake"], s["push"]], np.float32)


def paired_action(u, max_hz=150.0):
    """Two commands in -1..1 -> the four DriveEnv rates [accelerator, brake, wheel left,
    wheel right]: u[0] speed (+ accelerator, - brake), u[1] steering (+ right, - left).
    Opposite inputs are never driven together, as with the page's keys: they cancel out,
    and both DNg12_e at once run the network away. 150 Hz is the rate of the Drive keys."""
    s, w = float(np.clip(u[0], -1, 1)), float(np.clip(u[1], -1, 1))
    return max_hz * np.array([max(s, 0.0), max(-s, 0.0), max(-w, 0.0), max(w, 0.0)])


def load_policy(path):
    """A driving policy saved as JSON -> function(observation) -> the four DriveEnv rates.
    kind "paired" (examples/drive_es.py): tanh(W @ [obs, 1]) -> paired_action; otherwise
    (examples/drive_hillclimb.py): max_hz x sigmoid(W @ [obs, 1])."""
    with open(path) as f: p = json.load(f)
    W, high = np.array(p["W"], dtype=float), float(p.get("max_hz", 250.0))
    if p.get("kind") == "paired":
        return lambda obs: paired_action(np.tanh(W @ np.append(obs, 1.0)), high)
    return lambda obs: high / (1.0 + np.exp(-(W @ np.append(obs, 1.0))))


def _inputs(brain, specs):
    return [brain.group(s["query"], s.get("field", "type"), s.get("side"), s.get("name") or s.get("label")) for s in specs]


class DriveEnv:
    """Learn to drive: the fly stands in a kart on a circular road (world.Track), its front
    legs on the steering wheel and its hind legs on the pedals (world.CONTROLS). Only its
    motor neurons move the kart.

    action: Poisson rate (Hz, 0..action_high) for each input group. Default inputs are the
      live page's Drive keys (live.KEYSETS["car"]): the descending neurons that move each
      front leg (wheel) and each hind leg (pedals). Pass `inputs=[dict(name, query, field,
      side), ...]` to act on any other neurons, e.g. visual or mechanosensory ones.
    observation: speed, wheel angle, lateral offset, heading error, and the pedal and
      wheel-push levels actually produced by the legs, all roughly in -1..1.
    reward: distance gained along the road, minus a small penalty for leaving the centre
      line; the episode ends when the kart leaves the road (|offset| > half width)."""

    observation_names = ["speed", "wheel", "offset", "heading_error", "throttle", "brake", "push"]

    def __init__(self, inputs=None, step_ms=50, max_steps=1200, max_hz=250.0, params=None, track=None, seed=0,
                 viewer=None):
        self.brain = FlyBrain(params, seed, viewer=_viewer(viewer, "DriveEnv", "kart"))
        self.inputs = _inputs(self.brain, inputs or live.KEYSETS["car"])
        self.action_names = [g.name for g in self.inputs]
        self.action_high = np.full(len(self.inputs), float(max_hz), np.float32)
        self.step_ms, self.max_steps = int(step_ms), int(max_steps)
        self.driver = world.Driver(track)
        self.brain.world_frame = self.driver.kart.frame    # the page draws the kart from the frames
        self.steps, self.episode, self.ret = 0, 0, 0.0

    def reset(self, seed=None):
        self.brain.reset(0 if seed is None else int(seed))
        self.driver.reset()
        self.steps, self.ret = 0, 0.0
        self.episode += 1
        if self.brain.viewer: self.brain.viewer.note(f"{self.brain.viewer.label}: episode {self.episode}")
        return self._obs(), dict(distance=0.0, t_ms=0.0)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), 0.0, self.action_high)
        drive = {g: hz for g, hz in zip(self.inputs, a)}
        gained, rates = 0.0, {}
        for _ in range(max(1, self.step_ms // int(FlyBrain.FRAME_MS))):
            rates = self.brain.step(drive)
            gained += self.driver.frame(rates, FlyBrain.FRAME_MS)
        self.steps += 1
        k = self.driver.kart
        off = abs(k.offset())
        terminated = off > k.track.W
        reward = gained - 0.02 * off / k.track.W - (5.0 if terminated else 0.0)
        truncated = self.steps >= self.max_steps
        self.ret += reward
        if (terminated or truncated) and self.brain.viewer:
            self.brain.viewer.note(f"episode {self.episode}: {'off the road' if terminated else 'end'} after "
                                   f"{k.dist:.1f} along the road, return {self.ret:.2f}")
        info = dict(distance=k.dist, t_ms=self.brain.t_ms, n_active=self.brain.n_active,
                    kart=k.state(), motor=dict(self.driver.S))           # smoothed rates the kart reads
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _obs(self):
        return drive_observation(self.driver.kart.state(), self.driver.kart.track)


class DanceEnv:
    """Learn to dance: the fly hears a metronome through its auditory neurons (JO-B gets a
    burst on every beat, as the live page does with music), and the agent drives input
    groups so that the chosen motor neurons fire on the beat and stay quiet in between.

    action: Poisson rate (Hz) for each input group (default: the Drive and Keyboard inputs,
      i.e. descending and premotor neurons that move legs and steer).
    observation: sin and cos of the beat phase, then the activation (0..1) of each motor
      channel in `motor`.
    reward: per step, the mean motor activation, counted + in the first quarter of each
      beat and - elsewhere."""

    # legs and abdomen: the wings already flick on the beat by themselves (JO-B -> DNp02/06)
    DEFAULT_MOTOR = ["T1L_pro", "T1R_pro", "T3R_trx", "T3L_tif", "abdL", "abdR"]

    def __init__(self, inputs=None, motor=None, bpm=120.0, step_ms=20, max_steps=500, max_hz=250.0,
                 hear=True, params=None, seed=0, viewer=None):
        self.brain = FlyBrain(params, seed, viewer=_viewer(viewer, "DanceEnv", None))
        self.inputs = _inputs(self.brain, inputs or (live.KEYSETS["car"] + live.KEYSETS["walk"]))
        self.action_names = [g.name for g in self.inputs]
        self.action_high = np.full(len(self.inputs), float(max_hz), np.float32)
        self.motor = [m for m in (motor or self.DEFAULT_MOTOR) if m in self.brain.channel_keys]
        self.observation_names = ["beat_sin", "beat_cos"] + self.motor
        self.period_ms, self.step_ms, self.max_steps = 60000.0 / bpm, int(step_ms), int(max_steps)
        self.hear = hear
        self.ear = self.brain.group(live.AUDIO_INPUT["query"], live.AUDIO_INPUT["field"], name="JO-B")
        self.steps, self.level = 0, np.zeros(len(self.motor))

    def reset(self, seed=None):
        self.brain.reset(0 if seed is None else int(seed))
        self.steps, self.level = 0, np.zeros(len(self.motor))
        return self._obs(), dict(t_ms=0.0)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), 0.0, self.action_high)
        reward, A = 0.0, live.AUDIO_INPUT
        frames = max(1, self.step_ms // int(FlyBrain.FRAME_MS))
        for _ in range(frames):
            phase = (self.brain.t_ms % self.period_ms) / self.period_ms
            drive = {g: hz for g, hz in zip(self.inputs, a)}
            if self.hear:
                drive[self.ear] = A["hz"] + (A["hit_hz"] if phase * self.period_ms < A["hit_ms"] else 0.0)
            rates = self.brain.step(drive)
            self.level = np.array([world.act(rates[m], 12.0) for m in self.motor])
            reward += (1.0 if phase < 0.25 else -1.0) * float(self.level.mean()) / frames
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, False, truncated, dict(t_ms=self.brain.t_ms, n_active=self.brain.n_active)

    def _obs(self):
        ph = 2 * math.pi * (self.brain.t_ms % self.period_ms) / self.period_ms
        return np.concatenate([[math.sin(ph), math.cos(ph)], self.level]).astype(np.float32)
