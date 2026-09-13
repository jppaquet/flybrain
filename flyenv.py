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
import math
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


class FlyBrain:
    """Leaky integrate-and-fire simulation of the whole CNS, advanced in 10 ms frames.

    Channels are the live page's readouts (live.channels): descending neurons for
    locomotion, motor neurons of every leg muscle group, wings, neck, antennae, abdomen,
    proboscis. `channel_keys` lists them."""
    FRAME_MS = live.FRAME_MS

    def __init__(self, params=None, seed=0):
        self.C = connectome()
        self.q = engine.resolve_params(dict(LIVE_PARAMS, **(params or {})))
        self.chans = live.channels(self.C)
        self.channel_keys = [c["key"] for c in self.chans]
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
        pairs = [(g.idx if isinstance(g, Group) else np.asarray(g, np.int32), float(hz))
                 for g, hz in (drive or {}).items() if hz > 0]
        tot = np.zeros(self.C.N, np.int64)
        for _ in range(frames):
            tot += np.bincount(self.sim.step(n, pairs), minlength=self.C.N)
        self.t_ms += frames * self.FRAME_MS
        self.spikes = np.flatnonzero(tot).astype(np.int32)
        self.n_active = int(len(self.spikes))
        for idx, _ in pairs: tot[idx] = 0                   # always through synapses
        sec = frames * self.FRAME_MS / 1000.0
        return {c["key"]: float(tot[c["idx"]].sum()) / len(c["idx"]) / sec for c in self.chans}


def drive_observation(s, track=None):
    """Kart state (dict with world.Kart.FRAME_KEYS, e.g. world.Kart.state() or a live
    frame zipped with its keys) -> the DriveEnv observation vector. Use it to run a policy
    trained offline on the live page."""
    W = (track or world.Track()).W
    return np.array([s["speed"] / world.Kart.VMAX, s["wheel"], s["offset"] / W, s["heading_error"] / math.pi,
                     s["throttle"], s["brake"], s["push"]], np.float32)


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

    def __init__(self, inputs=None, step_ms=50, max_steps=1200, max_hz=250.0, params=None, track=None, seed=0):
        self.brain = FlyBrain(params, seed)
        self.inputs = _inputs(self.brain, inputs or live.KEYSETS["car"])
        self.action_names = [g.name for g in self.inputs]
        self.action_high = np.full(len(self.inputs), float(max_hz), np.float32)
        self.step_ms, self.max_steps = int(step_ms), int(max_steps)
        self.driver = world.Driver(track)
        self.steps = 0

    def reset(self, seed=None):
        self.brain.reset(0 if seed is None else int(seed))
        self.driver.reset()
        self.steps = 0
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
                 hear=True, params=None, seed=0):
        self.brain = FlyBrain(params, seed)
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
