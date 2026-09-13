# flybrain API

Two ways to use the fly from code:

- **Python** (`flyenv.py`): the simulation runs in your process, synchronously, as fast
  as the CPU allows — for training and batch experiments. No dependency beyond numpy.
- **HTTP** (the server started by `./run.sh`): the live simulation behind the 3D page —
  to watch a policy, or to control the fly from any language.

Both share the same model (`engine.py`), the same readout channels (`live.channels`), the
same kart (`world.py`) and the same rule.

## The rule: always through synapses

You act on the fly only by driving groups of neurons with Poisson spikes (a rate in Hz).
You observe only what the network makes of it: every readout ignores the spikes of the
neurons you drive. So a command always crosses at least one synapse before it moves
anything — driving DNp09 and reading DNp09 back is impossible by construction.

The body of the fly (and of the kart) is the only hand-written part: the tripod gait, the
jump trajectory and the mapping from motor neuron rates to joint angles or kart controls
(`world.CONTROLS`). The network has no rhythm generator.

## Python

```python
import numpy as np, flyenv

brain = flyenv.FlyBrain()                  # loads the connectome once per process (~1 s)
lal = brain.group("^LAL018$", side="L")    # regex on a field: type, instance, class, superclass, any, body, idx
rates = brain.step({lal: 150}, ms=300)     # {channel key: mean Hz over the 300 ms}
rates["turnL"]                             # DNa01/02 left, one synapse downstream: ~50 Hz
brain.n_active, brain.spikes               # neurons that fired during the step
brain.channel_keys                         # every readout channel (see below)
brain.reset(seed=0)                        # network at rest, clock at 0
```

`FlyBrain(params)` accepts the live parameters: `dt` (ms, default 0.2), `w_scale` (0.5),
`std_u`, `std_tau` (short-term depression), `model` (`"shiu"` or `"simple"`).

### DriveEnv — learn to drive

The fly stands in a kart on a circular road, front legs on the steering wheel, hind legs
on the pedals. Only its motor neurons move the kart (`world.CONTROLS`):

| Control | Motor channel | Meaning |
|---|---|---|
| wheel, turn right | `T1L_pro` | left front leg pushes the rim (coxa promotors) |
| wheel, turn left | `T1R_pro` | right front leg pushes the rim |
| accelerator | `T3R_trx` | right hind leg extends (trochanter extensors) |
| brake | `T3L_tif` | left hind leg flexes (tibia flexors) and pulls the brake lever |

```python
env = flyenv.DriveEnv(step_ms=50, max_steps=1200, max_hz=250)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.array([150, 0, 0, 0]))
env.action_names        # the input groups (default: the Drive keys of the 3D page)
env.action_high         # max rate per input (Hz)
env.observation_names   # speed, wheel, offset, heading_error, throttle, brake, push
info["kart"], info["distance"], info["motor"], info["n_active"]
```

- **Action**: one Poisson rate (Hz) per input group. Default inputs, from
  `live.KEYSETS["car"]`: DNg16 right (accelerator), DNpe008 left (brake), DNg12_e right
  (turn left), DNg12_e left (turn right) — the descending neurons that move each leg most
  specifically. Pass `inputs=[dict(name=..., query=..., field=..., side=...), ...]` to act
  on any other neurons, e.g. sensory ones.
- **Observation** (≈ −1..1): speed / max speed, wheel angle, lateral offset / half width,
  heading error / π, and the throttle, brake and wheel-push levels the legs produce.
- **Reward**: distance gained along the road per step, minus 0.02 × |offset| / half
  width; −5 and `terminated` when the kart leaves the road.
- Speed: one 50 ms step takes about 40 ms of CPU on a laptop.

### DanceEnv — learn to dance

The fly hears a metronome through its auditory neurons (JO-B gets a burst on every beat,
as the 3D page does with music). The agent drives input groups so that chosen motor
channels fire on the beat and stay quiet in between.

```python
env = flyenv.DanceEnv(bpm=120, motor=["T1L_pro", "T1R_pro", "abdL", "abdR"], step_ms=20)
obs, info = env.reset()                    # obs: sin/cos of the beat phase + motor activations
obs, reward, terminated, truncated, info = env.step(np.zeros(len(env.action_names)))
```

Reward per step: the mean activation of the `motor` channels, counted + in the first
quarter of each beat and − elsewhere. Default inputs: the Drive and Keyboard keys.

### Channels

`brain.channel_keys` (and `channels` in the HTTP start response) — rates in Hz:

| Keys | Neurons |
|---|---|
| `esc` | DNp02/06/11, escape descending neurons (downstream of hearing and looming) |
| `fwd`, `back`, `turnL`, `turnR` | DNp09, MDN, DNa01/02 left and right — the locomotion commands |
| `gf`, `ttmn` | giant fiber DNp01, jump motor neuron TTMn |
| `mn9`, `pm` | proboscis motor neurons |
| `wpowL/R`, `wstrL/R`, `halL/R` | flight power, wing steering and haltere motor neurons |
| `neckL/R`, `antL/R`, `abdL/R` | neck, antenna and abdomen motor neurons |
| `T1L_pro` … `T3R_tal` | leg motor neurons: leg T1/T2/T3, side L/R, muscle group `pro` `rem` (coxa promotors / remotors), `trx` `trf` (trochanter extensors / flexors), `tix` `tif` (tibia extensors / flexors), `tad` `tal` (tarsus depressors / levators) |

### Watching a training run in 3D

Pass `viewer=True` (or a server URL) to `DriveEnv`, `DanceEnv` or `FlyBrain`, with the
server running (`./run.sh`):

```python
env = flyenv.DriveEnv(viewer=True)          # viewer="http://host:8765" for another server
env.brain.viewer.note("new best policy")    # a line in the page's event log
flyenv.Viewer(realtime=True)                # slows the simulation to real time, to watch calmly
```

Every simulated 10 ms frame is streamed to the server (`/api/view/start`, `/api/view/push`)
from a background thread — it never slows training, and frames are dropped if the server is
unreachable. The server makes it the current session, and the 3D page (`/fly`) follows the
current session by itself: it switches to the kart when there is one, shows the fly, its
3D brain (the driven groups in amber), the gauges and the episode notes, and turns its
keys off since the inputs belong to your process. *Start simulation* on the page takes the
server back; streaming then stops until `viewer.reconnect()`.

The page follows any session the same way — also one that a script starts or drives over
HTTP.

### Running a trained policy on the live page

`examples/drive_hillclimb.py` trains a linear policy offline and saves it as JSON;
`examples/live_policy.py` joins the page's session over HTTP and drives the kart with it
(open `/fly`, choose Drive, run the script). `flyenv.drive_observation(state)` turns a
live kart frame into the same observation vector as `DriveEnv`.

## HTTP — the live simulation

Base URL `http://127.0.0.1:8765`. POST bodies are JSON. One session at a time: starting
a new one replaces the previous one, and a session stops by itself when nobody has read
its frames for 20 s.

| Method and path | Body | Effect / answer |
|---|---|---|
| `POST /api/live/start` | `{params: {dt, w_scale, std_u, std_tau, quench_ms, world}}` | starts a session; answers `id`, `channels`, `events`, `keysets`, `reach`, `controls`, `track`, `kart_keys`, `audio_idx` |
| `GET /api/live/session` | | `{id, alive, kind, label, world}` of the current session (`kind`: `sim` or `remote`) |
| `GET /api/live/info?id=` | | the start information of the current session, with `groups` (inputs named by scripts) |
| `POST /api/view/start` | `{label, world}` | another process will stream frames: a `remote` session becomes current; answers its `id` |
| `POST /api/view/push` | `{id, frames, groups}` | frames in the format below, and new input groups `{name: [indices]}` |
| `GET /api/live/frames?id=&since=` | | frames since index `since` (up to 300), and `next` |
| `POST /api/live/stim` | `{id, name, query, field, side, hz, ms}` | drives any group (`ms` omitted: until `hz` 0); answers `{name, n}` |
| `POST /api/live/event` | `{id, key}` | a stimulus button (`loomL`, `loomR`, `loom`, `sugar`, `sound`, `wind`, `cva`, `gf`, `pip10`) |
| `POST /api/live/drive` | `{id, keys: [...], keyset}` | held keys of a key set (`walk` or `car`); resend every ~100 ms, inputs expire after 300 ms |
| `POST /api/live/reach` | `{id, side}` | the switchboard's front-leg burst (DNg12_e, `L` or `R`) |
| `POST /api/live/audio` | `{id, level, hit}` | sound → JO-B (`level` 0..1 background, `hit` a burst) |
| `POST /api/live/world` | `{id, world: "kart" or null, reset}` | turns the kart on or off, or puts it back on the road |
| `POST /api/live/stop` | `{id}` | stops the session |

A frame is `[t_ms, rates, n_spikes, n_active, inputs, note, brain, kart]`:

- `rates`: one rate (Hz) per channel, in the order of `channels`, over the 10 ms frame,
  the driven neurons masked out;
- `inputs`: names of the active inputs (`ev:…`, `key:walk:up`, `api:…`, `audio`, …);
- `note`: a safeguard message, or null;
- `brain`: every 4th frame, base64 of little-endian int32 neuron indices that fired
  during the last 40 ms (at most 4,000), else null;
- `kart`: with the kart world, the values named by `kart_keys` (`x z yaw speed wheel
  throttle brake offset distance push heading_error`), else null.

Safeguards: with `quench_ms` > 0 the network is reset when activity persists that long
after the last input; while only keys drive it, it is also reset if more than 200
neurons fire per 10 ms on average over 300 ms (the runaway state).

### Batch simulations (dashboard)

| Method and path | Effect |
|---|---|
| `GET /api/meta` | binary pack: `[u32 header size][JSON header][arrays]` — positions (µm), groups, types, presets |
| `GET /api/search?q=&field=` | count and types of the neurons matching a pattern |
| `GET /api/neuron/<idx>` | a neuron's annotations, inputs and outputs |
| `POST /api/run` | `{params, stims, silence, readouts}` → `{job}`; then `GET /api/job/<id>` (progress), `GET /api/job/<id>/result` (binary pack), `POST /api/job/<id>/cancel` |
