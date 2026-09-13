# flybrain

Simulate the complete connectome of the male fruit fly (*Drosophila melanogaster*,
MaleCNS v1.0: 165,384 neurons, 10.4 M connections, brain **and** ventral nerve cord)
on your own machine, and watch it drive a 3D fly.

- **Connectome dashboard** (`/`) — stimulate any group of neurons (by type, instance,
  class, body ID…), silence others, choose readouts, and run a spiking simulation.
  Results: activity map of every soma, raster, population rates per anatomical group,
  most active neurons and cell types, and a per-neuron inspector (inputs, outputs).
- **3D fly** (`/fly`) — a procedural 3D male fly with inverse-kinematics legs, driven by
  a continuous simulation of the connectome in four modes:
  - *Brain*: sensory stimuli (looming, taste, sound, wind, odour) and the music you play,
    through Johnston's organ, move the body through its descending and motor neurons.
  - *Keyboard*: the arrow keys steer the fly through neurons presynaptic to its walking
    commands; Space shows a looming threat and the fly jumps.
  - *Switches*: the fly, tethered in front of a console, flips switches with its front
    legs when you press the number keys.
  - *Drive*: the fly drives a kart — front legs on the steering wheel, hind legs on the
    accelerator and brake.

  Above the fly, a 3D point cloud of all 165,384 neurons lights up as they fire.
- **Machine learning** — `flyenv.py` turns the fly into an end user for your agents:
  Gymnasium-style environments to learn to drive (`DriveEnv`) or to dance (`DanceEnv`),
  a low-level `FlyBrain`, and an HTTP API to run a trained policy on the live 3D page.
  See [API.md](API.md).

**Everything goes through synapses.** Keys, stimuli and agents only drive neurons with
spikes; the body only reads neurons downstream of them — the spikes of driven neurons are
never counted. The hand-written parts are the body mechanics only: the tripod gait, the
jump trajectory, the path of a leg toward a switch, and the mapping from motor neurons to
joint angles or kart controls.

## Requirements

- macOS or Linux (Windows: use WSL), with `bash` and `curl`
- Python **3.12 or newer** (`brew install python@3.13` on macOS)
- ~1.7 GB of free disk space (1.1 GB of connectome data, plus the converted graph)
- ~4 GB of RAM for the one-time conversion, ~0.5–1.5 GB while running
- A recent browser with WebGL (Chrome, Firefox, Safari)

## Quick start

```sh
git clone https://github.com/jppaquet/flybrain.git
cd flybrain
./setup.sh      # downloads the data, creates .venv, converts the connectome
./run.sh        # starts the local server
```

Then open <http://127.0.0.1:8765> (dashboard) or <http://127.0.0.1:8765/fly> (3D fly).
The first start takes a few extra seconds to estimate the position of neurons that have
no annotated soma; the result is cached in `meta_cache.npz`.

`./run.sh --port 9000` uses another port. Stop the server with Ctrl+C.

## What `setup.sh` does

It can be re-run at any time: finished steps are skipped.

1. Finds a Python ≥ 3.12 (`PYTHON=/path/to/python3 ./setup.sh` to choose one).
2. Downloads the three MaleCNS v1.0 “flat connectome” files from the public FlyEM bucket
   and checks each one against the MD5 published by the server. Interrupted downloads
   resume; a corrupted file is downloaded again.
3. Creates `.venv` and installs the pinned dependencies from `requirements.txt`
   (numpy, pyarrow — exact versions, hashes verified by pip, no transitive dependencies).
4. Runs `convert.py`, which keeps connections of at least 3 synapses between identified
   neurons and writes `malecns_graph.npz` (CSR graph) and `neurons.csv`.
   `MIN_WEIGHT=5 ./setup.sh` rebuilds the graph with another threshold.

The data files are not stored in git (see `.gitignore`): the connectome weights alone
are 1 GB, far above GitHub's 100 MB file limit. `setup.sh` downloads them, and
`malecns_graph.npz` / `neurons.csv` are generated from them — a fresh clone needs no
file that is not in the repository.

## Using it

**Dashboard.** Pick neurons to stimulate (regex on a field, or presets such as LC4,
ORN DA1, Johnston's organ…), set a Poisson rate and a time window, add readouts, then
*Simulate* (⌘/Ctrl+Enter). Click any neuron (map, raster, tables) to inspect its
connections and to stimulate, read out or silence it.

**3D fly — Brain.** *Start simulation*, then send stimuli: sensory (looming left, right
or on both eyes, taste, sound, wind, cVA odour) or the direct activation of a neuron
whose effect is read downstream (giant fiber → TTMn → jump; pIP10 → wing motor neurons).
The panel shows the live rates of the motor channels, including a legs × muscles grid.
Tick *The sound drives Johnston's organ* and play music (demo beat, audio file or
microphone): the auditory neurons JO-B receive a background drive that follows the
volume, plus a burst on every kick drum, and the fly flicks its wings on the beat.

The brain floating above the fly shows every neuron at its soma position, in the fly's
own orientation (brain over the head, ventral nerve cord over the thorax), coloured by
anatomical group. A neuron flashes white when it fires, and the neurons driven by a
stimulus glow amber. Untick *Show the brain in 3D above the fly* to hide it.

**3D fly — Keyboard.** The body walks, backs up and turns according to the descending
neurons DNp09, MDN and DNa01/02, so the keys never drive those: each drives their
strongest clean presynaptic partner, found by screening their inputs — ↑ ICL012m (DNp09
reaches ~32 Hz), ↓ DNpe023 (MDN ~63 Hz), ← → LAL018 of that side (DNa01/02 ~45 Hz; the
lateral accessory lobe is the fly's steering centre), all at 150 Hz with at most a few
hundred neurons active. Each key lights up with the firing of the descending neurons it
recruits. Space shows a looming threat to both eyes, which fires the jump motor neuron
TTMn within ~15 ms. The chase camera stays behind the fly; pick a view to leave it.

**3D fly — Switches.** The fly is tethered in front of a console (4 to 10 switches,
seen from the front), and keys 1–9, 0 flip the switches. The switches on the fly's right
are pressed by its right front leg, those on its left by its left one. A key sends a
400 ms burst to DNg12_e on that side — in a screen of all 472 descending neuron types,
the one that moves a front leg most specifically: alone it recruits about 20 neurons,
mostly that leg's coxa promotors. The leg reaches toward the switch as far as those
motor neurons fire, and the switch flips only if they reach 16 Hz (they peak at
30–60 Hz); the log gives their peak rate. Keys pressed during a reach wait their turn:
one leg at a time, because driving both DNg12_e together tips the network into its
self-sustained state.

**3D fly — Drive.** The fly stands on a kart on a circular road, its front legs on the
steering wheel and its hind legs on the pedals, and only its motor neurons move the
kart: the front legs' coxa promotors push the wheel rim, the right hind leg's extensors
press the accelerator, the left hind leg's flexors pull the brake lever (no descending
neuron extends the left hind leg cleanly in this connectome; one flexes it). Each key
drives the descending neuron that moves one leg most specifically (screens of all DN
types): ↑ DNg16 right, ↓ DNpe008 left, ← DNg12_e right, → DNg12_e left. R puts the kart
back on the road. The kart's physics runs on the server (`world.py`), so the same kart
is available for learning (`flyenv.DriveEnv`).

**Command line.** `flysim.py` runs the same model without the browser:

```sh
.venv/bin/python flysim.py types LC        # list cell types matching a regex
.venv/bin/python flysim.py info '^DNp01$'  # details of matching neurons
.venv/bin/python flysim.py run --stim '^LC4$' --readout '^DNp01$' --ms 300
```

## Machine learning

```python
import numpy as np, flyenv
env = flyenv.DriveEnv()                              # or flyenv.DanceEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.array([150, 0, 0, 0]))
```

- **Actions** are Poisson rates for groups of neurons (by default the Drive keys; any
  neurons with `inputs=[...]`); **observations** are the kart state or the beat phase and
  the motor activations the legs actually produce; the simulation steps synchronously,
  about as fast as real time on a laptop.
- `examples/drive_hillclimb.py` learns a linear driving policy by hill climbing and saves
  it; `examples/live_policy.py` then drives the kart of the live 3D page with it, over
  HTTP (open `/fly`, choose Drive, run the script).
- The HTTP API drives any group of neurons in the live session (`POST /api/live/stim`)
  and streams the channel rates, the spikes and the kart state. Details in
  [API.md](API.md).

## Modes and options

### The four modes of the 3D fly

| Mode | Input | What the body reads |
|---|---|---|
| **Brain** | stimulus buttons; optionally the sound (→ JO-B) | descending and motor neurons downstream of the stimulated ones |
| **Keyboard** | arrow keys / WASD, Space | DNp09, MDN, DNa01/02 — one synapse downstream of the neurons the keys drive — and the motor neurons they recruit |
| **Switches** | keys 1–9, 0 | the front-leg motor neurons recruited by DNg12_e |
| **Drive** | arrow keys / WASD, R | the leg motor neurons recruited by the four leg descending neurons |

In every mode the neurons decide whether the fly walks, turns, jumps, reaches or drives,
and how strongly; the spikes of the neurons that a key or a stimulus drives are never
read. The 3D brain shows the live spikes.

### Controls in every mode

| Control | Default | Effect |
|---|---|---|
| *Rear ¾ / Side / Front / Top*, ↻ | Rear ¾ | camera presets and auto-rotate; drag to orbit, scroll to zoom |
| *▶ Demo beat / ♫ File… / 🎤 Mic / ■ Stop* | no source | sound source; an audio file can also be dropped on the 3D view; the fly hears it only with *The sound drives Johnston's organ* |
| *Demo: one silent bar in 8* | on | the demo beat goes silent one bar in eight |
| *Volume* | 80 % | playback volume |
| *Silence threshold* | −50 dB | below it, the sound counts as silence (no drive to JO-B) |
| *Stop after* | 350 ms | how long the silence must last |
| *Body* sliders, *Reset* | rest pose | height, pitch, roll, heading, leg spread, head yaw and pitch, abdomen lift and side, each wing, proboscis, antennae; every mode adds its movement to this pose |
| Browser console | | `fly.pose`, `fly.set("abdPitch", 0.4)` |

### The live simulation

| Control | Default | Effect |
|---|---|---|
| *Start simulation / Stop* | | Keyboard, Switches and Drive start it by themselves |
| *Reset network* | | restarts from a network at rest, same parameters |
| *Recenter* | | brings the fly back to the centre |
| *Weight scale* | 0.5 | multiplies every synaptic weight; at 1 the activity runs away |
| *dt (ms)* | 0.2 | simulation time step |
| *Depression U* | 0 | short-term synaptic depression (0 = off): prevents the runaway state but cuts almost all transmission |
| *Recovery τ (ms)* | 200 | recovery time of that depression |
| *Safeguard* | on | resets the network if activity persists 1 s after the last stimulus |
| *Show the brain in 3D above the fly* | on | the point cloud of all 165,384 neurons |
| *The sound drives Johnston's organ* | off | background drive on JO-B that follows the volume, plus a burst on every kick drum |

Stimulus buttons (Brain mode):

| Button | Neurons driven | Drive | Read downstream |
|---|---|---|---|
| *Looming, left / right / both eyes* | LC4 | 150 Hz for 300 ms | giant fiber, TTMn (jump), escape DNs |
| *Taste (LB3c + taste pegs)* | gustatory receptor neurons that reach MN9 | 150 Hz for 800 ms | MN9, proboscis |
| *Sound (JO-B)* | auditory neurons of Johnston's organ | 200 Hz for 500 ms | escape DNs, wing motor neurons |
| *Wind (JO-C/E)* | wind and gravity neurons of Johnston's organ | 150 Hz for 500 ms | antennae, front legs |
| *cVA odour (ORN DA1)* | pheromone olfactory receptor neurons | 150 Hz for 800 ms | |
| *Giant fiber* | DNp01 | 200 Hz for 100 ms | TTMn (jump) |
| *pIP10: song* | pIP10 | 150 Hz for 1.5 s | wing motor neurons |

### Keyboard

| Key | Neurons driven | Read (one synapse downstream) | Effect |
|---|---|---|---|
| ↑ or W | ICL012m, 150 Hz | DNp09 ~32 Hz | walk forward |
| ↓ or S | DNpe023, 150 Hz | MDN ~63 Hz | walk backward |
| ← or A, → or D | LAL018 of that side, 150 Hz | DNa01/02 of that side ~45 Hz | turn |
| Space | LC4 on both eyes, 150 Hz for 300 ms | TTMn | jump |

Opposite keys cancel out. *Chase camera* (on): the camera stays behind the fly; choosing
a camera preset turns it off.

### Switches

| Control | Default | Effect |
|---|---|---|
| Keys 1–9, 0 (numeric keypad too) | | flip switch 1–10; switches on the fly's right are pressed by its right front leg, those on its left by its left one |
| *Switches* | 8 | 4 to 10 switches on the console |

Each key sends 250 Hz for 400 ms to DNg12_e on that side; the switch flips when that
leg's coxa promotors reach 16 Hz. Up to 4 keys wait in a queue, one leg at a time.

### Drive

| Key | Neurons driven (150 Hz) | Motor neurons read | Control |
|---|---|---|---|
| ↑ or W | DNg16 right | right hind leg trochanter extensors (~17 Hz) | accelerator |
| ↓ or S | DNpe008 left | left hind leg tibia flexors (~12 Hz) | brake |
| ← or A | DNg12_e right | right front leg coxa promotors (~16–21 Hz) | wheel to the left |
| → or D | DNg12_e left | left front leg coxa promotors | wheel to the right |
| R | | | back on the road |

A spring recentres the wheel: to follow the curve, the fly must keep pushing. The panel
shows the speed, the distance along the road, the offset from its centre, and the wheel,
accelerator and brake as the legs actually move them.

### Connectome dashboard

| Control | Default | Effect |
|---|---|---|
| *Stimulation* groups | LC4 at 150 Hz | regex on a field (type, instance, superclass, class, any, bodyId), Poisson frequency, start and end (ms); presets |
| *Readouts* | giant fiber DNp01, descending neurons | up to 4 groups plotted as population rates; presets |
| *Silencing* | none | neurons removed from the network |
| *Dynamics* | Shiu et al. 2024 | or *flysim* (instantaneous current, no synaptic filter or delay) |
| *Duration*, *Trials*, *Seed* | 500 ms, 1, 0 | 10–5,000 ms, 1–10 trials |
| *dt*, *τ mem.*, *τ syn.*, *Threshold*, *Refract.*, *Delay*, *mV / synapse*, *Poisson weight* | 0.1 ms, 20 ms, 5 ms, 7 mV, 2.2 ms, 1.8 ms, 0.275 mV, 250 | model parameters (flysim: dt 0.2 ms, τ mem. 5 ms) |
| *Weight scale* | 1 | multiplies every synaptic weight |
| *Simulate* | ⌘/Ctrl + Enter | runs the simulation; *Cancel* stops it |

Results: activity map (dorsal, frontal or lateral view; play or scrub through time, or Σ
for the whole run; isolate an anatomical group; hover and click a neuron), readout and
group charts (*Table* shows the numbers), raster, most active neurons and cell types
(regex filter, click a column to sort), and a neuron inspector (*Stimulate*, *Readout*,
*Silence*, *Whole type*, top inputs and outputs).

## The model, and its limits

- Leaky integrate-and-fire network after Shiu et al., *Nature* 2024: one synapse adds
  0.275 mV, acetylcholine is excitatory, GABA and glutamate inhibitory (other
  transmitters are treated as excitatory), 1.8 ms synaptic delay. A simpler
  instantaneous-current variant (`flysim`) is also available.
- MaleCNS has ~1.6× more synapses per neuron than FlyWire, on which that model was
  tuned: at full weight, activity explodes and never stops. The live simulation
  therefore scales weights by 0.5.
- Always through synapses: the readouts ignore the spikes of every driven neuron, so
  driving a descending neuron and reading it back is impossible. The input neurons of
  each mode were chosen by screening the connectome for the most specific clean path
  to what the body reads.
- Sound follows the real auditory route (JO-A/B hear vibrations, JO-C/E sense wind and
  gravity). In this connectome JO-B reaches the giant fiber, the escape descending
  neurons DNp02/06/11 and the wing motor neurons, but no path reaches the walking
  descending neurons: music makes the fly move its wings and antennae, not walk.
- Some stimuli (wind, taste) push the network into a self-sustained state, mostly in the
  central complex. A safeguard resets the network 1 s after the last stimulus if
  activity persists, and logs it; while only keys drive the network, it is also reset as
  soon as it runs away. Short-term synaptic depression prevents that state but blocks
  almost all transmission downstream.
- The network has no rhythm generator: the neurons decide to walk, back up, turn, jump,
  reach or press, and how strongly; the tripod gait, the jump trajectory, the path of a
  leg toward its target and the kart's physics are hand-written.

## Project layout

| Path | Role |
|---|---|
| `setup.sh`, `run.sh` | installation and start scripts |
| `convert.py` | MaleCNS feather files → `malecns_graph.npz` + `neurons.csv` |
| `engine.py` | connectome loading, LIF simulation (`Stepper`, `simulate`) |
| `live.py` | continuous simulation, readout channels, key sets and inputs of the 3D fly |
| `world.py` | the kart and its road, driven by the leg motor neurons |
| `flyenv.py` | machine learning API: `FlyBrain`, `DriveEnv`, `DanceEnv` |
| `examples/` | learning to drive offline, then driving the live page with the policy |
| `dashboard.py` | local HTTP server (standard library) and JSON/binary API ([API.md](API.md)) |
| `flysim.py` | command-line simulator |
| `static/` | dashboard (`index.html`, `app.js`, `results.js`, `charts.js`) |
| `static/fly.html`, `static/fly/` | 3D fly: model, brain link, 3D brain, switchboard, kart, audio analysis |
| `static/vendor/three/` | three.js 0.185.1, vendored (see `VERSION`) |

## Troubleshooting

- **`Python >= 3.12 not found`** — install a recent Python or set `PYTHON=`.
- **`Address already in use`** — another server runs on 8765: `./run.sh --port 9000`.
- **Download interrupted** — run `./setup.sh` again; it resumes and verifies.
- **Microphone** — browsers only allow it on `localhost`/`127.0.0.1` or HTTPS.

## Credits

- Connectome: MaleCNS v1.0, Janelia FlyEM and Google Research (CC-BY), downloaded from
  `storage.googleapis.com/flyem-male-cns/v1.0`.
- Neuron model: Shiu, P. K. et al. *A Drosophila computational brain model reveals
  sensorimotor processing.* Nature (2024).
- 3D rendering: [three.js](https://threejs.org) (MIT licence), vendored in `static/vendor/three`.

## Licence

The code in this repository is released under the [MIT licence](LICENSE).
three.js keeps its own MIT licence (`static/vendor/three/LICENSE`). The MaleCNS
connectome data is not part of this repository: `setup.sh` downloads it from FlyEM, and
it remains under its CC-BY licence — cite the MaleCNS paper and Shiu et al. (2024) if you
publish results obtained with it.
