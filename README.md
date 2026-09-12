# flybrain

Simulate the complete connectome of the male fruit fly (*Drosophila melanogaster*,
MaleCNS v1.0: 165,384 neurons, 10.4 M connections, brain **and** ventral nerve cord)
on your own machine, and watch it drive a 3D fly.

- **Connectome dashboard** (`/`) — stimulate any group of neurons (by type, instance,
  class, body ID…), silence others, choose readouts, and run a spiking simulation.
  Results: activity map of every soma, raster, population rates per anatomical group,
  most active neurons and cell types, and a per-neuron inspector (inputs, outputs).
- **3D fly** (`/fly`) — a procedural 3D male fly with inverse-kinematics legs, and two
  ways to drive it:
  - *Dance*: twerks to music (demo beat, audio file or microphone) and stops on silence.
    Pure choreography, no neurons involved.
  - *Brain*: a continuous simulation of the connectome drives the body through its
    descending neurons and motor neurons (legs by muscle, wings, neck, abdomen,
    proboscis). Looming → giant fiber → TTMn → jump; taste → MN9 → proboscis;
    DNp09 → walking; DNa02 → turning…

The interface is in French.

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
*Simuler* (⌘/Ctrl+Enter). Click any neuron (map, raster, tables) to inspect its
connections and to stimulate, read out or silence it.

**3D fly — brain mode.** Choose *Cerveau (connectome)*, *Démarrer la simulation*, then
send stimuli: sensory (looming left/right, taste, sound, cVA odour) or direct
“optogenetic” activation of a descending neuron (giant fiber, DNp09, MDN, DNa02, pIP10,
MN9). The panel shows the live rates of the 64 motor channels, including a legs × muscles
grid. Tick *Le son stimule l'organe de Johnston* to feed the audio source to the brain.

**Command line.** `flysim.py` runs the same model without the browser:

```sh
.venv/bin/python flysim.py types LC        # list cell types matching a regex
.venv/bin/python flysim.py info '^DNp01$'  # details of matching neurons
.venv/bin/python flysim.py run --stim '^LC4$' --readout '^DNp01$' --ms 300
```

## The model, and its limits

- Leaky integrate-and-fire network after Shiu et al., *Nature* 2024: one synapse adds
  0.275 mV, acetylcholine is excitatory, GABA and glutamate inhibitory (other
  transmitters are treated as excitatory), 1.8 ms synaptic delay. A simpler
  instantaneous-current variant (`flysim`) is also available.
- MaleCNS has ~1.6× more synapses per neuron than FlyWire, on which that model was
  tuned: at full weight, activity explodes and never stops. Brain mode therefore
  scales weights by 0.5.
- Even so, some stimuli (sound, taste) push the network into a self-sustained state,
  mostly in the central complex. A safeguard resets the network 1 s after the last
  stimulus if activity persists, and logs it. Short-term synaptic depression is
  available as an option: it prevents that state but also silences the pathways that
  need fast firing (TTMn, MN9).
- In brain mode, the rates come from the connectome; turning them into joint angles,
  the tripod walking gait and the jump trajectory are hand-written (the network has no
  rhythm generator). The dance mode uses no neurons at all.

## Project layout

| Path | Role |
|---|---|
| `setup.sh`, `run.sh` | installation and start scripts |
| `convert.py` | MaleCNS feather files → `malecns_graph.npz` + `neurons.csv` |
| `engine.py` | connectome loading, LIF simulation (`Stepper`, `simulate`) |
| `live.py` | continuous simulation and motor channels for the 3D fly |
| `dashboard.py` | local HTTP server (standard library) and JSON/binary API |
| `flysim.py` | command-line simulator |
| `static/` | dashboard (`index.html`, `app.js`, `results.js`, `charts.js`) |
| `static/fly.html`, `static/fly/` | 3D fly: model, audio analysis, dance, brain link |
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
