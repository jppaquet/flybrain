/* main.js - three.js scene, animation loop and control panel of the fly. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createFly, REST } from "./model.js";
import { AudioEngine } from "./audio.js";
import { Dancer } from "./dance.js";
import { BrainLink, BodyController } from "./neural.js";
import { Brain3D } from "./brain3d.js";

const $ = s => document.querySelector(s);
const stage = $("#stage"), canvas = $("#gl");
const status = (t, bad) => { $("#fstatus").textContent = t; $("#fstatus").style.color = bad ? "var(--critical)" : ""; };

/* ------------------------------------------------------------------ scene */
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b0d);
scene.fog = new THREE.Fog(0x0b0b0d, 6, 16);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(32, 1, 0.02, 100);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 0.42, -0.1);
controls.minDistance = 1.2; controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.49;

scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x201810, 0.35));
const key = new THREE.DirectionalLight(0xfff1dc, 2.4);
key.position.set(3, 6, 2.5); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 1, far: 15 });
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.01; key.shadow.radius = 3;
scene.add(key);
const rim = new THREE.DirectionalLight(0x7fb0ff, 1.6); rim.position.set(-3, 2.5, -4); scene.add(rim);
const spot = new THREE.SpotLight(0xffffff, 0, 14, 0.45, 0.6, 1.2);
spot.position.set(0, 6, 0); scene.add(spot, spot.target);

// dance floor: glossy ground + additive light tiles
const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 64),
  new THREE.MeshStandardMaterial({ color: 0x0e0e10, roughness: 0.7, metalness: 0.05, envMapIntensity: 0.12 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const G = 11, S = 0.62;
const tiles = new THREE.InstancedMesh(new THREE.PlaneGeometry(S * 0.9, S * 0.9),
  new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), G * G);
const TILE_COLORS = [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x9085e9].map(c => new THREE.Color(c));
const tileLevel = new Float32Array(G * G), tileHue = new Uint8Array(G * G), tileFall = new Float32Array(G * G);
{
  const d = new THREE.Object3D(), black = new THREE.Color(0);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const k = i * G + j, x = (i - (G - 1) / 2) * S, z = (j - (G - 1) / 2) * S;
    d.position.set(x, 0.002, z); d.rotation.x = -Math.PI / 2; d.updateMatrix();
    tiles.setMatrixAt(k, d.matrix); tiles.setColorAt(k, black);
    tileFall[k] = Math.exp(-((x * x + z * z) / 9));
  }
  scene.add(tiles);
}
function lightTiles(n) {
  const mode = Math.floor(n / 8) % 3, hue = n % TILE_COLORS.length;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const di = i - (G - 1) / 2, dj = j - (G - 1) / 2;
    const on = mode === 0 ? (i + j + n) % 2 === 0
             : mode === 1 ? Math.round(Math.hypot(di, dj)) % 3 === n % 3
             : Math.random() < 0.28;
    if (on) { tileLevel[i * G + j] = 1; tileHue[i * G + j] = hue; }
  }
}
const _c = new THREE.Color();
function updateTiles(dt) {
  for (let k = 0; k < G * G; k++) {
    tileLevel[k] *= Math.exp(-dt * 3.2);
    tiles.setColorAt(k, _c.copy(TILE_COLORS[tileHue[k]]).multiplyScalar(tileLevel[k] * tileFall[k] * 0.32));
  }
  tiles.instanceColor.needsUpdate = true;
}

const fly = createFly();
scene.add(fly.root);
const grid = new THREE.GridHelper(80, 160, 0x2c2c31, 0x1d1d21);   // ground reference to see the walking
grid.position.y = 0.001; scene.add(grid);

const brain = new BrainLink(), body = new BodyController(fly), brain3d = new Brain3D(scene);
brain.onNote = txt => body.note(txt);
let mode = "dance", lastAudio = 0, lastKick = -10, stimNames = "", liftLeft = 0;
const lastRoot = new THREE.Vector3();
function followCamera() {
  const p = fly.root.position, d = p.clone().sub(lastRoot);
  if (d.lengthSq()) { controls.target.add(d); camera.position.add(d); }
  lastRoot.copy(p);
}

/* ---------------------------------------------------------------- camera */
const CAMS = { rear: [3.6, 2.3, -5.2], side: [6.2, 1.5, 0.1], front: [1.6, 1.7, 6.0], top: [0.01, 7.2, 0.4] };
let camTween = null;
function setCam(name) {
  const o = new THREE.Vector3(controls.target.x, 0, controls.target.z + 0.1);   // relative to the fly
  camTween = { from: camera.position.clone(), to: new THREE.Vector3(...CAMS[name]).add(o), t: 0 };
  for (const b of $("#cams").querySelectorAll("[data-cam]")) b.classList.toggle("on", b.dataset.cam === name);
}
camera.position.set(...CAMS.rear);
$("#cams").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.id === "autorot") { controls.autoRotate = !controls.autoRotate; b.classList.toggle("on", controls.autoRotate); }
  else setCam(b.dataset.cam);
});
function resize() {
  const r = stage.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize();

/* ------------------------------------------------------------ manual pose */
const deg = v => `${Math.round(v * 180 / Math.PI)}°`, pct = v => `${Math.round(v * 100)} %`;
const JOINTS = [
  ["height", "Height", 0.35, 0.8, v => `${v.toFixed(2)} mm`],
  ["pitch", "Pitch", -0.5, 0.5, deg], ["roll", "Roll", -0.4, 0.4, deg],
  ["yaw", "Heading", -Math.PI, Math.PI, deg], ["stance", "Leg spread", 0.75, 1.3, v => `${v.toFixed(2)}×`],
  ["headYaw", "Head: yaw", -0.7, 0.7, deg], ["headPitch", "Head: pitch", -0.6, 0.6, deg],
  ["abdPitch", "Abdomen: lift", -0.5, 0.8, deg], ["abdYaw", "Abdomen: side", -0.5, 0.5, deg],
  ["wingSpreadL", "Left wing", 0, 1, pct], ["wingSpreadR", "Right wing", 0, 1, pct],
  ["proboscis", "Proboscis", 0, 1, pct], ["antenna", "Antennae", -0.4, 0.6, deg],
];
let base = { ...REST };
const sliders = {};
for (const [k, label, min, max, fmt] of JOINTS) {
  const val = document.createElement("b"); val.className = "val";
  const lab = document.createElement("span"); lab.className = "lab"; lab.append(label, val);
  const input = Object.assign(document.createElement("input"), { type: "range", min, max, step: (max - min) / 200 });
  input.addEventListener("input", () => { base[k] = +input.value; val.textContent = fmt(base[k]); });
  const field = document.createElement("label"); field.className = "field"; field.append(lab, input);
  $("#joints").append(field);
  sliders[k] = { input, val, fmt };
}
function syncSliders() {
  for (const [k, s] of Object.entries(sliders)) { s.input.value = base[k]; s.val.textContent = s.fmt(base[k]); }
}
syncSliders();
$("#reset-pose").addEventListener("click", () => { base = { ...REST }; syncSliders(); });

/* ------------------------------------------------------------------- audio */
const audio = new AudioEngine();
audio.el = $("#player");
let volume = 0.8;
const LABEL = { demo: "demo beat · 98 BPM", mic: "microphone", none: "no source" };
function setSource(mode, name) {
  for (const [id, m] of [["#src-demo", "demo"], ["#src-file-btn", "file"], ["#src-mic", "mic"]])
    $(id).classList.toggle("on", mode === m);
  $("#src-label").textContent = mode === "file" ? name : LABEL[mode];
  $("#player").hidden = mode !== "file";
  if (audio.ctx) audio.setVolume(volume);
  status(mode === "none" ? "Source off" : `Source: ${mode === "file" ? name : LABEL[mode]}`);
}
$("#src-demo").addEventListener("click", () => { audio.useDemo({ breaks: $("#demo-breaks").checked }); setSource("demo"); });
$("#demo-breaks").addEventListener("change", e => { if (audio.demo) audio.demo.breaks = e.target.checked; });
async function playFile(file) {
  try { $("#player").hidden = false; await audio.useFile(file); setSource("file", file.name); }
  catch (err) { status(`Cannot play: ${err.message}`, true); }
}
$("#src-file").addEventListener("change", e => { const f = e.target.files[0]; if (f) playFile(f); e.target.value = ""; });
$("#player").addEventListener("error", () => status("Audio format not supported by the browser", true));
$("#src-mic").addEventListener("click", async () => {
  try { await audio.useMic(); setSource("mic"); }
  catch (err) { status(`Microphone unavailable: ${err.message}`, true); }
});
$("#src-stop").addEventListener("click", () => { audio.stop(); setSource("none"); });
stage.addEventListener("dragover", e => { e.preventDefault(); $("#drop").hidden = false; });
stage.addEventListener("dragleave", () => { $("#drop").hidden = true; });
stage.addEventListener("drop", e => {
  e.preventDefault(); $("#drop").hidden = true;
  const f = [...e.dataTransfer.files].find(f => f.type.startsWith("audio/")) || e.dataTransfer.files[0];
  if (f) playFile(f);
});

const opts = { enabled: true, intensity: 1, cadence: "auto", song: true };
const bindRange = (id, valId, fmt, apply) => {
  const i = $(id), show = () => { $(valId).textContent = fmt(+i.value); apply(+i.value); };
  i.addEventListener("input", show); show();
};
bindRange("#volume", "#vol-val", pct, v => { volume = v; if (audio.ctx) audio.setVolume(v); });
bindRange("#thr", "#thr-val", v => `${v} dB`, v => { audio.threshold = v; $("#meter-thr").style.left = `${(v + 80) / 80 * 100}%`; });
bindRange("#hold", "#hold-val", v => `${Math.round(v * 1000)} ms`, v => { audio.hold = v; });
bindRange("#intensity", "#int-val", pct, v => { opts.intensity = v; });
$("#dance-on").addEventListener("change", e => { opts.enabled = e.target.checked; });
$("#cadence").addEventListener("change", e => { opts.cadence = e.target.value; });
$("#song").addEventListener("change", e => { opts.song = e.target.checked; });

/* ------------------------------------------------------- level plot */
let ink = {};
const readInk = () => {
  const s = getComputedStyle(document.documentElement), g = n => s.getPropertyValue(n).trim();
  ink = { grid: g("--grid"), ink2: g("--ink-2"), muted: g("--muted"), accent: g("--accent"), axis: g("--axis") };
};
readInk();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readInk);
function drawSig(t) {
  const c = $("#sig"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr)) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); }
  const g = c.getContext("2d"), w = r.width, h = r.height, t0 = t - 6, top = 4, bot = h - 16;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const y = db => bot - (Math.max(-80, Math.min(0, db)) + 80) / 80 * (bot - top);
  const x = tt => (tt - t0) / 6 * w;
  g.fillStyle = ink.axis; g.fillRect(0, bot, w, 1);
  g.fillStyle = ink.ink2; g.fillRect(0, Math.round(y(audio.threshold)), w, 1);
  g.font = "10.5px system-ui, sans-serif"; g.fillStyle = ink.muted; g.textAlign = "right";
  g.fillText("threshold", w - 2, y(audio.threshold) - 3);
  g.textBaseline = "top"; g.fillText("now", w, bot + 3); g.textAlign = "left"; g.fillText("−6 s", 0, bot + 3);
  g.fillStyle = ink.ink2;
  for (const [tt, kick] of audio.onsets) g.fillRect(Math.round(x(tt)), bot - (kick ? 10 : 5), 1, kick ? 10 : 5);
  g.beginPath();
  audio.hist.forEach(([tt, db], i) => (i ? g.lineTo(x(tt), y(db)) : g.moveTo(x(tt), y(db))));
  g.strokeStyle = ink.accent; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();
}

/* ------------------------------------------------------------------ loop */
const dancer = new Dancer(), leds = [...$("#beats").children];
let beats = 0, spotPulse = 0, lastText = 0, lastChip = "", last = null;
function frame(ts) {
  requestAnimationFrame(frame);
  const now = ts / 1000, dt = last == null ? 0 : Math.min(0.1, now - last);
  last = now;
  const f = audio.update();
  let energy = 0;
  if (mode === "brain") {
    brain.consume(dt);
    const out = body.update(dt, brain.S, brain.raw, base);
    fly.apply(out.pose, out.feet);
    followCamera();
    if (brain3d.group.visible) {
      for (const s of brain.spikes) brain3d.spike(s);
      const names = brain.cur ? brain.cur[4] : [], k = names.join(",");
      if (k !== stimNames) { stimNames = k; brain3d.setStim(brain.inputIdx(names)); }
      brain3d.update(dt, fly.root.position, fly.root.rotation.y, renderer.domElement.height);
    }
    brain.spikes.length = 0;
    if ($("#live-audio").checked && brain.id != null) {
      // JO-B burst on every kick drum (or on any onset if the music has none)
      if (f.onset && (f.kick || now - lastKick > 1.5)) brain.audio(f.level, true);
      if (f.kick) lastKick = now;
      if (now - lastAudio > 0.1) { lastAudio = now; brain.audio(f.active ? f.level : 0); }
    }
  } else {
    const d = dancer.update(dt, f, opts, base);
    energy = d.energy;
    fly.apply(d.pose);
    if (f.beat) { lightTiles(beats++); spotPulse = 1; }
  }
  if (Math.abs(liftLeft) > 1e-4) {                             // camera rises to frame the 3D brain
    const d = liftLeft * (1 - Math.exp(-dt * 5));
    controls.target.y += d; camera.position.y += d; liftLeft -= d;
  }
  updateTiles(dt);
  spotPulse *= Math.exp(-dt * 6);
  spot.intensity = energy * (12 + 40 * spotPulse);
  if (camTween) {
    camTween.t = Math.min(1, camTween.t + dt / 0.7);
    const e = 1 - Math.pow(1 - camTween.t, 3);
    camera.position.lerpVectors(camTween.from, camTween.to, e);
    if (camTween.t >= 1) camTween = null;
  }
  controls.update(dt);
  renderer.render(scene, camera);

  // interface
  $("#meter-fill").style.width = `${Math.max(0, Math.min(100, (f.db + 80) / 80 * 100))}%`;
  const bi = ((Math.floor(f.phase) % 4) + 4) % 4, lit = f.active && f.phase - Math.floor(f.phase) < 0.3;
  leds.forEach((l, i) => l.classList.toggle("on", lit && i === bi));
  if (audio.hist.length) drawSig(f.t);
  if (now - lastText > 0.12) {
    lastText = now;
    $("#db-val").textContent = f.db > -99 ? `${Math.round(f.db)} dB` : "–";
    $("#bpm").textContent = f.bpm ? `${Math.round(f.bpm)} BPM` : "– BPM";
    if (mode === "brain") updateBrainUI();
    const state = mode === "brain" ? brainChip()
      : f.mode === "none" ? ["", "No audio source"]
      : !f.active ? ["listening", "Stopped · silence"]
      : !opts.enabled ? ["listening", "Sound detected · dancing off"]
      : ["dancing", `Twerking${f.bpm ? ` · ${Math.round(f.bpm)} BPM` : ""}`];
    if (state[1] !== lastChip) {
      lastChip = state[1];
      $("#state-chip").className = `chip ${state[0]}`; $("#state-text").textContent = state[1];
    }
  }
}

/* ------------------------------------------------------------- 3D brain */
async function showBrain(on) {
  on = on && mode === "brain";
  if (on === brain3d.group.visible) return;
  if (on) {
    try {
      if (!brain3d.N) status("Loading the neuron positions…");
      await brain3d.load();
      if (!$("#live-brain").checked || mode !== "brain") return;
      status(`3D brain: ${brain3d.N.toLocaleString("en-US")} neurons at their soma position`);
    } catch (e) { return status(`3D brain unavailable: ${e.message}`, true); }
  } else brain3d.clear();
  brain3d.group.visible = on; stimNames = "";
  liftLeft += on ? 0.6 : -0.6;
}
$("#live-brain").addEventListener("change", e => showBrain(e.target.checked));

/* ----------------------------------------------------------- brain mode */
const GROUPS = ["hearing", "locomotion", "jump", "proboscis", "wings", "head", "abdomen"];
const GROUP_LABEL = { hearing: "Hearing · Johnston's organ → descending neurons",
                      locomotion: "Descending neurons · locomotion", jump: "Jump", proboscis: "Proboscis",
                      wings: "Wings and halteres", head: "Head", abdomen: "Abdomen" };
const FUNCS = [["pro", "pro"], ["rem", "rem"], ["trx", "tr ext"], ["trf", "tr flx"],
               ["tix", "ti ext"], ["tif", "ti flx"], ["tad", "ta dep"], ["tal", "ta lev"]];
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
let chanRows = {}, legCells = {}, chanInfo = {};

function setMode(m) {
  mode = m;
  for (const b of $("#mode-seg").children) b.classList.toggle("on", b.dataset.mode === m);
  $("#brain-sec").hidden = m !== "brain"; $("#dance-sec").hidden = m === "brain"; $("#evlog").hidden = m !== "brain";
  $("#mode-note").textContent = m === "brain"
    ? "The body follows the descending and motor neurons of the connectome simulation."
    : "Choreography locked to the sound: no neurons are simulated.";
  if (m === "dance") { brain.stop(); $("#live-start").textContent = "Start simulation"; body.reset(); fly.root.position.set(0, 0, 0); followCamera(); }
  showBrain($("#live-brain").checked);
  lastChip = "";
}
$("#mode-seg").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setMode(b.dataset.mode); });

function buildBrainUI(info) {
  chanInfo = Object.fromEntries(info.channels.map(c => [c.key, c]));
  for (const kind of ["sens", "opto"]) {
    $(kind === "sens" ? "#ev-sens" : "#ev-opto").replaceChildren(...info.events.filter(e => e.kind === kind).map(e => {
      const b = el("button", "ghost", e.label);
      b.title = `${e.n} neurons · ${e.query}${e.side ? ` · ${e.side === "L" ? "left" : "right"} side` : ""} · ${e.hz} Hz for ${e.ms} ms`;
      b.addEventListener("click", () => { brain.event(e.key); body.note(`Stimulus: ${e.label} (${e.n} neurons)`); });
      return b;
    }));
  }
  const bars = $("#chan-bars"); bars.replaceChildren(); chanRows = {};
  const rows = info.channels.filter(c => c.group !== "legs").sort((a, b) => GROUPS.indexOf(a.group) - GROUPS.indexOf(b.group));
  let g = null;
  for (const c of rows) {
    if (c.group !== g) { g = c.group; bars.append(el("div", "g", GROUP_LABEL[g] || g)); }
    const lab = el("span", null, c.label); lab.title = `${c.n} neuron${c.n > 1 ? "s" : ""}`;
    const bar = el("div", "bar"), fill = el("i"); bar.append(fill);
    const v = el("span", "v", "0 Hz");
    bars.append(lab, bar, v); chanRows[c.key] = { fill, v };
  }
  const grid = $("#leg-grid"); grid.replaceChildren(el("span")); legCells = {};
  for (const [, l] of FUNCS) grid.append(el("span", "h", l));
  for (const leg of ["T1", "T2", "T3"]) for (const s of ["L", "R"]) {
    grid.append(el("span", "r", `${leg} ${s}`));
    for (const [f, l] of FUNCS) {
      const key = `${leg}${s}_${f}`, c = chanInfo[key], cell = el("div", c ? "c" : "c na");
      cell.title = c ? `${c.label} · ${c.n} MN` : `${leg} ${l}: no annotated motor neuron`;
      grid.append(cell); if (c) legCells[key] = cell;
    }
  }
}

function updateBrainUI() {
  const S = brain.S;
  for (const [k, r] of Object.entries(chanRows)) {
    const v = S[k] || 0; r.fill.style.width = `${Math.min(100, v / 2)}%`; r.v.textContent = `${Math.round(v)} Hz`;
  }
  for (const [k, c] of Object.entries(legCells)) {
    const a = 1 - Math.exp(-(S[k] || 0) / 30);
    c.style.background = `color-mix(in srgb, var(--accent) ${Math.round(a * 100)}%, var(--surface-2))`;
    c.title = `${chanInfo[k].label} · ${chanInfo[k].n} MN · ${Math.round(S[k] || 0)} Hz`;
  }
  const st = brain.status;
  $("#live-state").textContent = brain.id == null ? "stopped" : st ? `t ${(st.t_ms / 1000).toFixed(1)} s` : "starting…";
  if (brain.error) status(`Simulation: ${brain.error}`, true);
  const log = body.events.slice(-4);
  $("#evlog").replaceChildren(...(log.length ? log : [[body.t, "Start the simulation, then send a stimulus"]]).map(([t, txt]) => {
    const d = el("div"); d.append(el("b", null, `${t.toFixed(1)} s`), txt); return d;
  }));
}

function brainChip() {
  if (brain.id == null) return ["", "Brain · simulation stopped"];
  const st = brain.status;
  if (!st) return ["listening", "Brain · starting…"];
  const n = brain.cur ? brain.cur[3] : 0;
  return [n > 0 ? "dancing" : "listening", `Brain · ${st.speed.toFixed(2)}× real time · ${n.toLocaleString("en-US")} active neurons`];
}

const liveParams = () => ({ model: "shiu", dt: +$("#live-dt").value, w_scale: +$("#live-w").value,
                            std_u: +$("#live-u").value, std_tau: +$("#live-tau").value,
                            quench_ms: $("#live-quench").checked ? 1000 : 0 });
async function startLive(note) {
  try {
    status("Starting the simulation…");
    const info = await brain.start(liveParams());
    buildBrainUI(info);
    brain3d.clear(); stimNames = "";
    $("#live-start").textContent = "Stop";
    status(`Continuous simulation · ${info.channels.length} motor channels read`);
    if (note) body.note(note);
  } catch (e) { status(`Simulation failed: ${e.message}`, true); }
}
$("#live-start").addEventListener("click", () => {
  if (brain.id != null) { brain.stop(); $("#live-start").textContent = "Start simulation"; return; }
  startLive();
});
$("#live-zero").addEventListener("click", () => startLive("Network reset to rest"));
$("#live-reset").addEventListener("click", () => { body.reset(); });

requestAnimationFrame(frame);
status("Ready · pick a sound source: demo beat, file or microphone");

window.fly = {
  get pose() { return { ...base }; },
  set(k, v) { if (k in base) { base[k] = +v; syncSliders(); } },
  audio, dancer, model: fly, scene, camera, brain3d,
};
