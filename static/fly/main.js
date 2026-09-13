/* main.js - three.js scene, animation loop and control panel of the fly. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createFly, REST } from "./model.js";
import { AudioEngine } from "./audio.js";
import { BrainLink, BodyController } from "./neural.js";
import { Brain3D } from "./brain3d.js";
import { Switchboard } from "./switchboard.js";
import { Kart3D, DECK } from "./kart.js";

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
// glossy ground
const floor = new THREE.Mesh(new THREE.CircleGeometry(60, 96),
  new THREE.MeshStandardMaterial({ color: 0x0e0e10, roughness: 0.7, metalness: 0.05, envMapIntensity: 0.12 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

const fly = createFly();
scene.add(fly.root);
const grid = new THREE.GridHelper(80, 160, 0x2c2c31, 0x1d1d21);   // ground reference to see the walking
grid.position.y = 0.001; scene.add(grid);

const brain = new BrainLink(), body = new BodyController(fly), brain3d = new Brain3D(scene), board = new Switchboard(scene);
const kart3d = new Kart3D(scene);
brain.onNote = txt => body.note(txt);
let mode = "brain", lastAudio = 0, lastKick = -10, stimNames = "", liftLeft = 0, liftApplied = 0;
let keysHeld = new Set(), lastDrive = 0, dragging = false;
let presses = { L: null, R: null }, reachInfo = {}, queue = [];   // switchboard: press in progress per front leg, keys waiting
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
  $("#chase").checked = false;                                 // a chosen view replaces the chase camera
}
/* Keyboard mode: the camera stays behind the fly (paused while the user drags the view). */
const _want = new THREE.Vector3();
function chaseCamera(dt, D = 5.4, H = 1.9) {
  const t = controls.target, yaw = fly.root.rotation.y;
  _want.set(t.x - Math.sin(yaw) * D, t.y + H, t.z - Math.cos(yaw) * D);
  camera.position.lerp(_want, 1 - Math.exp(-dt * 2.5));
}
controls.addEventListener("start", () => { dragging = true; });
controls.addEventListener("end", () => { dragging = false; });
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

const bindRange = (id, valId, fmt, apply) => {
  const i = $(id), show = () => { $(valId).textContent = fmt(+i.value); apply(+i.value); };
  i.addEventListener("input", show); show();
};
bindRange("#volume", "#vol-val", pct, v => { volume = v; if (audio.ctx) audio.setVolume(v); });
bindRange("#thr", "#thr-val", v => `${v} dB`, v => { audio.threshold = v; $("#meter-thr").style.left = `${(v + 80) / 80 * 100}%`; });
bindRange("#hold", "#hold-val", v => `${Math.round(v * 1000)} ms`, v => { audio.hold = v; });

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
const leds = [...$("#beats").children];
let lastText = 0, lastChip = "", last = null;
function frame(ts) {
  requestAnimationFrame(frame);
  const now = ts / 1000, dt = last == null ? 0 : Math.min(0.1, now - last);
  last = now;
  const f = audio.update();
  {
    brain.consume(dt);
    const k = brain.kart;
    if (mode === "car" && k) { body.pos.set(k[0], 0, k[1]); body.yaw = k[2]; }   // the fly rides the kart
    const out = body.update(dt, brain.S, brain.raw, base,
      mode === "board" ? { tethered: true, reach: reachOpt() } : mode === "car" ? { tethered: true } : undefined);
    if (mode === "car") seatInKart(out, dt);
    fly.apply(out.pose, out.feet);
    followCamera();
    if (mode === "board") { checkPresses(now, out.reach); board.update(dt); }
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
    if (mode === "keys" || mode === "car") {
      if (keysHeld.size && now - lastDrive > 0.1) sendDrive();   // keep-alive: inputs expire after 300 ms
      if (mode === "keys" && $("#chase").checked && !dragging && !camTween) chaseCamera(dt);
      if (mode === "car" && !dragging && !camTween) chaseCamera(dt, 8.5, 3.2);
    }
  }
  if (Math.abs(liftLeft) > 1e-4) {                             // camera rises to frame the 3D brain
    const d = liftLeft * (1 - Math.exp(-dt * 5));
    controls.target.y += d; camera.position.y += d; liftLeft -= d; liftApplied += d;
  }
  if (camTween) {
    camTween.t = Math.min(1, camTween.t + dt / 0.7);
    const e = 1 - Math.pow(1 - camTween.t, 3);
    camera.position.lerpVectors(camTween.from, camTween.to, e);
    if (camTween.tt) controls.target.lerpVectors(camTween.tf, camTween.tt, e);   // also move the point looked at
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
    updateBrainUI();
    const state = brainChip();
    if (state[1] !== lastChip) {
      lastChip = state[1];
      $("#state-chip").className = `chip ${state[0]}`; $("#state-text").textContent = state[1];
    }
  }
}

/* ------------------------------------------------------------- 3D brain */
async function showBrain(on) {
  if (on === brain3d.group.visible) return;
  if (on) {
    try {
      if (!brain3d.N) status("Loading the neuron positions…");
      await brain3d.load();
      if (!$("#live-brain").checked) return;
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
let chanRows = {}, legCells = {}, chanInfo = {}, carChan = {};

const MODE_NOTE = {
  brain: "The body follows the descending and motor neurons of the connectome simulation.",
  keys: "The arrow keys drive the fly's command descending neurons; the body follows the connectome simulation.",
  board: "The number keys make the tethered fly press switches with its front legs, moved by its own nerve cord.",
  car: "The fly drives a kart: its leg motor neurons turn the wheel and press the pedals.",
};
function setMode(m) {
  releaseKeys();
  const prev = mode;
  mode = m;
  if (prev === "board" && m !== "board") {                     // leave the console view: back to the fly
    controls.target.set(fly.root.position.x, 0.42 + liftApplied, fly.root.position.z - 0.1);
    if (m !== "keys" && m !== "car") setCam("rear");
  }
  if (prev === "car" && m !== "car") {                         // out of the kart, back on the floor
    brain.world(null); body.reset(); fly.root.position.set(0, 0, 0); followCamera();
  }
  for (const b of $("#mode-seg").children) b.classList.toggle("on", b.dataset.mode === m);
  $("#keys-sec").hidden = m !== "keys"; $("#board-sec").hidden = m !== "board"; $("#car-sec").hidden = m !== "car";
  $("#mode-note").textContent = MODE_NOTE[m];
  board.group.visible = m === "board";
  kart3d.setVisible(m === "car");
  if (m !== "board") { presses = { L: null, R: null }; queue = []; }
  if (m === "board") { body.reset(); fly.root.position.set(0, 0, 0); followCamera(); boardView(); }
  if (m === "car") { body.reset(); if (brain.id != null) brain.world("kart", true); }
  if (m !== "brain") {
    document.activeElement?.blur();                            // Space and digits must not click a focused button
    if (brain.id == null) startLive();
  }
  showBrain($("#live-brain").checked);
  lastChip = "";
}
$("#mode-seg").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setMode(b.dataset.mode); });

/* ------------------------------------------------------------- keyboard */
const KEYMAP = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
                 KeyW: "up", KeyS: "down", KeyA: "left", KeyD: "right" };
const KEY_CHAN = { up: "fwd", down: "back", left: "turnL", right: "turnR", jump: "ttmn" };
const typing = e => /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && !["checkbox", "button"].includes(e.target.type);
function sendDrive() {
  lastDrive = performance.now() / 1000;
  // opposite commands cancel out (driving both of a pair together can also run the network away)
  const drop = k => (keysHeld.has("up") && keysHeld.has("down") && (k === "up" || k === "down"))
                 || (keysHeld.has("left") && keysHeld.has("right") && (k === "left" || k === "right"));
  brain.drive([...keysHeld].filter(k => !drop(k)), mode === "car" ? "car" : "walk");
}
function setKey(k, on) {
  if (on === keysHeld.has(k)) return;
  if (on) keysHeld.add(k); else keysHeld.delete(k);
  for (const e of document.querySelectorAll(`.keypad [data-k="${k}"]`)) e.classList.toggle("on", on);
  sendDrive();
}
function releaseKeys() { for (const k of [...keysHeld]) setKey(k, false); }
function jump() {
  if (brain.id == null) return;
  brain.event("loom"); body.note("Space: looming on both eyes");
}
addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey || typing(e)) return;
  if (mode === "board" && /^(Digit|Numpad)\d$/.test(e.code)) {
    e.preventDefault();
    const d = +e.code.slice(-1);
    if (!e.repeat) press(d === 0 ? 9 : d - 1);
    return;
  }
  if (mode !== "keys" && mode !== "car") return;
  if (mode === "car" && e.code === "KeyR") { e.preventDefault(); if (!e.repeat) carReset(); return; }
  if (e.code === "Space") {
    e.preventDefault();
    if (mode !== "keys") return;
    $('#keypad [data-k="jump"]').classList.add("on");
    if (!e.repeat) jump();
    return;
  }
  const k = KEYMAP[e.code];
  if (k) { e.preventDefault(); setKey(k, true); }
});
addEventListener("keyup", e => {
  if (e.code === "Space") { $('#keypad [data-k="jump"]').classList.remove("on"); if (mode === "keys") e.preventDefault(); }
  const k = KEYMAP[e.code];
  if (k) setKey(k, false);
});
addEventListener("blur", releaseKeys);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseKeys(); });

/* ----------------------------------------------------------- switchboard */
/* Three-quarter front view: the console in the foreground, the fly reaching toward us,
   the 3D brain above it in the background. */
function boardView() {
  camTween = { from: camera.position.clone(), to: new THREE.Vector3(1.2, 1.95, 4.6),
               tf: controls.target.clone(), tt: new THREE.Vector3(0, 0.55, 1.0), t: 0 };
  for (const b of $("#cams").querySelectorAll("[data-cam]")) b.classList.remove("on");
}
/* Number key i: the descending neuron of that side's front leg gets a burst; the leg then
   reaches as far as its motor neurons fire (BodyController), and checkPresses flips the
   switch when the tarsus gets there. */
function press(i) {
  if (i >= board.n) return;
  if (brain.id == null || !reachInfo.L) return body.note("Start the simulation first");
  if (queue.length < 4) queue.push(i);
  startNext(); renderSwitchRow();
}
/* One leg at a time: driving both DNg12_e together tips the network into its runaway state. */
function startNext() {
  if (presses.L || presses.R || !queue.length) return;
  const i = queue.shift(), s = board.side(i);
  presses[s] = { i, t0: performance.now() / 1000, peak: 0, hit: 0, done: false };
  brain.takePeak(reachInfo[s].read);                           // forget activity from before the key
  brain.reach(s);
  body.note(`Key ${board.label(i)} → ${reachInfo[s].label}`);
}
const _knob = { L: new THREE.Vector3(), R: new THREE.Vector3() };
function reachOpt() {
  const o = {};
  for (const s of ["L", "R"]) {
    const p = presses[s], R = reachInfo[s];
    if (R) o[s] = { target: p && !p.done ? board.pos(p.i, _knob[s]) : null, read: R.read, ref: R.ref };
  }
  return o;
}
function checkPresses(now, reach) {
  for (const s of ["L", "R"]) {
    const p = presses[s], R = reachInfo[s];
    if (!p || !R) continue;
    const leg = s === "L" ? "left" : "right";
    p.peak = Math.max(p.peak, brain.takePeak(R.read));
    // the motor neurons decide (peak over every simulated frame); the flip waits for the leg to arrive
    if (!p.hit && p.peak >= R.press) p.hit = now;
    if (p.hit && !p.done && (reach[s] > 0.75 || now - p.hit > 0.25)) {
      p.done = true;
      const on = board.toggle(p.i);
      body.note(`Switch ${board.label(p.i)} ${on ? "on" : "off"} · ${leg} front leg motor neurons at ${Math.round(p.peak)} Hz`);
    } else if (!p.hit && !p.done && now - p.t0 > R.ms / 1000 + 0.8) {
      p.done = true;
      body.note(`Switch ${board.label(p.i)}: no press, the ${leg} front leg motor neurons only reached ${Math.round(p.peak)} Hz (needs ${R.press})`);
    }
    if (p.done && reach[s] < 0.15 && now - p.t0 > 0.3) presses[s] = null;
  }
  startNext(); renderSwitchRow();
}
function renderSwitchRow() {
  const row = $("#sw-row");
  if (row.children.length !== board.n) row.replaceChildren(...[...Array(board.n).keys()].map(i => el("span", null, board.label(i))));
  [...row.children].forEach((c, i) => {
    c.classList.toggle("on", board.isOn(i));
    c.classList.toggle("busy", presses.L?.i === i || presses.R?.i === i || queue.includes(i));
  });
}
function renderReachMap() {
  if (!reachInfo.L) return;
  const range = s => { const k = [...Array(board.n).keys()].filter(i => board.side(i) === s).map(i => board.label(i));
                       return k.length > 1 ? `${k[0]}–${k[k.length - 1]}` : k[0] || "–"; };
  $("#reachmap").replaceChildren(...["R", "L"].flatMap(s => [el("dt", null, `Keys ${range(s)}`),   // switch 1 is on the right
    el("dd", null, `${reachInfo[s].label} · ${reachInfo[s].hz} Hz × ${reachInfo[s].ms} ms · flips when the ${reachInfo[s].read_label} reach ${reachInfo[s].press} Hz`)]));
}
/* ------------------------------------------------------------------ kart */
/* The fly stands on the kart's deck: front tarsi on the steering wheel rim, hind tarsi on
   the pedals. The kart itself moves as the server's world.Driver computes it. */
function seatInKart(out, dt) {
  kart3d.update(brain.kart, dt);
  if (!brain.kart) return;
  fly.root.position.y = DECK;
  for (const f of out.feet) f.y += DECK;
  out.feet[1] = kart3d.grip("L", new THREE.Vector3()); out.feet[0] = kart3d.grip("R", new THREE.Vector3());
  out.feet[5] = kart3d.pedal("L", new THREE.Vector3()); out.feet[4] = kart3d.pedal("R", new THREE.Vector3());
}
function carReset() { if (brain.id != null) { brain.world("kart", true); body.note("Kart back on the road"); } }
$("#car-reset").addEventListener("click", carReset);

$("#sw-count").addEventListener("change", e => {
  board.setCount(+e.target.value); presses = { L: null, R: null }; queue = [];
  $("#board-state").textContent = `keys 1–${board.label(board.n - 1)}`;
  renderSwitchRow(); renderReachMap(); e.target.blur();      // digits must reach the page, not the select
});
renderSwitchRow();
function buildBrainUI(info) {
  chanInfo = Object.fromEntries(info.channels.map(c => [c.key, c]));
  reachInfo = Object.fromEntries(info.reach.map(r => [r.side, r])); renderReachMap();
  const ARROW = { up: "↑ W", down: "↓ S", left: "← A", right: "→ D" }, loom = info.events.find(e => e.key === "loom");
  $("#keymap").replaceChildren(...info.keysets.walk.flatMap(d => [el("dt", null, ARROW[d.key]),
    el("dd", null, `${d.label} · ${d.hz} Hz · ${d.n} neuron${d.n > 1 ? "s" : ""}`)]),
    el("dt", null, "Space"), el("dd", null, `Jump: looming on both eyes (LC4, ${loom.n} neurons) → TTMn`));
  $("#carmap").replaceChildren(...info.keysets.car.flatMap(d => [el("dt", null, ARROW[d.key]),
    el("dd", null, `${d.label} · ${d.hz} Hz · ${d.n} neuron${d.n > 1 ? "s" : ""}`)]),
    el("dt", null, "R"), el("dd", null, "back on the road"));
  const C = info.controls;
  carChan = { up: C.throttle.read, down: C.brake.read, left: C.wheel_left.read, right: C.wheel_right.read };
  kart3d.setTrack(info.track);
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
  // each key lights up with the firing of the neurons it drives (fwd, back, turn, TTMn)
  for (const [k, ch] of Object.entries(KEY_CHAN)) {
    const a = 1 - Math.exp(-(S[ch] || 0) / (k === "jump" ? 5 : 20));
    $(`#keypad [data-k="${k}"]`).style.background = `color-mix(in srgb, var(--accent) ${Math.round(a * 70)}%, var(--surface-2))`;
  }
  // Drive: keys light up with the motor neurons the kart reads; gauges from the server's kart
  for (const [k, chs] of Object.entries(carChan)) {
    const r = chs.reduce((s, c) => s + (S[c] || 0), 0) / chs.length, a = 1 - Math.exp(-r / 12);
    $(`#carpad [data-k="${k}"]`).style.background = `color-mix(in srgb, var(--accent) ${Math.round(a * 70)}%, var(--surface-2))`;
  }
  if (mode === "car" && brain.kart) {
    const [, , , v, wheel, thr, brk, off, dist] = brain.kart;
    $("#car-speed").textContent = v.toFixed(1); $("#car-dist").textContent = Math.round(dist);
    $("#car-off").textContent = off.toFixed(1);
    $("#car-wheel").style.left = `${50 + Math.min(0, wheel) * 50}%`; $("#car-wheel").style.width = `${Math.abs(wheel) * 50}%`;
    $("#car-thr").style.width = `${thr * 100}%`; $("#car-brk").style.width = `${brk * 100}%`;
  }
  if (mode === "keys") $("#keys-state").textContent = brain.id == null ? "start the simulation"
    : keysHeld.size ? [...keysHeld].map(k => ({ up: "↑", down: "↓", left: "←", right: "→" })[k]).join(" ") : "arrow keys or WASD";
  const st = brain.status;
  $("#live-state").textContent = brain.id == null ? "stopped" : st ? `t ${(st.t_ms / 1000).toFixed(1)} s` : "starting…";
  if (brain.error) status(`Simulation: ${brain.error}`, true);
  const log = body.events.slice(-4);
  const hint = brain.id == null ? "Start the simulation" : mode === "keys" ? "Hold an arrow key to walk, Space to jump"
    : mode === "board" ? `Press 1–${board.label(board.n - 1)} to flip a switch`
    : mode === "car" ? "Hold ↑ to accelerate, ↓ to brake, ← → to steer · R: back on the road" : "Send a stimulus";
  $("#evlog").replaceChildren(...(log.length ? log : [[body.t, hint]]).map(([t, txt]) => {
    const d = el("div"); d.append(el("b", null, `${t.toFixed(1)} s`), txt); return d;
  }));
}

function brainChip() {
  const who = { keys: "Keyboard", board: "Switchboard", car: "Drive" }[mode] || "Brain";
  if (brain.id == null) return ["", `${who} · simulation stopped`];
  const st = brain.status;
  if (!st) return ["listening", `${who} · starting…`];
  const n = brain.cur ? brain.cur[3] : 0;
  return [n > 0 ? "dancing" : "listening", `${who} · ${st.speed.toFixed(2)}× real time · ${n.toLocaleString("en-US")} active neurons`];
}

const liveParams = () => ({ model: "shiu", dt: +$("#live-dt").value, w_scale: +$("#live-w").value,
                            std_u: +$("#live-u").value, std_tau: +$("#live-tau").value,
                            quench_ms: $("#live-quench").checked ? 1000 : 0, world: mode === "car" ? "kart" : null });
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
  if (brain.id != null) { releaseKeys(); brain.stop(); $("#live-start").textContent = "Start simulation"; return; }
  startLive();
});
$("#live-zero").addEventListener("click", () => startLive("Network reset to rest"));
$("#live-reset").addEventListener("click", () => { body.reset(); });

setMode("brain");
requestAnimationFrame(frame);
status("Ready · start the simulation");

window.fly = {
  get pose() { return { ...base }; },
  set(k, v) { if (k in base) { base[k] = +v; syncSliders(); } },
  audio, model: fly, scene, camera, brain3d, board, kart3d, brain,
};
