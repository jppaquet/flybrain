/* brain3d.js - the connectome as a 3D point cloud floating above the fly: every neuron at
   its soma position (MaleCNS), lit up when it fires in the live simulation.

   MaleCNS axes, checked on the data: +x = the fly's left, +y = ventral, +z = posterior.
   The scene's fly faces +Z with +Y up and +X to its left, so the cloud keeps the fly's own
   orientation: brain above the head, ventral nerve cord above the thorax. */
import * as THREE from "three";

const SCALE = 0.002;          // scene units per µm: the ~1 mm long CNS spans 2 units, like the fly
const LIFT = 1.55;            // height of the cloud's centre above the ground
const GROUP_RGB = [           // dim base colour per anatomical group (engine.GROUPS order)
  0x3987e5, 0x199e70, 0x9085e9, 0xd95926, 0xc98500, 0xd55181, 0xe8e7e1, 0x898781];

async function unpack(resp) {
  const buf = await resp.arrayBuffer();
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const T = { f4: Float32Array, f8: Float64Array, i4: Int32Array, u1: Uint8Array, u2: Uint16Array, i2: Int16Array };
  const arr = {};
  for (const s of header.arrays) arr[s.name] = new T[s.dtype](buf, 4 + hl + s.offset, s.length);
  return { ...header, ...arr };
}

const VERT = `
  attribute vec3 baseCol;
  attribute float heat;
  attribute float stim;
  uniform float uSize, uViewH, uBase;
  varying vec3 vCol;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vCol = baseCol * uBase + vec3(1.0, 0.55, 0.12) * 0.6 * stim + vec3(1.0, 0.93, 0.8) * heat * heat * 1.3;
    float s = uSize * (1.0 + 2.2 * heat + 1.2 * stim);
    gl_PointSize = max(1.0, s * projectionMatrix[1][1] * uViewH * 0.5 / -mv.z);
  }`;
const FRAG = `
  varying vec3 vCol;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    gl_FragColor = vec4(vCol * (1.0 - 4.0 * r2), 1.0);
  }`;

export class Brain3D {
  constructor(scene) {
    this.group = new THREE.Group(); this.group.visible = false;
    scene.add(this.group);
    this.ready = null; this.N = 0; this.nHot = 0; this.dirty = false; this.stimKey = "";
  }

  /* Loads the soma positions once (/api/meta, a few MB) and builds the cloud. */
  load() { return this.ready || (this.ready = this._load()); }
  async _load() {
    const M = await unpack(await fetch("/api/meta"));
    const N = this.N = M.N, P = M.pos;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
      const v = P[i * 3 + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v;
    }
    const c = lo.map((l, k) => (l + hi[k]) / 2);
    const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), rgb = GROUP_RGB.map(h => new THREE.Color(h));
    for (let i = 0; i < N; i++) {
      pos[i * 3] = (P[i * 3] - c[0]) * SCALE;              // +x left  -> +X left
      pos[i * 3 + 1] = -(P[i * 3 + 1] - c[1]) * SCALE;     // +y ventral -> -Y
      pos[i * 3 + 2] = -(P[i * 3 + 2] - c[2]) * SCALE;     // +z posterior -> -Z
      rgb[Math.min(M.group[i], 7)].toArray(col, i * 3);
    }
    this.heat = new Float32Array(N); this.stim = new Float32Array(N); this.hot = new Int32Array(N);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("baseCol", new THREE.BufferAttribute(col, 3));
    g.setAttribute("heat", this.heatAttr = new THREE.BufferAttribute(this.heat, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("stim", this.stimAttr = new THREE.BufferAttribute(this.stim, 1).setUsage(THREE.DynamicDrawUsage));
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uSize: { value: 0.006 }, uViewH: { value: 800 }, uBase: { value: 0.07 } },
      vertexShader: VERT, fragmentShader: FRAG,
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    });
    const pts = new THREE.Points(g, this.mat);
    pts.frustumCulled = false;
    this.group.add(pts);
  }

  /* Neurons that fired (Int32Array of indices): full brightness, then decay. */
  spike(list) {
    if (!this.N) return;
    const H = this.heat;
    for (let j = 0; j < list.length; j++) {
      const i = list[j];
      if (H[i] === 0) this.hot[this.nHot++] = i;
      H[i] = 1;
    }
    this.dirty = true;
  }

  /* Neurons currently driven by a stimulus (arrays of indices): tinted amber. */
  setStim(lists) {
    if (!this.N) return;
    const key = lists.map(l => l.length).join(",");
    this.stim.fill(0);
    for (const l of lists) for (const i of l) this.stim[i] = 1;
    this.stimAttr.needsUpdate = true; this.stimKey = key;
  }

  clear() {
    if (!this.N) return;
    this.heat.fill(0); this.nHot = 0; this.stim.fill(0);
    this.heatAttr.needsUpdate = this.stimAttr.needsUpdate = true;
  }

  /* Each rendered frame: decay, follow the fly (position and heading), point scale. */
  update(dt, flyPos, yaw, viewH) {
    if (!this.N) return;
    const H = this.heat, k = Math.exp(-dt / 0.15);
    let n = 0;
    for (let j = 0; j < this.nHot; j++) {
      const i = this.hot[j];
      H[i] *= k;
      if (H[i] > 0.02) this.hot[n++] = i; else H[i] = 0;
    }
    if (n || this.nHot || this.dirty) this.heatAttr.needsUpdate = true;
    this.nHot = n; this.dirty = false;
    this.group.position.set(flyPos.x, LIFT, flyPos.z);
    this.group.rotation.y = yaw;
    this.mat.uniforms.uViewH.value = viewH;
  }
}
