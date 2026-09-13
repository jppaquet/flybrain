/* neural.js - the fly driven by the connectome simulation (/api/live).

   What comes from the connectome: the rates of the descending neurons and of the motor
   neurons, read every 10 ms of simulated time, and the neurons that fired (3D brain).
   What is procedural (hand-chosen, as in Eon's work): turning those rates into angles,
   the tripod gait (the LIF has no rhythm generator) and the jump trajectory. */
import * as THREE from "three";

const act = (r, ref) => 1 - Math.exp(-Math.max(0, r) / ref);    // rate (Hz) -> activation 0..1
const frac = x => x - Math.floor(x);
const clamp01 = x => Math.min(1, Math.max(0, x));
const b64i32 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Int32Array(u.buffer); };

/* ------------------------------------------------------------ frame stream */
export class BrainLink {
  constructor() { this.reset(); }
  reset() {
    this.id = null; this.info = null; this.frames = []; this.next = 0; this.tPlay = null;
    this.S = {}; this.raw = {}; this.cur = null; this.status = null; this.error = null; this.spikes = [];
  }
  async start(params) {
    this.stop();
    const r = await fetch("/api/live/start", { method: "POST", body: JSON.stringify({ params }) });
    const info = await r.json();
    if (!r.ok) throw new Error(info.error || "cannot start");
    this.reset(); this.info = info; this.id = info.id;
    this.keys = info.channels.map(c => c.key);
    this.poll(info.id);
    return info;
  }
  async poll(id) {
    while (this.id === id) {
      try {
        const r = await fetch(`/api/live/frames?id=${id}&since=${this.next}`);
        const d = await r.json();
        if (!r.ok) { this.error = d.error; break; }
        if (this.id !== id) break;
        this.frames.push(...d.frames); this.next = d.next; this.status = d;
        if (d.error) { this.error = d.error; break; }
      } catch (e) { this.error = e.message; }
      await new Promise(res => setTimeout(res, 80));
    }
  }
  stop() {
    if (this.id != null) fetch("/api/live/stop", { method: "POST", body: JSON.stringify({ id: this.id }) });
    this.id = null;
  }
  event(key) { if (this.id != null) fetch("/api/live/event", { method: "POST", body: JSON.stringify({ id: this.id, key }) }); }
  audio(level, hit = false) { if (this.id != null) fetch("/api/live/audio", { method: "POST", body: JSON.stringify({ id: this.id, level, hit }) }); }

  /* Neurons driven by the current inputs (frame names -> indices from the start info). */
  inputIdx(names) {
    return names.map(n => n.startsWith("ev:") ? this.info.events.find(e => e.key === n.slice(3))?.idx
                         : n.startsWith("audio") ? this.info.audio_idx : null).filter(Boolean);
  }

  /* Advances the playback clock at the pace of simulated time and consumes the frames.
     Rates are smoothed (30 ms constant); `raw` keeps the last raw frame; `spikes`
     collects the neurons that fired (brain frames), emptied by the caller. */
  consume(dt) {
    const F = this.frames;
    if (!F.length) return false;
    const latest = F[F.length - 1][0];
    if (this.tPlay == null) this.tPlay = F[0][0] - 10;
    this.tPlay = Math.min(this.tPlay + dt * 1000, latest - 10);
    if (latest - this.tPlay > 400) this.tPlay = latest - 60;           // too far behind: skip ahead
    let n = 0;
    while (F.length && F[0][0] <= this.tPlay) {
      const f = F.shift(); this.cur = f; n++;
      if (f[5] && this.onNote) this.onNote(f[5]);
      if (f[6]) this.spikes.push(b64i32(f[6]));
      const k = 1 - Math.exp(-10 / 30);
      f[1].forEach((r, i) => {
        const key = this.keys[i];
        this.S[key] = (this.S[key] || 0) + (r - (this.S[key] || 0)) * k;
        this.raw[key] = Math.max(r, n > 1 ? this.raw[key] || 0 : 0);
      });
    }
    return n > 0;
  }
}

/* ------------------------------------------------------------ body */
const TRIPOD_A = new Set([1, 2, 5]);     // order of fly.legs: T1R, T1L, T2R, T2L, T3R, T3L -> L1, R2, L3
const VMAX = 8, WMAX = 2.6;              // mm/s, rad/s

export class BodyController {
  constructor(fly) { this.fly = fly; this.reset(); }

  reset() {
    this.pos = new THREE.Vector3(); this.yaw = 0; this.v = 0; this.w = 0; this.phase = 0;
    this.feet = this.fly.legs.map((_, k) => this.nominal(k, this.pos, this.yaw));
    this.swing = this.fly.legs.map(() => null);
    this.jump = null; this.lastJump = -10; this.t = 0; this.events = [];
  }

  nominal(k, pos, yaw, stance = 1, out = new THREE.Vector3()) {
    const f = this.fly.legs[k].foot;
    const x = f.x * stance, z = f.z, c = Math.cos(yaw), s = Math.sin(yaw);
    return out.set(pos.x + x * c + z * s, 0, pos.z - x * s + z * c);
  }

  note(text) { this.events.push([this.t, text]); if (this.events.length > 6) this.events.shift(); }

  update(dt, S, raw, base) {
    this.t += dt;
    const g = k => S[k] || 0, a = (k, ref = 30) => act(g(k), ref);
    const p = { ...base };
    const root = this.fly.root;

    // --- jump: one spike of TTMn (motor neuron of the jump muscle, downstream of the
    //     giant fiber) is enough, as in the real fly
    if (!this.jump && this.t - this.lastJump > 1.2 && (raw.ttmn || 0) > 0) {
      this.jump = { t: 0, vx: Math.sin(this.yaw), vz: Math.cos(this.yaw) };
      this.lastJump = this.t;
      this.note(`Jump: TTMn fired (giant fiber at ${Math.round(S.gf || 0)} Hz)`);
    }

    // --- locomotion: DNp09 (forward), MDN (backward), DNa01/02 (ipsilateral turn)
    const vT = VMAX * a("fwd", 40) - 0.6 * VMAX * a("back", 40);
    const wT = WMAX * (a("turnL", 40) - a("turnR", 40));
    const kk = 1 - Math.exp(-dt / 0.15);
    this.v += (vT - this.v) * kk; this.w += (wT - this.w) * kk;
    if (!this.jump) {
      this.yaw += this.w * dt;
      this.pos.x += Math.sin(this.yaw) * this.v * dt; this.pos.z += Math.cos(this.yaw) * this.v * dt;
    }
    p.yaw = base.yaw + this.yaw;

    let feet;
    if (this.jump) {
      const T = 0.7, j = this.jump; j.t += dt;
      const u = Math.min(1, j.t / T), h = 4 * 2.2 * u * (1 - u);
      this.pos.x += j.vx * 3 * dt; this.pos.z += j.vz * 3 * dt;
      root.position.set(this.pos.x, h, this.pos.z);
      p.height += j.t < 0.08 ? 0.12 : 0;                    // middle legs extend
      p.wingSpreadL = p.wingSpreadR = 1;
      const flap = 0.55 * Math.sin(2 * Math.PI * 28 * this.t);
      p.wingElevL += flap; p.wingElevR += flap;
      root.updateMatrixWorld(true);
      feet = this.fly.legs.map((leg, k) => root.localToWorld(new THREE.Vector3(leg.foot.x * 0.7, 0.18, leg.foot.z * 0.8)));
      if (u >= 1) {
        this.jump = null; root.position.y = 0;
        this.feet = this.fly.legs.map((_, k) => this.nominal(k, this.pos, this.yaw));
        this.swing = this.swing.map(() => null);
      }
    } else {
      root.position.set(this.pos.x, 0, this.pos.z);
      feet = this.gait(dt, base.stance);
    }

    // --- leg motor neurons: offsets added to the tarsus targets
    if (!this.jump) {
      let push = 0;
      const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
      feet = feet.map((f, k) => {
        const leg = this.fly.legs[k], id = `${leg.name}${leg.side > 0 ? "L" : "R"}_`;
        const dz = 0.22 * (a(id + "pro") - a(id + "rem"));
        const flex = a(id + "trf") + a(id + "tif"), ext = a(id + "trx") + a(id + "tix");
        push += ext - flex;
        const lift = 0.28 * clamp01(flex - ext) + 0.08 * a(id + "tal");
        return new THREE.Vector3(f.x + dz * s, f.y + lift, f.z + dz * c);
      });
      p.height += 0.05 * push / feet.length;
    }

    // --- other motor neurons
    p.proboscis = Math.max(p.proboscis, clamp01(act(g("mn9"), 50) + 0.3 * a("pm", 40)));
    // wing steering: 20-30 Hz on every hit of sound (JO-B -> DNp02/06 -> MN); spread + raise
    for (const side of ["L", "R"]) {
      const steer = a("wstr" + side, 20), power = a("wpow" + side);
      const vib = 0.14 * steer * Math.sin(2 * Math.PI * 17 * this.t) + 0.5 * power * Math.sin(2 * Math.PI * 26 * this.t);
      p["wingSpread" + side] = Math.max(p["wingSpread" + side], 0.85 * steer, 0.95 * power);
      p["wingElev" + side] += vib + 0.25 * steer;
    }
    p.headYaw += 0.45 * (a("neckL") - a("neckR"));
    p.headPitch -= 0.15 * (a("neckL") + a("neckR")) / 2;
    p.antenna += 0.35 * (a("antL", 12) + a("antR", 12)) / 2;
    p.abdPitch -= 0.35 * (a("abdL") + a("abdR")) / 2;
    p.abdYaw += 0.3 * (a("abdL") - a("abdR"));
    return { pose: p, feet };
  }

  /* Tripod gait: feet fixed on the ground in stance, swing arc toward the anticipated
     nominal position. */
  gait(dt, stance) {
    const legs = this.fly.legs;
    const speed = Math.abs(this.v) / VMAX + Math.abs(this.w) / WMAX;
    let disp = 0;
    const tmp = new THREE.Vector3();
    legs.forEach((_, k) => { disp = Math.max(disp, Math.hypot(...[this.feet[k].x - this.nominal(k, this.pos, this.yaw, stance, tmp).x, this.feet[k].z - tmp.z])); });
    const stepping = speed > 0.03 || disp > 0.12 || this.swing.some(Boolean);
    const freq = 3 + 9 * Math.min(1, speed);
    if (stepping) this.phase += dt * freq;
    const Ts = 0.5 / freq;                                   // stance duration
    const yawP = this.yaw + this.w * Ts / 2;
    const posP = this.pos.clone().add(new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw)).multiplyScalar(this.v * Ts / 2));
    return legs.map((_, k) => {
      const ph = frac(this.phase + (TRIPOD_A.has(k) ? 0 : 0.5));
      if (stepping && ph >= 0.5) {
        if (!this.swing[k]) this.swing[k] = this.feet[k].clone();
        const s = (ph - 0.5) / 0.5, e = s * s * (3 - 2 * s);
        const target = this.nominal(k, posP, yawP, stance);
        this.feet[k].lerpVectors(this.swing[k], target, e);
        this.feet[k].y = 0.13 * Math.sin(Math.PI * s);
      } else if (this.swing[k]) {
        this.swing[k] = null; this.feet[k].y = 0;
      }
      return this.feet[k].clone();
    });
  }
}
