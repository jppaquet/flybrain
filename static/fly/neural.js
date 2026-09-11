/* neural.js - la mouche pilotée par la simulation du connectome (/api/live).

   Ce qui vient du connectome : les taux des neurones descendants et des motoneurones,
   lus toutes les 10 ms de temps simulé.
   Ce qui est procédural (choisi à la main, comme chez Eon) : la traduction de ces
   taux en angles, la marche en trépied (le LIF n'a pas de générateur de rythme)
   et la trajectoire du saut. */
import * as THREE from "three";

const act = (r, ref) => 1 - Math.exp(-Math.max(0, r) / ref);    // taux (Hz) -> activation 0..1
const frac = x => x - Math.floor(x);
const clamp01 = x => Math.min(1, Math.max(0, x));

/* ------------------------------------------------------------ flux de trames */
export class BrainLink {
  constructor() { this.reset(); }
  reset() {
    this.id = null; this.info = null; this.frames = []; this.next = 0; this.tPlay = null;
    this.S = {}; this.raw = {}; this.cur = null; this.status = null; this.error = null;
  }
  async start(params) {
    this.stop();
    const r = await fetch("/api/live/start", { method: "POST", body: JSON.stringify({ params }) });
    const info = await r.json();
    if (!r.ok) throw new Error(info.error || "démarrage impossible");
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
  audio(level) { if (this.id != null) fetch("/api/live/audio", { method: "POST", body: JSON.stringify({ id: this.id, level }) }); }

  /* Avance l'horloge de lecture au rythme du temps simulé et consomme les trames.
     Les taux sont lissés (constante 30 ms) ; `raw` garde la dernière trame brute. */
  consume(dt) {
    const F = this.frames;
    if (!F.length) return false;
    const latest = F[F.length - 1][0];
    if (this.tPlay == null) this.tPlay = F[0][0] - 10;
    this.tPlay = Math.min(this.tPlay + dt * 1000, latest - 10);
    if (latest - this.tPlay > 400) this.tPlay = latest - 60;           // trop en retard : on saute
    let n = 0;
    while (F.length && F[0][0] <= this.tPlay) {
      const f = F.shift(); this.cur = f; n++;
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

/* ------------------------------------------------------------ corps */
const TRIPOD_A = new Set([1, 2, 5]);     // ordre de fly.legs : T1D, T1G, T2D, T2G, T3D, T3G -> G1, D2, G3
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

    // --- saut : TTMn (motoneurone du muscle de saut, en aval de la fibre géante)
    if (!this.jump && this.t - this.lastJump > 1.2 && ((raw.ttmn || 0) > 45 || (raw.gf || 0) > 180)) {
      this.jump = { t: 0, vx: Math.sin(this.yaw), vz: Math.cos(this.yaw) };
      this.lastJump = this.t;
      this.note(`Saut · TTMn ${Math.round(raw.ttmn || 0)} Hz, fibre géante ${Math.round(raw.gf || 0)} Hz`);
    }

    // --- locomotion : DNp09 (avant), MDN (arrière), DNa01/02 (virage ipsilatéral)
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
      p.height += j.t < 0.08 ? 0.12 : 0;                    // extension des pattes médianes
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

    // --- motoneurones des pattes : décalages ajoutés aux cibles de tarse
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

    // --- autres motoneurones
    p.proboscis = Math.max(p.proboscis, clamp01(act(g("mn9"), 50) + 0.3 * a("pm", 40)));
    for (const side of ["L", "R"]) {
      const steer = a("wstr" + side), power = a("wpow" + side);
      const vib = 0.14 * steer * Math.sin(2 * Math.PI * 17 * this.t) + 0.5 * power * Math.sin(2 * Math.PI * 26 * this.t);
      p["wingSpread" + side] = Math.max(p["wingSpread" + side], 0.85 * steer, 0.95 * power);
      p["wingElev" + side] += vib;
    }
    p.headYaw += 0.45 * (a("neckL") - a("neckR"));
    p.headPitch -= 0.15 * (a("neckL") + a("neckR")) / 2;
    p.antenna += 0.35 * (a("antL") + a("antR")) / 2;
    p.abdPitch -= 0.35 * (a("abdL") + a("abdR")) / 2;
    p.abdYaw += 0.3 * (a("abdL") - a("abdR"));
    return { pose: p, feet };
  }

  /* Marche en trépied : pieds fixes au sol en appui, arc de transfert vers la
     position nominale anticipée pendant le balancement. */
  gait(dt, stance) {
    const legs = this.fly.legs;
    const speed = Math.abs(this.v) / VMAX + Math.abs(this.w) / WMAX;
    let disp = 0;
    const tmp = new THREE.Vector3();
    legs.forEach((_, k) => { disp = Math.max(disp, Math.hypot(...[this.feet[k].x - this.nominal(k, this.pos, this.yaw, stance, tmp).x, this.feet[k].z - tmp.z])); });
    const stepping = speed > 0.03 || disp > 0.12 || this.swing.some(Boolean);
    const freq = 3 + 9 * Math.min(1, speed);
    if (stepping) this.phase += dt * freq;
    const Ts = 0.5 / freq;                                   // durée d'appui
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
