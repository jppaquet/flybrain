/* dance.js - comportements superposés à la pose manuelle : repos (respiration,
   antennes) et twerk calé sur la phase du temps musical. */

const frac = x => x - Math.floor(x);
// « pop » : montée rapide sur le temps, retombée exponentielle
const pop = x => (x < 0.12 ? Math.sin((x / 0.12) * Math.PI / 2) : Math.exp(-(x - 0.12) * 5.5));

export class Dancer {
  constructor() { this.E = 0; this.t = 0; this.buzz = 0; this.buzzSide = 1; this.twitch = 0; this.nextTwitch = 2; }

  /* f : caractéristiques audio ; o : {enabled, intensity, cadence, song} ; base : pose manuelle */
  update(dt, f, o, base) {
    this.t += dt;
    const on = o.enabled && f.active;
    this.E += ((on ? 1 : 0) - this.E) * (1 - Math.exp(-dt / (on ? 0.12 : 0.3)));
    const E = this.E * o.intensity * (0.6 + 0.4 * f.level);
    const c = o.cadence === "auto" ? (f.bpm > 135 ? 1 : 2) : +o.cadence;
    const ph = f.phase, p1 = pop(frac(c * ph)), p2 = pop(frac(c * ph - 0.09));
    const sway = Math.sin(Math.PI * ph);                 // balancier sur 2 temps
    const p = { ...base };

    // repos
    p.abdPitch += 0.025 * Math.sin(2 * Math.PI * 0.33 * this.t);
    this.nextTwitch -= dt;
    if (this.nextTwitch < 0) { this.twitch = 1; this.nextTwitch = 1.5 + Math.random() * 3; }
    this.twitch *= Math.exp(-dt * 8);
    p.antenna += 0.25 * this.twitch;
    p.headYaw += 0.05 * Math.sin(this.t * 0.7) * (1 - this.E);

    // twerk : penchée en avant, fléchie, l'abdomen claque sur le temps
    p.pitch += E * 0.22;
    p.height -= E * (0.06 + 0.045 * p1);
    p.abdPitch += E * (0.12 + 0.55 * p1);
    p.abd2Pitch += E * (0.08 + 0.5 * p2);
    p.abdYaw += E * 0.2 * Math.sin(Math.PI * ph + 0.5);
    p.abdRoll += E * 0.14 * sway;
    p.roll += E * 0.07 * sway;
    p.headPitch -= (p.pitch - base.pitch) * 0.8 - E * 0.05 * p1;   // stabilisation du regard + hochement
    p.antenna += E * 0.18 * p1;
    const open = E * 0.28;                                       // ailes écartées pour libérer l'abdomen
    p.wingSpreadL = Math.max(p.wingSpreadL, open);
    p.wingSpreadR = Math.max(p.wingSpreadR, open);

    // chant de cour : sur le premier temps, une aile s'étend et vibre (côtés alternés)
    if (o.song && f.downbeat) { this.buzz = 1; this.buzzSide *= -1; }
    this.buzz = Math.max(0, this.buzz - dt / 0.45);
    if (this.buzz > 0) {
      const b = Math.sin(Math.PI * this.buzz) * this.E, v = 0.14 * Math.sin(2 * Math.PI * 22 * this.t) * b;
      if (this.buzzSide > 0) { p.wingSpreadL = Math.max(p.wingSpreadL, 0.85 * b); p.wingElevL += v; }
      else { p.wingSpreadR = Math.max(p.wingSpreadR, 0.85 * b); p.wingElevR += v; }
    }
    return { pose: p, energy: this.E };
  }
}
