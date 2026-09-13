/* kart.js - the kart the fly drives and its circular road (same geometry as world.py).
   Kart frame = the fly's frame: +Z forward, +X the fly's left. The fly stands on the deck,
   its front legs on the steering wheel, its hind legs on the pedals (right: accelerator,
   left: brake). The kart's motion comes from the server (world.Driver), frame by frame. */
import * as THREE from "three";

export const DECK = 0.06;                       // height of the deck the fly stands on
const RIM = 0.3, TYRE = 0.42;

export class Kart3D {
  constructor(scene) {
    this.group = new THREE.Group(); this.road = new THREE.Group();
    this.group.visible = this.road.visible = false;
    scene.add(this.group, this.road);
    const paint = new THREE.MeshStandardMaterial({ color: 0xc93a2a, roughness: 0.35, metalness: 0.4 });
    const metal = new THREE.MeshStandardMaterial({ color: 0x2a2b30, roughness: 0.45, metalness: 0.6 });
    const rubber = new THREE.MeshStandardMaterial({ color: 0x151518, roughness: 0.85 });
    const chrome = new THREE.MeshStandardMaterial({ color: 0xd8d8dc, roughness: 0.2, metalness: 1 });
    const shade = m => { m.castShadow = true; m.receiveShadow = true; return m; };

    const deck = shade(new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.1, 3.4), metal));
    deck.position.set(0, DECK - 0.05, 0.1); this.group.add(deck);
    const nose = shade(new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.22, 0.7), paint));
    nose.position.set(0, 0.11, 1.75); this.group.add(nose);
    const tail = shade(new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.3, 0.5), paint));
    tail.position.set(0, 0.15, -1.55); this.group.add(tail);

    this.tyres = [];
    for (const [x, z, front] of [[1.2, 1.25, true], [-1.2, 1.25, true], [1.2, -1.2, false], [-1.2, -1.2, false]]) {
      const hub = new THREE.Group(); hub.position.set(x, TYRE, z);
      const spin = new THREE.Group(); hub.add(spin);
      const tyre = shade(new THREE.Mesh(new THREE.CylinderGeometry(TYRE, TYRE, 0.32, 28), rubber));
      tyre.rotation.z = Math.PI / 2; spin.add(tyre);
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.34, 16), chrome);
      cap.rotation.z = Math.PI / 2; spin.add(cap);
      this.group.add(hub); this.tyres.push({ hub, spin, front });
    }

    // steering column and wheel, leaning toward the fly
    const col = shade(new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.75, 10), metal));
    col.position.set(0, 0.35, 1.28); col.rotation.x = -0.6; this.group.add(col);
    this.wheelMount = new THREE.Group(); this.wheelMount.position.set(0, 0.66, 1.08); this.wheelMount.rotation.x = -0.6;
    this.group.add(this.wheelMount);
    this.wheel = new THREE.Group(); this.wheelMount.add(this.wheel);
    this.wheel.add(shade(new THREE.Mesh(new THREE.TorusGeometry(RIM, 0.035, 12, 40), rubber)));
    for (const a of [Math.PI / 2, Math.PI * 7 / 6, Math.PI * 11 / 6]) {
      const spoke = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, RIM, 8), chrome);
      spoke.position.set(Math.cos(a) * RIM / 2, Math.sin(a) * RIM / 2, 0); spoke.rotation.z = a - Math.PI / 2;
      this.wheel.add(spoke);
    }
    // where the front tarsi hold the rim (+X = the fly's left), rotating with the wheel
    this.grips = { L: new THREE.Object3D(), R: new THREE.Object3D() };
    this.grips.L.position.set(RIM * Math.sin(1.1), RIM * Math.cos(1.1), 0);
    this.grips.R.position.set(-RIM * Math.sin(1.1), RIM * Math.cos(1.1), 0);
    this.wheel.add(this.grips.L, this.grips.R);

    // pedals under the hind tarsi: right = accelerator (green), left = brake (red)
    this.pedals = {};
    for (const [s, x, c] of [["R", -0.72, 0x2fa15a], ["L", 0.72, 0xc9352a]]) {
      const pivot = new THREE.Group(); pivot.position.set(x, DECK, -0.5);
      const plate = shade(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.04, 0.42),
        new THREE.MeshStandardMaterial({ color: c, roughness: 0.4, metalness: 0.3 })));
      plate.position.set(0, 0.08, -0.2); pivot.add(plate);
      const top = new THREE.Object3D(); top.position.set(0, 0.1, -0.18); pivot.add(top);
      this.group.add(pivot); this.pedals[s] = { pivot, top };
    }
    this.state = null;
  }

  /* The road: a ring of radius R and half width W around (cx, cz), from the server. */
  setTrack(t) {
    this.road.clear();
    const asphalt = new THREE.Mesh(new THREE.RingGeometry(t.radius - t.half_width, t.radius + t.half_width, 160, 1),
      new THREE.MeshStandardMaterial({ color: 0x2b2c31, roughness: 0.9 }));
    asphalt.rotation.x = -Math.PI / 2; asphalt.position.set(t.cx, 0.003, t.cz); asphalt.receiveShadow = true;
    this.road.add(asphalt);
    const white = new THREE.MeshBasicMaterial({ color: 0xe8e7e1 });
    for (const r of [t.radius - t.half_width + 0.15, t.radius + t.half_width - 0.15]) {
      const edge = new THREE.Mesh(new THREE.RingGeometry(r - 0.08, r + 0.08, 160, 1), white);
      edge.rotation.x = -Math.PI / 2; edge.position.set(t.cx, 0.005, t.cz); this.road.add(edge);
    }
    const n = 72;
    for (let i = 0; i < n; i++) {                             // dashed centre line
      const a0 = (i / n) * Math.PI * 2, dash = new THREE.Mesh(
        new THREE.RingGeometry(t.radius - 0.07, t.radius + 0.07, 6, 1, a0, Math.PI / n), white);
      dash.rotation.x = -Math.PI / 2; dash.position.set(t.cx, 0.006, t.cz); this.road.add(dash);
    }
  }

  setVisible(on) { this.group.visible = this.road.visible = on; }

  /* s: the server's kart frame [x, z, yaw, speed, wheel, throttle, brake, offset, distance, ...]. */
  update(s, dt) {
    if (!s) return;
    this.state = s;
    const [x, z, yaw, v, wheel, throttle, brake] = s;
    this.group.position.set(x, 0, z); this.group.rotation.y = yaw;
    this.wheel.rotation.z = -wheel * 1.6;                     // + wheel = turning right
    for (const t of this.tyres) {
      if (t.front) t.hub.rotation.y = -wheel * 0.5;
      t.spin.rotation.x += v * dt / TYRE;
    }
    this.pedals.R.pivot.rotation.x = -0.35 * throttle;
    this.pedals.L.pivot.rotation.x = -0.35 * brake;
    this.group.updateMatrixWorld(true);
  }

  grip(side, out) { return this.grips[side].getWorldPosition(out); }
  pedal(side, out) { return this.pedals[side].top.getWorldPosition(out); }
}
