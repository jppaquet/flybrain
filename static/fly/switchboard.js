/* switchboard.js - a console of toggle switches in front of the tethered fly (switchboard
   mode), seen from the front: switch 1 is on the fly's right (-X, screen left), the last one
   on its left; the fly flips each half with the front leg on that side. */
import * as THREE from "three";

const Z = 1.3, SPAN = 1.8, H = 0.1;        // console position (ahead of the fly), width of the switch row, height
const LEVER = 0.17, TILT = 0.45;           // lever length, lever angle (off: toward the fly, on: away)

function digitTexture(txt) {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#e8e7e1"; g.font = "600 44px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(txt, 32, 34);                 // lying on the console, readable from the front
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Switchboard {
  constructor(scene) {
    this.group = new THREE.Group(); this.group.visible = false; scene.add(this.group);
    this.metal = new THREE.MeshStandardMaterial({ color: 0x2a2b30, roughness: 0.45, metalness: 0.6 });
    this.dark = new THREE.MeshStandardMaterial({ color: 0x131316, roughness: 0.6, metalness: 0.3 });
    this.chrome = new THREE.MeshStandardMaterial({ color: 0xd8d8dc, roughness: 0.2, metalness: 1 });
    this.sw = [];
    this.setCount(8);
  }

  get n() { return this.sw.length; }
  label(i) { return String((i + 1) % 10); }
  side(i) { return this.sw[i].x >= 0 ? "L" : "R"; }
  isOn(i) { return this.sw[i].on; }

  setCount(n) {
    this.group.clear(); this.sw = [];
    const base = new THREE.Mesh(new THREE.BoxGeometry(SPAN + 0.4, H, 0.55), this.metal);
    base.position.set(0, H / 2, Z); base.castShadow = base.receiveShadow = true;
    this.group.add(base);
    for (let i = 0; i < n; i++) {
      const x = n > 1 ? -SPAN / 2 + i * SPAN / (n - 1) : 0;
      const plate = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.025, 0.2), this.dark);
      plate.position.set(x, H + 0.0125, Z);
      const lever = new THREE.Group(); lever.position.set(x, H + 0.025, Z);
      const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.02, LEVER, 12), this.chrome);
      stick.position.y = LEVER / 2; stick.castShadow = true;
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 14, 10), this.chrome);
      knob.position.y = LEVER; knob.castShadow = true;
      lever.add(stick, knob);
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.028, 14, 10),
        new THREE.MeshStandardMaterial({ color: 0x1a0d0d, emissive: 0x401010, emissiveIntensity: 0.4, roughness: 0.3 }));
      led.position.set(x, H + 0.02, Z - 0.17);
      const num = new THREE.Mesh(new THREE.PlaneGeometry(0.1, 0.1),
        new THREE.MeshBasicMaterial({ map: digitTexture(this.label(i)), transparent: true }));
      num.rotation.x = -Math.PI / 2; num.position.set(x, H + 0.002, Z + 0.2);
      this.group.add(plate, lever, led, num);
      this.sw.push({ x, on: false, angle: -TILT, lever, led, flash: 0 });
    }
    this.update(0);
  }

  /* World position of switch i's knob: the target of the front leg's tarsus. */
  pos(i, out = new THREE.Vector3()) {
    const s = this.sw[i];
    return out.set(s.x, H + 0.025 + LEVER * Math.cos(s.angle), Z + LEVER * Math.sin(s.angle));
  }

  toggle(i) { const s = this.sw[i]; s.on = !s.on; s.flash = 1; return s.on; }

  update(dt) {
    for (const s of this.sw) {
      s.angle += ((s.on ? TILT : -TILT) - s.angle) * (1 - Math.exp(-dt * 18));
      s.lever.rotation.x = s.angle;
      s.flash *= Math.exp(-dt * 4);
      s.led.material.emissive.setHex(s.on ? 0x39d353 : 0x401010);
      s.led.material.emissiveIntensity = (s.on ? 1.6 : 0.4) + 2 * s.flash;
    }
  }
}
