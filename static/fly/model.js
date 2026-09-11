/* model.js - Drosophila melanogaster mâle, procédurale et articulée (unités : mm).
   Repère : +Z vers l'avant, +Y vers le haut, +X vers la gauche de la mouche. */
import * as THREE from "three";

const COL = {
  thorax: 0x9b6d3a, abdomen: 0xc0915a, band: 0x3a2515, male: 0x1b120c, pale: 0xe0c393,
  leg: 0xb58b56, eye: 0xb3141c, bristle: 0x140e0a, frons: 0xc9884a,
};
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const col = h => new THREE.Color(h);

const cuticle = (color, extra = {}) => new THREE.MeshPhysicalMaterial({
  color, roughness: 0.42, clearcoat: 0.65, clearcoatRoughness: 0.32,
  sheen: 0.35, sheenColor: col(0xffe1b0), ...extra,
});
const shade = m => { m.castShadow = true; m.receiveShadow = true; return m; };

/* Ellipsoïde ; colorFn(x, y, z, color) peut teinter chaque sommet (sphère unité)
   et renvoyer un facteur radial (sillons entre segments). */
function ellipsoid(rx, ry, rz, material, colorFn, seg = 44) {
  const g = new THREE.SphereGeometry(1, seg, Math.round(seg * 0.7));
  if (colorFn) {
    const pos = g.attributes.position, cols = new Float32Array(pos.count * 3), c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const r = colorFn(x, y, z, c);
      c.toArray(cols, i * 3);
      if (typeof r === "number") pos.setXYZ(i, x * r, y * r, z * r);
    }
    g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
  }
  g.scale(rx, ry, rz);
  g.computeVertexNormals();
  return shade(new THREE.Mesh(g, material));
}

/* Segment de patte le long de +X, du rayon r0 (proximal) à r1 (distal). */
function segment(len, r0, r1, material) {
  const g = new THREE.CylinderGeometry(r1, r0, len, 14, 3);
  g.rotateZ(-Math.PI / 2); g.translate(len / 2, 0, 0);
  const m = shade(new THREE.Mesh(g, material));
  m.add(shade(new THREE.Mesh(new THREE.SphereGeometry(r0 * 1.1, 14, 10), material)));
  return m;
}

function bristle(parent, at, dir, len, r, material) {
  const g = new THREE.CylinderGeometry(r * 0.12, r, len, 6);
  g.translate(0, len / 2, 0);
  const m = new THREE.Mesh(g, material);
  m.position.copy(at);
  m.quaternion.setFromUnitVectors(V(0, 1, 0), dir.clone().normalize());
  m.castShadow = true;
  parent.add(m);
}

/* ------------------------------------------------------------- textures */
function facetTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#4a0306"; g.fillRect(0, 0, 512, 512);
  const r = 6, h = Math.sqrt(3) * r;
  for (let col = -1; col * 1.5 * r < 512 + r; col++) {
    for (let row = -1; row * h < 512 + h; row++) {
      const x = col * 1.5 * r, y = row * h + (col & 1 ? h / 2 : 0);
      const gr = g.createRadialGradient(x - r * 0.25, y - r * 0.25, 0, x, y, r);
      gr.addColorStop(0, "#f04a3c"); gr.addColorStop(0.55, "#b8141c"); gr.addColorStop(1, "#6d060b");
      g.fillStyle = gr; g.beginPath();
      for (let k = 0; k < 6; k++) {
        const a = Math.PI / 3 * k;
        g.lineTo(x + Math.cos(a) * r * 0.92, y + Math.sin(a) * r * 0.92);
      }
      g.closePath(); g.fill();
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(2, 2); t.anisotropy = 4;
  return t;
}

// aile : x = envergure (0 → 2,1 mm), y = corde (costa côté y négatif)
const WX = [-0.05, 2.15], WY = [-0.5, 0.55];
const wingUV = (x, y) => [(x - WX[0]) / (WX[1] - WX[0]), 1 - (y - WY[0]) / (WY[1] - WY[0])];
function wingShape() {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.3, -0.12, 1.0, -0.42, 1.6, -0.40);
  s.bezierCurveTo(1.95, -0.38, 2.09, -0.18, 2.06, 0.02);
  s.bezierCurveTo(2.0, 0.28, 1.55, 0.47, 1.05, 0.43);
  s.bezierCurveTo(0.6, 0.39, 0.25, 0.26, 0.08, 0.1);
  s.lineTo(0, 0);
  return s;
}
function wingTexture() {
  const W = 1024, H = 512, c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d");
  const P = (x, y) => { const [u, v] = wingUV(x, y); return [u * W, (1 - v) * H]; };
  const path = pts => { g.beginPath(); pts.forEach(([x, y], i) => { const [a, b] = P(x, y); i ? g.lineTo(a, b) : g.moveTo(a, b); }); };
  const outline = wingShape().getPoints(80).map(p => [p.x, p.y]);
  path(outline); g.closePath();
  g.fillStyle = "rgba(210,220,235,0.26)"; g.fill();
  g.strokeStyle = "rgba(90,70,45,0.9)"; g.lineCap = g.lineJoin = "round";
  g.lineWidth = 7; path(outline.slice(0, 48)); g.stroke();              // costa
  g.lineWidth = 2.5; path(outline); g.closePath(); g.stroke();          // bord
  const veins = [
    [[0.12, -0.06], [0.7, -0.2], [1.2, -0.31], [1.58, -0.39]],             // L2
    [[0.12, 0.0], [0.8, -0.05], [1.5, -0.08], [2.04, -0.06]],               // L3
    [[0.16, 0.05], [0.8, 0.09], [1.45, 0.16], [1.98, 0.24]],                // L4
    [[0.2, 0.11], [0.7, 0.2], [1.1, 0.3], [1.52, 0.45]],                    // L5
    [[0.83, -0.05], [0.84, 0.09]], [[1.27, 0.17], [1.24, 0.33]],            // nervures transverses
  ];
  g.lineWidth = 4.5;
  for (const v of veins) { path(v); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}
function wingGeometry(side) {
  const g = new THREE.ShapeGeometry(wingShape(), 40);
  const pos = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, ...wingUV(pos.getX(i), pos.getY(i)));
  g.rotateX(-Math.PI / 2);                  // y -> -z : costa vers +z
  g.scale(0.85, 1, 0.85);
  if (side < 0) g.scale(1, 1, -1);          // aile droite en miroir
  return g;
}

/* ---------------------------------------------------------------- pattes */
const LEGS = {
  T1: { at: [0.11, -0.22, 0.27], foot: [0.55, 0.85], c: -1.15, L: [0.2, 0.48, 0.42, 0.46] },
  T2: { at: [0.14, -0.27, 0.02], foot: [0.9, 0.1], c: -1.25, L: [0.15, 0.52, 0.5, 0.5] },
  T3: { at: [0.12, -0.24, -0.22], foot: [0.72, -0.68], c: -1.2, L: [0.17, 0.58, 0.56, 0.54] },
};
const TARSUS_TILT = 0.5;   // angle du tarse avec le sol

function buildLeg(thorax, name, side, mat, dark) {
  const d = LEGS[name], [Lc, Lf, Lt, Ls] = d.L;
  const root = new THREE.Object3D(); root.position.set(side * d.at[0], d.at[1], d.at[2]);
  const yaw = new THREE.Object3D(), coxa = new THREE.Object3D(), femur = new THREE.Object3D();
  const tibia = new THREE.Object3D(), tarsus = new THREE.Object3D();
  thorax.add(root); root.add(yaw); yaw.add(coxa); coxa.add(femur); femur.add(tibia); tibia.add(tarsus);
  femur.position.x = Lc; tibia.position.x = Lf; tarsus.position.x = Lt;
  coxa.add(segment(Lc, 0.075, 0.058, mat));
  femur.add(segment(Lf, 0.054, 0.042, mat));
  tibia.add(segment(Lt, 0.039, 0.031, mat));
  tarsus.add(segment(Ls, 0.029, 0.017, mat));
  for (let k = 1; k < 5; k++) {       // tarsomères
    const b = shade(new THREE.Mesh(new THREE.SphereGeometry(0.024 - k * 0.002, 10, 8), mat));
    b.position.x = Ls * (0.36 + k * 0.15); tarsus.add(b);
  }
  for (const s of [-1, 1]) {           // griffes
    const cl = new THREE.Mesh(new THREE.ConeGeometry(0.008, 0.05, 6), dark);
    cl.rotation.z = -Math.PI / 2 - 0.5; cl.position.set(Ls + 0.012, -0.01, s * 0.012); tarsus.add(cl);
  }
  if (name === "T1") {                 // peigne sexuel du mâle
    for (let k = 0; k < 9; k++) {
      const tooth = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.03, 0.006), dark);
      tooth.position.set(0.05 + k * 0.009, -0.025, 0.018); tarsus.add(tooth);
    }
  }
  return { name, side, root, yaw, coxa, femur, tibia, tarsus, L: d.L, c: d.c,
           foot: V(side * d.foot[0], 0, d.foot[1]) };
}

const _p = V(0, 0, 0), _a = V(0, 0, 0), _h = V(0, 0, 0), _o = V(0, 0, 0);
function solveLeg(leg, footW) {
  const [Lc, Lf, Lt, Ls] = leg.L;
  const p = leg.root.worldToLocal(_p.copy(footW));
  const yaw = Math.atan2(-p.z, p.x);
  leg.yaw.rotation.y = yaw;
  leg.root.getWorldPosition(_h);
  _o.set(footW.x - _h.x, 0, footW.z - _h.z).normalize();
  _a.copy(footW).addScaledVector(_o, -Ls * Math.cos(TARSUS_TILT));
  _a.y += Ls * Math.sin(TARSUS_TILT);
  const a = leg.root.worldToLocal(_a);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const fu = p.x * cy - p.z * sy, fv = p.y, au = a.x * cy - a.z * sy, av = a.y;
  const Fx = Lc * Math.cos(leg.c), Fy = Lc * Math.sin(leg.c);
  const dx = au - Fx, dy = av - Fy;
  const d = Math.min(Math.max(Math.hypot(dx, dy), Math.abs(Lf - Lt) + 1e-3), Lf + Lt - 1e-3);
  const tf = Math.atan2(dy, dx) + Math.acos((Lf * Lf + d * d - Lt * Lt) / (2 * Lf * d));
  const Kx = Fx + Lf * Math.cos(tf), Ky = Fy + Lf * Math.sin(tf);
  const tt = Math.atan2(av - Ky, au - Kx);
  const Ax = Kx + Lt * Math.cos(tt), Ay = Ky + Lt * Math.sin(tt);
  const ts = Math.atan2(fv - Ay, fu - Ax);
  leg.coxa.rotation.z = leg.c; leg.femur.rotation.z = tf - leg.c;
  leg.tibia.rotation.z = tt - tf; leg.tarsus.rotation.z = ts - tt;
}

/* ------------------------------------------------------------------ mouche */
export const REST = {
  height: 0.6, pitch: 0.05, roll: 0, yaw: 0, headYaw: 0, headPitch: 0, headRoll: 0,
  abdPitch: -0.08, abdYaw: 0, abdRoll: 0, abd2Pitch: -0.06, wingSpreadL: 0, wingSpreadR: 0,
  wingElevL: 0, wingElevR: 0, proboscis: 0, antenna: 0, stance: 1,
};

export function createFly() {
  const root = new THREE.Group();
  const thorax = new THREE.Group(); root.add(thorax);
  const legMat = cuticle(COL.leg, { roughness: 0.5, clearcoat: 0.4 });
  const dark = new THREE.MeshStandardMaterial({ color: COL.bristle, roughness: 0.6 });

  // thorax : dessus plus sombre, flancs clairs
  const cThorax = col(COL.thorax), cPale = col(COL.pale), cTop = col(0x6f4a25);
  thorax.add(ellipsoid(0.34, 0.33, 0.47, cuticle(0xffffff, { vertexColors: true }), (x, y, z, c) => {
    c.copy(cThorax).lerp(cPale, THREE.MathUtils.smoothstep(-y, 0.1, 0.8) * 0.7);
    if (y > 0.5 && Math.abs(Math.abs(x) - 0.18) < 0.07) c.lerp(cTop, 0.5);   // bandes dorsales
    return 1 - 0.07 * Math.max(0, z);                                          // plus fin à l'avant
  }));
  const scut = ellipsoid(0.17, 0.09, 0.14, cuticle(COL.thorax));
  scut.position.set(0, 0.24, -0.36); thorax.add(scut);
  for (const s of [-1, 1]) {           // macrochètes
    for (const [x, z] of [[0.1, 0.1], [0.1, -0.13]]) {
      const y = 0.33 * Math.sqrt(Math.max(0, 1 - (x / 0.34) ** 2 - (z / 0.47) ** 2));
      bristle(thorax, V(s * x, y - 0.01, z), V(s * 0.1, 0.5, -1), 0.28, 0.011, dark);
    }
    bristle(thorax, V(s * 0.07, 0.29, -0.45), V(s * 0.35, 0.35, -1), 0.32, 0.011, dark);
    bristle(thorax, V(s * 0.31, 0.1, 0.1), V(s, 0.6, -0.6), 0.2, 0.01, dark);
    const hal = new THREE.Group(); hal.position.set(s * 0.26, 0.02, -0.28);
    const stalk = segment(0.12, 0.012, 0.01, legMat); stalk.rotation.z = -0.6; hal.add(stalk);
    const knob = shade(new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 10), cuticle(0xd8b98a)));
    knob.position.set(0.1, -0.08, 0); hal.add(knob);
    hal.rotation.y = s > 0 ? 0.4 : Math.PI - 0.4; thorax.add(hal);
  }

  // tête
  const neck = new THREE.Group(); neck.position.set(0, 0.09, 0.44); neck.rotation.order = "YXZ"; thorax.add(neck);
  neck.add(ellipsoid(0.11, 0.11, 0.1, cuticle(COL.thorax)));
  const cFrons = col(COL.frons), cHead = col(0xa87542);
  const head = ellipsoid(0.32, 0.3, 0.19, cuticle(0xffffff, { vertexColors: true }), (x, y, z, c) => {
    c.copy(cHead).lerp(cFrons, THREE.MathUtils.smoothstep(z, 0.2, 0.9));
  });
  head.position.set(0, 0, 0.18); neck.add(head);
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: facetTexture(), roughness: 0.28, clearcoat: 1,
    clearcoatRoughness: 0.12, bumpScale: 0.6 });
  eyeMat.bumpMap = eyeMat.map;
  for (const s of [-1, 1]) {
    const eye = ellipsoid(0.15, 0.25, 0.2, eyeMat);
    eye.position.set(s * 0.22, 0.02, 0.18); eye.rotation.y = s * 0.25; neck.add(eye);
  }
  for (const [x, y, z] of [[0, 0.3, 0.18], [0.045, 0.28, 0.12], [-0.045, 0.28, 0.12]]) {
    const oc = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8), cuticle(0x7a1010, { roughness: 0.2 }));
    oc.position.set(x, y, z); neck.add(oc);
  }
  for (const s of [-1, 1]) bristle(neck, V(s * 0.09, 0.27, 0.16), V(s * 0.3, 1, -0.6), 0.2, 0.009, dark);
  const antennae = [];
  for (const s of [-1, 1]) {
    const an = new THREE.Group(); an.position.set(s * 0.06, 0.09, 0.35); neck.add(an); antennae.push(an);
    const scape = shade(new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 10), cuticle(COL.frons)));
    an.add(scape);
    const funi = ellipsoid(0.045, 0.075, 0.04, cuticle(0xd29a5a));
    funi.position.set(s * 0.01, -0.08, 0.03); an.add(funi);
    const arista = new THREE.Group(); arista.position.set(s * 0.04, -0.05, 0.05);
    arista.rotation.set(0.2, s * 0.9, 0); an.add(arista);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.007, 0.28, 5), dark);
    shaft.rotation.z = -Math.PI / 2; shaft.position.x = 0.14; arista.add(shaft);
    for (let k = 0; k < 6; k++) {
      const br = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.003, 0.07, 4), dark);
      br.position.set(0.05 + k * 0.04, k & 1 ? 0.025 : -0.025, 0); br.rotation.z = k & 1 ? -0.6 : 0.6;
      arista.add(br);
    }
  }
  const probo = new THREE.Group(); probo.position.set(0, -0.18, 0.23); neck.add(probo);
  const rostrum = segment(0.2, 0.07, 0.055, cuticle(0xc9a070)); rostrum.rotation.z = -Math.PI / 2; probo.add(rostrum);
  const labellum = ellipsoid(0.08, 0.05, 0.07, cuticle(0xd8b489));
  labellum.position.set(0, -0.22, 0.01); probo.add(labellum);

  // abdomen en deux articulations, tergites cerclés de brun, bout noir (mâle)
  const abd1 = new THREE.Group(); abd1.position.set(0, -0.02, -0.36); abd1.rotation.order = "YXZ"; thorax.add(abd1);
  const abd2 = new THREE.Group(); abd2.position.set(0, -0.02, -0.44); abd2.rotation.order = "YXZ"; abd1.add(abd2);
  const cAbd = col(COL.abdomen), cBand = col(COL.band), cMale = col(COL.male);
  const abdMat = cuticle(0xffffff, { vertexColors: true });
  const tergites = (n, maleFrom) => (x, y, z, c) => {
    const t = (1 - z) / 2 * n, k = Math.floor(t), f = t - k;   // t : 0 à l'avant -> n à l'arrière
    const band = THREE.MathUtils.smoothstep(f, 0.6, 0.8);
    c.copy(cAbd).lerp(cBand, band * 0.9);
    if (k >= maleFrom) c.copy(cMale);
    c.lerp(cPale, THREE.MathUtils.smoothstep(-y, 0.15, 0.7) * (k >= maleFrom ? 0.25 : 0.85));
    return 1 - 0.035 * Math.exp(-(((f < 0.5 ? f : 1 - f) / 0.06) ** 2));   // sillons
  };
  const a1 = ellipsoid(0.3, 0.27, 0.34, abdMat, tergites(3, 99)); a1.position.z = -0.26; abd1.add(a1);
  const a2 = ellipsoid(0.26, 0.24, 0.29, abdMat, tergites(3, 1)); a2.position.z = -0.16; abd2.add(a2);
  const tip = ellipsoid(0.09, 0.09, 0.07, cuticle(COL.male)); tip.position.set(0, -0.07, -0.4); abd2.add(tip);

  // ailes
  const wingMat = new THREE.MeshPhysicalMaterial({ map: wingTexture(), transparent: true, side: THREE.DoubleSide,
    depthWrite: false, roughness: 0.18, iridescence: 1, iridescenceIOR: 1.35,
    iridescenceThicknessRange: [250, 650] });
  const wings = [-1, 1].map(side => {
    const yawN = new THREE.Object3D(); yawN.position.set(side * 0.18, 0.27, -0.05);
    const elevN = new THREE.Object3D(); yawN.add(elevN);
    const m = new THREE.Mesh(wingGeometry(side), wingMat); m.castShadow = true; m.renderOrder = 2;
    elevN.add(m); thorax.add(yawN);
    return { side, yawN, elevN };
  });

  const legs = [];
  for (const n of ["T1", "T2", "T3"]) for (const s of [-1, 1]) legs.push(buildLeg(thorax, n, s, legMat, dark));

  const _f = V(0, 0, 0);
  /* feet (optionnel) : 6 cibles de tarse en coordonnées monde, dans l'ordre de `legs`
     (marche, saut) ; sinon les pieds restent à leur place nominale sous la mouche. */
  function apply(p, feet) {
    root.rotation.y = p.yaw;
    thorax.position.y = p.height;
    thorax.rotation.set(p.pitch, 0, p.roll);
    neck.rotation.set(p.headPitch, p.headYaw, p.headRoll);
    abd1.rotation.set(p.abdPitch, p.abdYaw, p.abdRoll);
    abd2.rotation.set(p.abd2Pitch, p.abdYaw * 0.6, p.abdRoll * 0.5);
    for (const w of wings) {
      const spread = w.side > 0 ? p.wingSpreadL : p.wingSpreadR;
      const yl = Math.PI / 2 - 0.08 - spread * (Math.PI / 2 + 0.07);
      w.yawN.rotation.y = w.side > 0 ? yl : Math.PI - yl;
      w.elevN.rotation.z = 0.07 + spread * 0.12 + (w.side > 0 ? p.wingElevL : p.wingElevR);
    }
    probo.rotation.x = 0.35 - p.proboscis * 0.9;
    probo.scale.y = 1 + p.proboscis * 0.8;
    antennae.forEach((a, k) => { a.rotation.x = -p.antenna; a.rotation.z = (k ? 1 : -1) * p.antenna * 0.4; });
    root.updateMatrixWorld(true);
    legs.forEach((leg, k) => {
      if (feet && feet[k]) return solveLeg(leg, feet[k]);
      _f.set(leg.foot.x * p.stance, 0, leg.foot.z);
      solveLeg(leg, root.localToWorld(_f));
    });
  }
  apply(REST);
  return { root, apply, legs };
}
