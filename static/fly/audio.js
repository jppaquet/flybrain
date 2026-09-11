/* audio.js - sources (fichier, micro, beat de démo) et analyse temps réel :
   niveau, détection de silence, attaques (flux spectral), tempo (autocorrélation)
   et phase du temps (boucle à verrouillage de phase sur les coups de grosse caisse). */

export class AudioEngine {
  constructor() {
    this.ctx = null; this.mode = "none";
    this.threshold = -50;          // dBFS : en dessous, c'est le silence
    this.hold = 0.35;              // s de silence avant l'arrêt
    this.phase = 0; this.bpm = 0; this.conf = 0; this.beat = 0;
    this.active = false; this.lastLoud = -1e9; this.level = 0; this.db = -100;
    this.flux = []; this.hist = []; this.onsets = []; this.lastOnset = -1; this.lastTempo = 0; this.prevT = null;
  }

  ensure() {
    if (!this.ctx) {
      const ctx = this.ctx = new AudioContext();
      // toutes les sources entrent par `input` : un analyseur court (attaques, 43 ms)
      // et un long pour le niveau (170 ms, robuste aux creux entre deux coups)
      this.input = ctx.createGain();
      this.an = ctx.createAnalyser(); this.an.fftSize = 2048; this.an.smoothingTimeConstant = 0;
      this.anL = ctx.createAnalyser(); this.anL.fftSize = 8192;
      this.input.connect(this.an); this.input.connect(this.anL);
      this.out = ctx.createGain(); this.out.connect(ctx.destination);
      this.td = new Float32Array(this.anL.fftSize);
      this.fd = new Float32Array(this.an.frequencyBinCount);
      this.prev = new Float32Array(this.an.frequencyBinCount);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  detach() {
    this.demo?.stop(); this.demo = null;
    if (this.el) this.el.pause();
    this.elNode?.disconnect();
    this.micStream?.getTracks().forEach(t => t.stop()); this.micStream = null;
    this.micNode?.disconnect(); this.micNode = null;
    this.mode = "none";
  }

  async useFile(file) {
    const ctx = this.ensure(); this.detach();
    if (!this.elNode) { this.el = this.el || new Audio(); this.elNode = ctx.createMediaElementSource(this.el); }
    if (this.url) URL.revokeObjectURL(this.url);
    this.el.src = this.url = URL.createObjectURL(file);
    this.elNode.connect(this.input); this.elNode.connect(this.out);
    this.mode = "file"; this.fileName = file.name; this.bpm = 0;
    await this.el.play();
    return this.el;
  }

  async useMic() {
    const ctx = this.ensure(); this.detach();
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    this.micNode = ctx.createMediaStreamSource(this.micStream);
    this.micNode.connect(this.input);              // pas vers les haut-parleurs (larsen)
    this.mode = "mic"; this.bpm = 0;
  }

  useDemo(opts) {
    const ctx = this.ensure(); this.detach();
    this.demo = new DemoBeat(ctx, opts);
    this.demo.out.connect(this.input); this.demo.out.connect(this.out);
    this.demo.start(); this.mode = "demo"; this.bpm = 0;
  }

  stop() { this.detach(); }
  setVolume(v) { this.ensure(); this.out.gain.value = v; }

  /* À appeler à chaque image. */
  update() {
    if (!this.ctx) return { t: 0, dt: 0, db: -100, level: 0, active: false, phase: this.phase, bpm: 0,
                            conf: 0, onset: false, kick: false, beat: false, downbeat: false, mode: "none" };
    const t = this.ctx.currentTime;
    const dt = this.prevT == null ? 0 : Math.min(0.1, Math.max(0, t - this.prevT));
    this.prevT = t;
    let db = -100, flux = 0, bass = 0;
    if (this.ctx && this.mode !== "none") {
      this.anL.getFloatTimeDomainData(this.td);
      let s = 0; for (let i = 0; i < this.td.length; i++) s += this.td[i] * this.td[i];
      db = 10 * Math.log10(s / this.td.length + 1e-12);
      this.an.getFloatFrequencyData(this.fd);
      const hz = this.ctx.sampleRate / this.an.fftSize;
      const k0 = Math.max(1, Math.round(30 / hz)), kb = Math.round(160 / hz), k1 = Math.round(5000 / hz);
      for (let k = k0; k < k1; k++) {
        const c = Math.log1p(100 * Math.pow(10, this.fd[k] / 20));
        const d = c - this.prev[k]; this.prev[k] = c;
        if (d > 0) { if (k < kb) { bass += 3 * d; flux += 3 * d; } else flux += d; }
      }
    }
    this.db = db;

    // silence : seuil avec hystérésis de 3 dB et temps de maintien
    if (db > this.threshold + (this.active ? 0 : 3)) this.lastLoud = t;
    this.active = t - this.lastLoud < this.hold;
    const lv = Math.min(1, Math.max(0, (db - this.threshold) / 30));
    this.level += (lv - this.level) * Math.min(1, dt * 8);

    // attaques : pic local du flux au-dessus de 1,5 x sa moyenne récente
    const F = this.flux; F.push([t, flux, bass]);
    while (F.length && F[0][0] < t - 8) F.shift();
    let onset = false, kick = false;
    const n = F.length;
    if (n > 3 && this.active) {
      const [tp, fp, bp] = F[n - 2];
      let m = 0, c = 0;
      for (let i = n - 1; i >= 0 && F[i][0] > t - 0.6; i--) { m += F[i][1]; c++; }
      m /= c;
      if (fp > F[n - 3][1] && fp >= flux && fp > m * 1.5 + 1e-3 && tp - this.lastOnset > 0.1) {
        onset = true; kick = bp > 0.5 * fp; this.lastOnset = tp; this.onsets.push([tp, kick]);
      }
    }
    while (this.onsets.length && this.onsets[0][0] < t - 6) this.onsets.shift();
    this.hist.push([t, db]);
    while (this.hist.length && this.hist[0][0] < t - 6) this.hist.shift();

    if (t - this.lastTempo > 0.5 && n > 30 && F[n - 1][0] - F[0][0] > 3) { this.lastTempo = t; this.tempo(); }

    // phase du temps : avance au tempo, recalée sur les coups de grosse caisse
    const bpm = this.bpm || 110;
    this.phase += dt * bpm / 60;
    if (kick) {
      const at = this.phase - (t - this.lastOnset) * bpm / 60;
      const e = at - Math.round(at);
      if (!this.bpm) this.phase -= e;
      else if (Math.abs(e) < 0.2) this.phase -= 0.25 * e;
    }
    const b = Math.floor(this.phase), beat = b !== this.beat && this.active;
    this.beat = b;
    return { t, dt, db, level: this.level, active: this.active, phase: this.phase, bpm: this.bpm,
             conf: this.conf, onset, kick, beat, downbeat: beat && ((b % 4) + 4) % 4 === 0, mode: this.mode };
  }

  tempo() {
    const F = this.flux, R = 100, t0 = F[0][0], N = Math.floor((F[F.length - 1][0] - t0) * R);
    const x = new Float32Array(N);
    for (let i = 0, j = 0; i < N; i++) {
      const tt = t0 + i / R;
      while (j < F.length - 2 && F[j + 1][0] < tt) j++;
      const [ta, fa] = F[j], [tb, fb] = F[j + 1], f = tb > ta ? (tt - ta) / (tb - ta) : 0;
      x[i] = fa + (fb - fa) * Math.min(1, Math.max(0, f));
    }
    let mean = 0; for (const v of x) mean += v; mean /= N;
    let e0 = 0; for (let i = 0; i < N; i++) { x[i] -= mean; e0 += x[i] * x[i]; }
    if (e0 <= 0) return;
    e0 /= N;
    const Lmin = Math.floor(R * 60 / 180), Lmax = Math.ceil(R * 60 / 65), acf = new Float32Array(Lmax + 2);
    for (let L = Lmin - 1; L <= Lmax + 1; L++) {
      let s = 0; for (let i = L; i < N; i++) s += x[i] * x[i - L];
      acf[L] = s / (N - L);
    }
    let best = -Infinity, bl = 0;
    for (let L = Lmin; L <= Lmax; L++) {
      if (acf[L] < acf[L - 1] || acf[L] < acf[L + 1]) continue;
      const w = Math.exp(-0.5 * (Math.log2(60 * R / L / 110) / 0.7) ** 2);
      if (acf[L] * w > best) { best = acf[L] * w; bl = L; }
    }
    if (!bl) return;
    const y0 = acf[bl - 1], y1 = acf[bl], y2 = acf[bl + 1], den = y0 - 2 * y1 + y2;
    const Lr = bl + (den ? 0.5 * (y0 - y2) / den : 0);
    this.conf = acf[bl] / e0;
    if (this.conf < 0.08) return;
    const nb = 60 * R / Lr;
    this.bpm = this.bpm && Math.abs(nb - this.bpm) / this.bpm < 0.06 ? this.bpm * 0.7 + nb * 0.3 : nb;
  }
}

/* Beat de démo synthétisé (dembow ~98 BPM). Une mesure sur huit est silencieuse,
   pour voir la mouche s'arrêter puis repartir. */
export class DemoBeat {
  constructor(ctx, { bpm = 98, breaks = true } = {}) {
    this.ctx = ctx; this.bpm = bpm; this.breaks = breaks;
    this.out = ctx.createGain(); this.out.gain.value = 0.7;
    const len = ctx.sampleRate, buf = this.noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0); for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }
  start() { this.step = 0; this.next = this.ctx.currentTime + 0.06; this.timer = setInterval(() => this.tick(), 25); }
  stop() {
    clearInterval(this.timer);
    this.out.gain.setTargetAtTime(0, this.ctx.currentTime, 0.02);
    setTimeout(() => this.out.disconnect(), 300);
  }
  tick() {
    const s16 = 60 / this.bpm / 4;
    while (this.next < this.ctx.currentTime + 0.12) { this.play(this.step, this.next); this.next += s16; this.step++; }
  }
  play(step, t) {
    const bar = Math.floor(step / 16), s = step % 16;
    if (this.breaks && bar % 8 === 7) return;
    if (s % 4 === 0) this.kick(t);
    if (s === 3 || s === 6 || s === 11 || s === 14) this.snare(t);
    if (s % 2 === 0) this.noiseHit(t, "highpass", 7500, 0.7, s % 4 === 2 ? 0.12 : 0.05, 0.04);
    const root = [55, 43.65, 49, 41.2][bar % 4];
    if (s === 0 || s === 8) this.bass(t, root, 60 / this.bpm * 1.5);
    if (s === 14) this.bass(t, root * 1.5, 60 / this.bpm * 0.4);
  }
  env(g, t, a, peak, decay) {
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + decay);
  }
  kick(t) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(42, t + 0.12);
    this.env(g, t, 0.003, 1, 0.32); o.connect(g).connect(this.out); o.start(t); o.stop(t + 0.4);
  }
  snare(t) {
    this.noiseHit(t, "bandpass", 1900, 0.8, 0.5, 0.15);
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    o.type = "triangle"; o.frequency.value = 185;
    this.env(g, t, 0.002, 0.25, 0.08); o.connect(g).connect(this.out); o.start(t); o.stop(t + 0.15);
  }
  noiseHit(t, type, freq, q, peak, decay) {
    const c = this.ctx, n = c.createBufferSource(), f = c.createBiquadFilter(), g = c.createGain();
    n.buffer = this.noise; f.type = type; f.frequency.value = freq; f.Q.value = q;
    this.env(g, t, 0.002, peak, decay);
    n.connect(f).connect(g).connect(this.out); n.start(t, Math.random() * 0.5); n.stop(t + decay + 0.05);
  }
  bass(t, freq, dur) {
    const c = this.ctx, o = c.createOscillator(), f = c.createBiquadFilter(), g = c.createGain();
    o.type = "sawtooth"; o.frequency.value = freq; f.type = "lowpass"; f.frequency.value = 380; f.Q.value = 4;
    this.env(g, t, 0.01, 0.35, dur); o.connect(f).connect(g).connect(this.out); o.start(t); o.stop(t + dur + 0.05);
  }
}
