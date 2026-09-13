/* charts.js - canvas components of the dashboard (no dependency). */
const Charts = (() => {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--other"];
  const groupColor = g => css(SLOTS[Math.min(g, 7)]);
  const isDark = () => css("color-scheme") === "dark";
  const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

  function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
  function rampLUT() {
    const stops = (isDark() ? [...RAMP].reverse() : RAMP).map(hexRgb);
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const x = i / 255 * (stops.length - 1), k = Math.min(Math.floor(x), stops.length - 2), f = x - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
    }
    return lut;
  }
  function rampCSS() {
    const r = isDark() ? [...RAMP].reverse() : RAMP;
    return `linear-gradient(90deg, ${r.join(",")})`;
  }

  function setup(canvas) {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height, dpr };
  }

  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  function fmt(v, d = 1) {
    if (v == null || isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, "") + " M";
    if (a >= 1e4) return (v / 1e3).toFixed(0) + " k";
    if (a >= 1000) return Math.round(v).toLocaleString("en-US");
    if (a >= 100 || Number.isInteger(v)) return String(Math.round(v));
    return v.toFixed(d);
  }

  /* ---------------------------------------------------------------- tooltip */
  const tip = () => document.getElementById("tooltip");
  function showTip(rows, x, y) {
    const el = tip(); el.replaceChildren();
    for (const r of rows) {
      const d = document.createElement("div");
      d.className = r.cls || "trow";
      if (r.color) { const k = document.createElement("span"); k.className = "ln"; k.style.background = r.color; d.append(k); }
      if (r.value != null) { const v = document.createElement("span"); v.className = "tv"; v.textContent = r.value; d.append(v); }
      if (r.label != null) { const l = document.createElement("span"); l.className = "tl"; l.textContent = r.label; d.append(l); }
      el.append(d);
    }
    el.hidden = false;
    const b = el.getBoundingClientRect();
    el.style.left = Math.min(x + 14, innerWidth - b.width - 8) + "px";
    el.style.top = Math.min(y + 14, innerHeight - b.height - 8) + "px";
  }
  const hideTip = () => { tip().hidden = true; };

  function stimBands(ctx, windows, x0, xs, y0, h) {
    ctx.fillStyle = css("--surface-2");
    for (const [a, b] of windows) ctx.fillRect(x0 + a * xs, y0, Math.max(1, (b - a) * xs), h);
  }

  /* ---------------------------------------------------------- small multiples
     panels: [{label, sub, color, values}]; one y scale per panel.            */
  class Multiples {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.opts = opts; this.data = null; this.cur = null; this.hover = null;
      canvas.addEventListener("pointermove", e => this.onMove(e));
      canvas.addEventListener("pointerleave", () => { this.hover = null; hideTip(); this.draw(); });
    }
    set(d) { this.data = d; this.draw(); }
    cursor(t) { this.cur = t; this.draw(); }
    layout(w) {
      const n = this.data.panels.length, cols = this.opts.cols ? this.opts.cols(w, n) : (w > 520 ? Math.min(n, 4) : Math.min(n, 2));
      return { cols, rows: Math.ceil(n / cols) };
    }
    draw() {
      const { ctx, w, h } = setup(this.c);
      ctx.clearRect(0, 0, w, h);
      if (!this.data) return;
      const { panels, bin_ms, T, windows } = this.data;
      const { cols, rows } = this.layout(w);
      const gap = 14, pw = (w - gap * (cols - 1)) / cols, ph = (h - gap * (rows - 1) - 14) / rows;
      this.geo = { cols, pw, ph, gap };
      panels.forEach((p, k) => {
        const px = (k % cols) * (pw + gap), py = Math.floor(k / cols) * (ph + gap);
        const top = py + 30, bot = py + ph, left = px, right = px + pw - 2;
        const xs = (right - left) / T;
        const mx = niceMax(Math.max(...p.values, 1e-9));
        ctx.font = "600 12px system-ui, sans-serif"; ctx.fillStyle = css("--ink"); ctx.textBaseline = "top";
        ctx.fillText(p.label, left, py + 1, pw - 60);
        ctx.font = "11.5px system-ui, sans-serif"; ctx.fillStyle = css("--ink-2");
        ctx.fillText(p.sub || "", left, py + 15, pw - 70);
        ctx.textAlign = "right"; ctx.fillStyle = css("--muted");
        ctx.fillText(`max ${fmt(mx)} Hz`, right, py + 15);
        ctx.textAlign = "left";
        stimBands(ctx, windows, left, xs, top, bot - top);
        ctx.fillStyle = css("--grid"); ctx.fillRect(left, top, right - left, 1);
        ctx.fillStyle = css("--axis"); ctx.fillRect(left, bot, right - left, 1);
        const yv = v => bot - (v / mx) * (bot - top);
        const xv = i => left + (i + 0.5) * bin_ms * xs;
        ctx.beginPath(); ctx.moveTo(xv(0), bot);
        p.values.forEach((v, i) => ctx.lineTo(xv(i), yv(v)));
        ctx.lineTo(xv(p.values.length - 1), bot); ctx.closePath();
        ctx.globalAlpha = 0.1; ctx.fillStyle = p.color; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath(); p.values.forEach((v, i) => i ? ctx.lineTo(xv(i), yv(v)) : ctx.moveTo(xv(i), yv(v)));
        ctx.lineWidth = 2; ctx.lineJoin = ctx.lineCap = "round"; ctx.strokeStyle = p.color; ctx.stroke();
        const ct = this.hover != null ? this.hover : this.cur;
        if (ct != null) {
          ctx.fillStyle = css("--ink-2"); ctx.fillRect(Math.round(left + ct * xs), top, 1, bot - top);
        }
        if (Math.floor(k / cols) === rows - 1) {
          ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "top";
          ctx.fillText("0", left, bot + 3); ctx.textAlign = "right";
          ctx.fillText(`${fmt(T)} ms`, right, bot + 3); ctx.textAlign = "left";
        }
      });
    }
    onMove(e) {
      if (!this.data || !this.geo) return;
      const r = this.c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const { cols, pw, gap } = this.geo;
      const col = Math.min(cols - 1, Math.floor(x / (pw + gap)));
      const lx = x - col * (pw + gap);
      const { bin_ms, T, panels } = this.data;
      const i = Math.max(0, Math.min(panels[0].values.length - 1, Math.floor(lx / (pw - 2) * T / bin_ms)));
      this.hover = (i + 0.5) * bin_ms;
      this.draw();
      const rows = [{ cls: "tl", label: `${fmt(i * bin_ms)}–${fmt((i + 1) * bin_ms)} ms` }];
      for (const p of panels) rows.push({ color: p.color, value: `${fmt(p.values[i])} Hz`, label: p.label });
      showTip(rows, e.clientX, e.clientY);
    }
  }

  /* ------------------------------------------------------------------ raster */
  class Raster {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.opts = opts; this.d = null; this.cur = null;
      canvas.addEventListener("pointermove", e => this.onMove(e));
      canvas.addEventListener("pointerleave", hideTip);
      canvas.addEventListener("click", e => { const h = this.hit(e); if (h && opts.onPick) opts.onPick(h.idx); });
    }
    set(d) { this.d = d; this.draw(); }
    cursor(t) { this.cur = t; this.draw(); }
    draw() {
      const { ctx, w, h, dpr } = setup(this.c);
      ctx.clearRect(0, 0, w, h);
      const d = this.d; if (!d) return;
      const G = 112, left = G, right = w - 4, top = 4, bot = h - 20;
      const n = d.rows.length, rh = (bot - top) / Math.max(n, 1), xs = (right - left) / d.T;
      this.geo = { left, right, top, bot, rh, xs };
      stimBands(ctx, d.windows, left, xs, top, bot - top);
      // spikes via ImageData (600 k ticks)
      const X0 = Math.round(left * dpr), Y0 = Math.round(top * dpr);
      const PW = Math.max(1, Math.round((right - left) * dpr)), PH = Math.max(1, Math.round((bot - top) * dpr));
      const img = ctx.getImageData(X0, Y0, PW, PH), px = img.data;
      const cols = Array.from({ length: 10 }, (_, k) => hexRgb(k === 9 ? css("--ink") : groupColor(k)));
      const rhp = rh * dpr, tall = Math.max(1, Math.round(rhp * 0.8));
      for (let s = 0; s < d.spk_t.length; s++) {
        const row = d.spk_row[s], x = Math.floor(d.spk_t[s] * xs * dpr);
        if (x < 0 || x >= PW) continue;
        const c = cols[d.rowGroup[row]], y0 = Math.floor(row * rhp);
        for (let y = y0; y < Math.min(PH, y0 + tall); y++) {
          const o = (y * PW + x) * 4; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
        }
      }
      ctx.putImageData(img, X0, Y0);
      // blocks (stimulated / readouts / groups) in the gutter
      ctx.font = "11.5px system-ui, sans-serif"; ctx.textBaseline = "middle";
      let start = 0;
      for (let r = 1; r <= n; r++) {
        if (r < n && d.blockKey[r] === d.blockKey[start]) continue;
        const y0 = top + start * rh, y1 = top + r * rh;
        if (start > 0) { ctx.fillStyle = css("--axis"); ctx.fillRect(left - 6, Math.round(y0), right - left + 6, 1); }
        if (y1 - y0 >= 11) {
          ctx.fillStyle = d.blockKey[start] < 10 ? css("--ink") : css("--ink-2");
          ctx.fillText(d.blockLabel[start], 0, (y0 + y1) / 2, G - 12);
        }
        start = r;
      }
      ctx.fillStyle = css("--axis"); ctx.fillRect(left, bot, right - left, 1);
      ctx.fillStyle = css("--muted"); ctx.textBaseline = "top"; ctx.font = "11px system-ui, sans-serif";
      const step = niceMax(d.T / 6);
      for (let t = 0; t <= d.T + 1e-6; t += step) {
        ctx.textAlign = t === 0 ? "left" : t + step > d.T + 1e-6 ? "right" : "center";
        ctx.fillText(`${fmt(t)}${t + step > d.T + 1e-6 ? " ms" : ""}`, left + t * xs, bot + 4);
      }
      ctx.textAlign = "left";
      if (this.cur != null) { ctx.fillStyle = css("--ink-2"); ctx.fillRect(Math.round(left + this.cur * xs), top, 1, bot - top); }
    }
    hit(e) {
      if (!this.d || !this.geo) return null;
      const r = this.c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const g = this.geo;
      if (x < g.left || x > g.right || y < g.top || y > g.bot) return null;
      const row = Math.min(this.d.rows.length - 1, Math.floor((y - g.top) / g.rh));
      return { row, idx: this.d.rows[row], t: (x - g.left) / g.xs };
    }
    onMove(e) {
      const h = this.hit(e);
      if (!h) return hideTip();
      const info = this.opts.describe ? this.opts.describe(h.idx) : [];
      showTip([{ cls: "tl", label: `${fmt(h.t)} ms` }, ...info], e.clientX, e.clientY);
    }
  }

  /* --------------------------------------------------------------- brain map */
  const VIEWS = { dorsal: [0, 2, false], frontal: [0, 1, false], lateral: [2, 1, false] };
  class BrainMap {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.opts = opts; this.meta = null; this.value = null;
      this.view = "dorsal"; this.isolate = null; this.sel = null; this.stim = null;
      canvas.addEventListener("pointermove", e => this.onMove(e));
      canvas.addEventListener("pointerleave", hideTip);
      canvas.addEventListener("click", e => { const i = this.nearest(e); if (i >= 0 && opts.onPick) opts.onPick(i); });
    }
    setMeta(m) { this.meta = m; this.project(); }
    setView(v) { this.view = v; this.project(); }
    setValue(val, vmax) { this.value = val; this.vmax = vmax; this.draw(); }
    setIsolate(g) { this.isolate = g; this.project(); }
    select(i) { this.sel = i; this.draw(); }
    project() {
      if (!this.meta) return;
      const { ctx, w, h, dpr } = setup(this.c);
      const [a, b] = VIEWS[this.view], P = this.meta.pos, N = this.meta.N;
      let mina = Infinity, maxa = -Infinity, minb = Infinity, maxb = -Infinity;
      for (let i = 0; i < N; i++) {
        const u = P[i * 3 + a], v = P[i * 3 + b];
        if (u < mina) mina = u; if (u > maxa) maxa = u; if (v < minb) minb = v; if (v > maxb) maxb = v;
      }
      const pad = 10, s = Math.min((w - 2 * pad) / (maxa - mina), (h - 2 * pad) / (maxb - minb));
      const ox = (w - s * (maxa - mina)) / 2, oy = (h - s * (maxb - minb)) / 2;
      this.px = new Int32Array(N); this.py = new Int32Array(N);
      for (let i = 0; i < N; i++) {
        this.px[i] = Math.round((ox + (P[i * 3 + a] - mina) * s) * dpr);
        this.py[i] = Math.round((oy + (P[i * 3 + b] - minb) * s) * dpr);
      }
      this.scale = s; this.span = [maxa - mina, maxb - minb];
      // spatial grid for hovering (8 css px cells)
      const cs = 8 * dpr; this.cs = cs; this.gw = Math.ceil(w * dpr / cs) + 1;
      const cells = new Map();
      for (let i = 0; i < N; i++) {
        const k = Math.floor(this.py[i] / cs) * this.gw + Math.floor(this.px[i] / cs);
        let l = cells.get(k); if (!l) cells.set(k, l = []); l.push(i);
      }
      this.cells = cells;
      this.draw();
    }
    draw() {
      if (!this.meta || !this.px) return;
      const { ctx, w, h, dpr } = setup(this.c);
      const W = this.c.width, H = this.c.height;
      ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, w, h);
      const img = ctx.getImageData(0, 0, W, H), d = img.data;
      const N = this.meta.N, grp = this.meta.group, iso = this.isolate;
      const idle = hexRgb(css("--idle")), dim = hexRgb(css("--grid"));
      const put = (x, y, c, r) => {
        for (let yy = y; yy < y + r; yy++) {
          if (yy < 0 || yy >= H) continue;
          for (let xx = x; xx < x + r; xx++) {
            if (xx < 0 || xx >= W) continue;
            const o = (yy * W + xx) * 4; d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
          }
        }
      };
      const r0 = Math.max(1, Math.round(dpr));
      for (let i = 0; i < N; i++) put(this.px[i], this.py[i], iso != null && grp[i] !== iso ? dim : idle, r0);
      const val = this.value;
      if (val) {
        const lut = rampLUT(), act = [];
        for (let i = 0; i < N; i++) if (val[i] > 0 && (iso == null || grp[i] === iso)) act.push(i);
        act.sort((p, q) => val[p] - val[q]);
        const lm = Math.log1p(this.vmax || 1), r1 = Math.max(2, Math.round(2 * dpr));
        const c = [0, 0, 0];
        for (const i of act) {
          const k = Math.min(255, Math.round(Math.log1p(val[i]) / lm * 255)) * 3;
          c[0] = lut[k]; c[1] = lut[k + 1]; c[2] = lut[k + 2];
          put(this.px[i] - (r1 >> 1), this.py[i] - (r1 >> 1), c, r1);
        }
      }
      ctx.putImageData(img, 0, 0);
      if (this.sel != null) {
        const x = this.px[this.sel] / dpr, y = this.py[this.sel] / dpr;
        ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.lineWidth = 4; ctx.strokeStyle = css("--surface"); ctx.stroke();
        ctx.lineWidth = 2; ctx.strokeStyle = css("--ink"); ctx.stroke();
      }
      // 100 µm scale bar
      const bar = 100 * this.scale;
      ctx.fillStyle = css("--ink-2"); ctx.fillRect(w - 16 - bar, h - 14, bar, 2);
      ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
      ctx.fillText("100 µm", w - 16, h - 18); ctx.textAlign = "left";
    }
    nearest(e) {
      if (!this.cells) return -1;
      const r = this.c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      const x = (e.clientX - r.left) * dpr, y = (e.clientY - r.top) * dpr, cs = this.cs;
      const cx = Math.floor(x / cs), cy = Math.floor(y / cs);
      let best = -1, bd = (12 * dpr) ** 2, bestAct = -1, bda = (12 * dpr) ** 2;
      for (let j = cy - 2; j <= cy + 2; j++) for (let i = cx - 2; i <= cx + 2; i++) {
        const l = this.cells.get(j * this.gw + i); if (!l) continue;
        for (const n of l) {
          if (this.isolate != null && this.meta.group[n] !== this.isolate) continue;
          const dd = (this.px[n] - x) ** 2 + (this.py[n] - y) ** 2;
          if (dd < bd) { bd = dd; best = n; }
          if (this.value && this.value[n] > 0 && dd < bda) { bda = dd; bestAct = n; }
        }
      }
      return bestAct >= 0 ? bestAct : best;   // prefer an active neuron
    }
    onMove(e) {
      const i = this.nearest(e);
      if (i < 0) return hideTip();
      showTip(this.opts.describe ? this.opts.describe(i) : [{ label: String(i) }], e.clientX, e.clientY);
    }
  }

  return { Multiples, Raster, BrainMap, groupColor, rampCSS, fmt, css, showTip, hideTip };
})();
