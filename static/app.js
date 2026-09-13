/* app.js - control panel, launching and tracking of simulations. */
(() => {
  const $ = s => document.querySelector(s);
  const { fmt } = Charts;
  const FIELDS = [["type", "type"], ["instance", "instance"], ["superclass", "superclass"],
                  ["class", "class"], ["any", "any"], ["body", "bodyId"]];
  const GENERAL = [["t_ms", "Duration (ms)", 50], ["trials", "Trials", 1], ["seed", "Seed", 1]];
  const MODEL = [["dt", "dt (ms)", 0.05], ["tau_m", "τ mem. (ms)", 1], ["tau_syn", "τ syn. (ms)", 0.5],
                 ["v_th", "Threshold (mV)", 0.5], ["refrac", "Refract. (ms)", 0.1], ["delay", "Delay (ms)", 0.1],
                 ["w_syn", "mV / synapse", 0.025], ["w_scale", "Weight scale", 0.1], ["f_poi", "Poisson weight", 10]];
  const DEFAULT = {
    stims: [{ query: "^LC4$", field: "type", hz: 150, t_on: 0, t_off: "", label: "LC4" }],
    readouts: [{ query: "^DNp01$", field: "type", label: "Giant fiber DNp01" },
               { query: "^descending", field: "superclass", label: "Descending neurons" }],
    silence: { query: "", field: "type" },
    params: { model: "shiu", t_ms: 500, trials: 1, seed: 0, w_scale: 1 },
  };
  let cfg, META;

  const el = (tag, props = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "class") e.className = v; else if (k.includes("-")) e.setAttribute(k, v); else e[k] = v;
    }
    for (const c of kids) if (c != null) e.append(c);
    return e;
  };
  function loadCfg() {
    try { const s = JSON.parse(localStorage.getItem("flybrain-cfg")); if (s && Array.isArray(s.stims)) return s; } catch {}
    return structuredClone(DEFAULT);
  }
  const save = () => { try { localStorage.setItem("flybrain-cfg", JSON.stringify(cfg)); } catch {} };
  const status = (t, bad) => { $("#status").textContent = t; $("#status").style.color = bad ? "var(--critical)" : ""; };

  async function unpack(resp) {
    const buf = await resp.arrayBuffer();
    const hl = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const T = { f4: Float32Array, f8: Float64Array, i4: Int32Array, u1: Uint8Array, u2: Uint16Array, i2: Int16Array };
    const arr = {};
    for (const s of header.arrays) arr[s.name] = new T[s.dtype](buf, 4 + hl + s.offset, s.length);
    return { header, arr };
  }

  /* ------------------------------------------------------------ search */
  const cache = new Map();
  function countInto(box, query, field) {
    clearTimeout(box._t);
    box.classList.remove("bad");
    if (!query.trim()) { box.textContent = ""; return; }
    box._t = setTimeout(async () => {
      const key = field + " " + query;
      let d = cache.get(key);
      if (!d) {
        const r = await fetch(`/api/search?q=${encodeURIComponent(query)}&field=${field}`);
        d = await r.json();
        if (r.ok) cache.set(key, d);
      }
      if (d.error) { box.textContent = "invalid pattern"; box.classList.add("bad"); return; }
      box.classList.toggle("bad", d.count === 0);
      box.textContent = d.count === 0 ? "no neuron"
        : `${fmt(d.count)} neuron${d.count > 1 ? "s" : ""} · ` +
          d.by_type.slice(0, 4).map(([t, n]) => `${t} (${n})`).join(", ") + (d.by_type.length > 4 ? "…" : "");
    }, 250);
  }

  /* ------------------------------------------------------------ controls */
  function card(item, kind, onRemove) {
    const count = el("div", { class: "count" });
    const refresh = () => { save(); countInto(count, item.query, item.field); };
    const q = el("input", { value: item.query, placeholder: "regex, e.g. ^LC4$", spellcheck: false, "aria-label": "Pattern",
                            oninput: e => { item.query = e.target.value; refresh(); } });
    const f = el("select", { "aria-label": "Field", onchange: e => { item.field = e.target.value; refresh(); } },
      ...FIELDS.map(([v, l]) => el("option", { value: v, textContent: l, selected: item.field === v })));
    const div = el("div", { class: "grp" });
    if (kind === "readout")
      div.append(el("input", { class: "lbl", value: item.label || "", placeholder: "Label", "aria-label": "Label",
                               oninput: e => { item.label = e.target.value; save(); } }));
    div.append(el("div", { class: "q" }, q, f,
      onRemove ? el("button", { class: "x", textContent: "✕", "aria-label": "Remove", onclick: onRemove }) : null), count);
    if (kind === "stim") {
      const num = (label, k, ph) => el("label", { class: "field" }, label,
        el("input", { type: "number", min: 0, value: item[k], placeholder: ph || "",
                      oninput: e => { item[k] = e.target.value === "" ? "" : +e.target.value; save(); } }));
      div.append(el("div", { class: "nums" }, num("Freq. (Hz)", "hz"), num("Start (ms)", "t_on"), num("End (ms)", "t_off", "end")));
    }
    refresh();
    return div;
  }

  function renderControls() {
    $("#stims").replaceChildren(...cfg.stims.map((s, k) => card(s, "stim", () => {
      cfg.stims.splice(k, 1); save(); renderControls();
    })));
    $("#readouts").replaceChildren(...cfg.readouts.map((r, k) => card(r, "readout", () => {
      cfg.readouts.splice(k, 1); save(); renderControls();
    })));
    $("#silence").replaceChildren(card(cfg.silence, "silence", null));
    $("#add-ro").disabled = $("#ro-preset").disabled = cfg.readouts.length >= 4;
    renderParams();
  }

  function renderParams() {
    const model = cfg.params.model, d = META.defaults[model];
    $("#model").value = model;
    const inp = ([k, label, step]) => el("label", { class: "field" }, label, el("input", {
      type: "number", step, value: cfg.params[k] ?? d[k] ?? "",
      oninput: e => { if (e.target.value === "") delete cfg.params[k]; else cfg.params[k] = +e.target.value; save(); },
    }));
    $("#params").replaceChildren(...GENERAL.map(inp), ...MODEL.filter(([k]) => k === "w_scale" || k in d).map(inp));
  }

  const addStim = t => { cfg.stims.push({ hz: 150, t_on: 0, t_off: "", ...t }); save(); renderControls(); status(`Stimulus added: ${t.label || t.query}`); };
  const addReadout = t => {
    if (cfg.readouts.length >= 4) return status("4 readouts at most", true);
    cfg.readouts.push({ ...t }); save(); renderControls(); status(`Readout added: ${t.label || t.query}`);
  };
  const addSilence = body => {
    const s = cfg.silence;
    if (s.field === "body" && s.query.trim()) s.query += `, ${body}`; else { s.field = "body"; s.query = String(body); }
    save(); renderControls(); status(`Neuron ${body} silenced`);
  };

  /* ------------------------------------------------------------ simulation */
  let job = null;
  async function run() {
    if (!META) return;
    const model = cfg.params.model, T = cfg.params.t_ms ?? 500;
    const body = {
      params: { ...META.defaults[model], ...cfg.params },
      stims: cfg.stims.map(s => ({ ...s, t_on: +s.t_on || 0, t_off: s.t_off === "" ? T : +s.t_off, hz: +s.hz || 0 })),
      silence: cfg.silence, readouts: cfg.readouts.filter(r => r.query.trim()),
    };
    $("#run").disabled = true; $("#cancel").hidden = false;
    $("#results").classList.add("stale");
    $("#progbar").style.width = "0";
    status("Starting…");
    try {
      const r = await fetch("/api/run", { method: "POST", body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error);
      job = d.job;
      let st;
      while (true) {
        await new Promise(res => setTimeout(res, 200));
        st = await (await fetch(`/api/job/${d.job}`)).json();
        if (job !== d.job) return;
        $("#progbar").style.width = `${Math.round(st.progress * 100)}%`;
        if (st.state !== "running") break;
        status(`${Math.round(st.progress * 100)} % · ${fmt(st.spikes)} spikes`);
      }
      if (st.state === "error") throw new Error(st.error);
      if (st.state === "cancelled") { status("Simulation cancelled"); $("#results").classList.remove("stale"); return; }
      status("Transferring the results…");
      const res = await unpack(await fetch(`/api/job/${d.job}/result`));
      Results.render(res);
      status(`Done · ${fmt(res.header.wall_s, 1)} s of compute`);
    } catch (e) {
      status(e.message || String(e), true);
      $("#results").classList.remove("stale");
    } finally {
      $("#run").disabled = false; $("#cancel").hidden = true;
    }
  }

  async function init() {
    cfg = loadCfg();
    try {
      const { header, arr } = await unpack(await fetch("/api/meta"));
      META = { ...header, ...arr };
    } catch (e) { return status("Cannot load the connectome: " + e.message, true); }
    $("#netstats").textContent = `MaleCNS v1.0 · ${fmt(META.N)} neurons · ${fmt(META.n_edges)} connections · ${fmt(META.n_synapses)} synapses`;
    for (const [sel, list] of [["#stim-preset", META.presets.stim], ["#ro-preset", META.presets.readout]])
      list.forEach((p, k) => $(sel).append(el("option", { value: k, textContent: p.label })));
    $("#stim-preset").addEventListener("change", e => {
      const p = META.presets.stim[e.target.value]; e.target.value = ""; if (p) addStim({ ...p });
    });
    $("#ro-preset").addEventListener("change", e => {
      const p = META.presets.readout[e.target.value]; e.target.value = ""; if (p) addReadout({ ...p });
    });
    $("#add-stim").addEventListener("click", () => addStim({ query: "", field: "type", label: "" }));
    $("#add-ro").addEventListener("click", () => addReadout({ query: "", field: "type", label: "" }));
    $("#model").addEventListener("change", e => {
      const keep = { model: e.target.value };
      for (const [k] of GENERAL) if (cfg.params[k] != null) keep[k] = cfg.params[k];
      if (cfg.params.w_scale != null) keep.w_scale = cfg.params.w_scale;
      cfg.params = keep; save(); renderParams();
    });
    $("#run").addEventListener("click", run);
    $("#cancel").addEventListener("click", () => { if (job) fetch(`/api/job/${job}/cancel`, { method: "POST" }); });
    document.addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$("#run").disabled) run(); });
    renderControls();
    Results.init(META, { addStim, addReadout, addSilence });
    $("#run").disabled = false;
    status("Ready · ⌘/Ctrl + Enter to simulate");
    run();
  }
  init();
})();
