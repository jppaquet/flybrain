/* results.js - rendering of a simulation result, tables, inspector. */
const Results = (() => {
  const $ = s => document.querySelector(s);
  const { fmt, groupColor, css } = Charts;
  let M, hooks, R = null, rate = null, rateMax = 1, stimSet = null, frameVal = null, frameMax = 1;
  let brain, roChart, grpChart, raster, windows = [];
  let frame = null, timer = null, isolate = null, sel = null;
  const topSort = { key: "rate", asc: false }, typeSort = { key: "mean", asc: false };
  let typeCount;

  const el = (tag, props = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "style") Object.assign(e.style, v); else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "class") e.className = v; else e[k] = v;
    }
    for (const c of kids) if (c != null) e.append(c);
    return e;
  };
  const typeName = i => M.types[M.type_id[i]] || `#${i}`;

  function describe(i) {
    const rows = [rate ? { value: `${fmt(rate[i])} Hz`, label: typeName(i) } : { value: typeName(i) }];
    const side = M.sides[M.side[i]], nt = M.nt_names[M.nt[i]] || "NT ?";
    rows.push({ cls: "tl", label: `${M.groups[M.group[i]]} · ${nt}${side ? " · " + side : ""}` });
    if (stimSet && stimSet[i]) rows.push({ cls: "tl", label: "stimulated" });
    if (M.pos_est[i]) rows.push({ cls: "tl", label: "estimated position (no soma)" });
    return rows;
  }

  function init(meta, h) {
    M = meta; hooks = h;
    typeCount = new Int32Array(M.types.length);
    for (let i = 0; i < M.N; i++) typeCount[M.type_id[i]]++;
    brain = new Charts.BrainMap($("#brain"), { describe, onPick: inspect });
    brain.setMeta(M);
    roChart = new Charts.Multiples($("#ro-chart"), { cols: (w, n) => (w > 420 ? Math.min(n, 2) : 1) });
    grpChart = new Charts.Multiples($("#groups-chart"), { cols: (w, n) => (w > 560 ? 4 : 2) });
    raster = new Charts.Raster($("#raster"), { describe, onPick: inspect });

    $("#view-seg").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      for (const x of $("#view-seg").children) x.classList.toggle("on", x === b);
      brain.setView(b.dataset.v);
    });
    $("#scrub").addEventListener("input", e => { stop(); setFrame(+e.target.value); });
    $("#cumul").addEventListener("click", () => { stop(); setFrame(null); });
    $("#play").addEventListener("click", () => (timer ? stop() : play()));
    $("#top-filter").addEventListener("input", renderTop);
    $("#type-filter").addEventListener("input", renderTypes);
    $("#insp-close").addEventListener("click", () => { $("#inspector").hidden = true; sel = null; brain.select(null); });
    $("#td-close").addEventListener("click", () => $("#table-dialog").close());
    document.querySelectorAll("[data-table]").forEach(b => b.addEventListener("click", () => tableView(b.dataset.table)));

    const ro = new ResizeObserver(entries => {
      for (const en of entries) {
        const c = en.target.querySelector("canvas");
        if (c === $("#brain")) brain.project(); else ({ "ro-chart": roChart, "groups-chart": grpChart, raster })[c.id]?.draw();
      }
    });
    document.querySelectorAll(".canvas-wrap").forEach(w => ro.observe(w));
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    buildBrainLegend();
  }

  function redraw() { brain.draw(); roChart.draw(); grpChart.draw(); raster.draw(); buildBrainLegend(); }

  /* ------------------------------------------------------------ map */
  function buildBrainLegend() {
    const L = $("#brain-legend"); L.replaceChildren();
    const vmax = frame == null ? rateMax : frameMax;
    L.append(el("span", {}, "0"), el("span", { class: "ramp", style: { background: Charts.rampCSS() } }),
             el("span", {}, `${fmt(vmax)} Hz · log scale`), el("span", { style: { flex: "1" } }));
    L.append(el("span", { class: "hint" }, "Isolate:"));
    const chip = (label, g) => el("button", {
      class: "ghost small" + (isolate === g ? " on" : ""), textContent: label,
      onclick: () => { isolate = g; brain.setIsolate(g); buildBrainLegend(); },
    });
    L.append(chip("All", null));
    M.groups.forEach((g, k) => { if (M.group_sizes[k]) L.append(chip(g, k)); });
  }

  function frameValue(f) {
    const h = R.header, a = R.arr, nb = h.nbins, s = 1 / (h.params.trials * h.bin_ms / 1000);
    const v = frameVal || (frameVal = new Float32Array(M.N));
    v.fill(0);
    for (let k = 0; k < a.active.length; k++) {
      let c = 0, n = 0;
      for (let j = Math.max(0, f - 1); j <= Math.min(nb - 1, f + 1); j++) { c += a.frames[k * nb + j]; n++; }
      v[a.active[k]] = (c / n) * s;
    }
    return v;
  }

  function setFrame(f) {
    frame = f;
    const bin = R.header.bin_ms;
    $("#cumul").classList.toggle("on", f == null);
    if (f == null) {
      brain.setValue(rate, rateMax);
      $("#tlabel").textContent = "cumulative";
      [roChart, grpChart, raster].forEach(c => c.cursor(null));
    } else {
      $("#scrub").value = f;
      brain.setValue(frameValue(f), frameMax);
      $("#tlabel").textContent = `${fmt(f * bin)}–${fmt((f + 1) * bin)} ms`;
      [roChart, grpChart, raster].forEach(c => c.cursor((f + 0.5) * bin));
    }
    buildBrainLegend();
  }
  function play() {
    if (!R) return;
    let f = frame == null || frame >= R.header.nbins - 1 ? 0 : frame;
    $("#play").textContent = "❚❚";
    setFrame(f);
    timer = setInterval(() => {
      if (++f >= R.header.nbins) return stop();
      setFrame(f);
    }, 90);
  }
  function stop() { clearInterval(timer); timer = null; $("#play").textContent = "▶"; }

  /* ------------------------------------------------------------ render */
  function render(res) {
    stop(); R = res; frameVal = null;
    const h = res.header, a = res.arr, N = M.N, T = h.params.t_ms, K = h.params.trials, nb = h.nbins;
    rate = new Float32Array(N); rateMax = 1;
    for (let k = 0; k < a.active.length; k++) {
      const r = a.active_n[k] / K / (T / 1000); rate[a.active[k]] = r; if (r > rateMax) rateMax = r;
    }
    stimSet = new Uint8Array(N); a.stim_idx.forEach(i => (stimSet[i] = 1));
    const s = 1 / (K * h.bin_ms / 1000); frameMax = 1;
    for (let k = 0; k < a.active.length; k++) for (let j = 0; j < nb; j++) {
      let c = 0, n = 0;
      for (let q = Math.max(0, j - 1); q <= Math.min(nb - 1, j + 1); q++) { c += a.frames[k * nb + q]; n++; }
      if (c / n * s > frameMax) frameMax = c / n * s;
    }
    windows = h.stims.map(x => [x.t_on, Math.min(x.t_off, T)]);
    $("#scrub").max = nb - 1;
    kpis(h, N, T, K);

    const ro = h.readouts.map((r, k) => {
      const values = Array.from(a.readouts.subarray(k * nb, (k + 1) * nb));
      return { label: r.label, sub: `${fmt(r.n)} neurons · peak ${fmt(Math.max(...values))} Hz`, color: css("--ink-2"), values };
    });
    roChart.set({ panels: ro, bin_ms: h.bin_ms, T, windows });
    $("#ro-legend").textContent = ro.length ? "" : "No readout: add one in the left panel.";

    const active = new Int32Array(M.groups.length);
    for (const i of a.active) if (!stimSet[i]) active[M.group[i]]++;
    const gp = [];
    M.groups.forEach((g, k) => {
      if (!M.group_sizes[k]) return;
      gp.push({ label: g, sub: `${fmt(active[k])} / ${fmt(M.group_sizes[k])} active`, color: groupColor(k),
                values: Array.from(a.groups.subarray(k * nb, (k + 1) * nb)) });
    });
    grpChart.set({ panels: gp, bin_ms: h.bin_ms, T, windows });

    const rows = a.raster_rows, role = a.raster_role, n = rows.length;
    const rowGroup = new Uint8Array(n), blockKey = new Uint8Array(n), blockLabel = new Array(n);
    for (let r = 0; r < n; r++) {
      rowGroup[r] = M.group[rows[r]];
      blockKey[r] = role[r] < 9 ? role[r] : 10 + M.group[rows[r]];
      blockLabel[r] = role[r] === 0 ? "Stimulated" : role[r] < 9 ? h.readouts[role[r] - 1].label : M.groups[M.group[rows[r]]];
    }
    raster.set({ rows, rowGroup, blockKey, blockLabel, spk_row: a.spk_row, spk_t: a.spk_t, T, windows });
    $("#raster-sub").textContent = `Trial 1 · ${fmt(n)} neurons: stimulated, readouts, then the most active per group` +
      (h.raster_truncated ? " · spikes subsampled" : "");
    const RL = $("#raster-legend"); RL.replaceChildren();
    M.groups.forEach((g, k) => { if (M.group_sizes[k]) RL.append(el("span", {}, el("span", { class: "sw", style: { background: groupColor(k) } }), g)); });

    setFrame(null);
    renderTop(); renderTypes();
    $("#results").classList.remove("empty", "stale");
    if (sel != null) inspect(sel);
  }

  function kpis(h, N, T, K) {
    const tiles = [
      ["Recruited neurons", fmt(h.n_active_nonstim), `${(100 * h.n_active_nonstim / N).toFixed(1)} % of the CNS · excluding ${fmt(h.n_stim)} stimulated`],
      ["Spikes", fmt(h.total_spikes), K > 1 ? `${fmt(h.total_spikes / K)} per trial` : `in ${fmt(T)} ms simulated`],
      ["Mean rate of active neurons", `${fmt(h.mean_rate_active)} Hz`, h.n_silenced ? `${fmt(h.n_silenced)} neurons silenced` : "over the whole run"],
      ["Compute", `${fmt(h.wall_s, 1)} s`, `${fmt(h.steps)} steps × ${K} trial${K > 1 ? "s" : ""} · dt ${h.params.dt} ms`],
    ];
    $("#kpis").replaceChildren(...tiles.map(([l, v, d]) =>
      el("div", { class: "kpi" }, el("div", { class: "label" }, l), el("div", { class: "value" }, v), el("div", { class: "detail" }, d))));
  }

  /* ------------------------------------------------------------ tables */
  function table(tbl, cols, rows, sort, rerender, onClick, limit = 300) {
    const k = sort.key, dir = sort.asc ? 1 : -1;
    rows.sort((p, q) => (p[k] < q[k] ? -dir : p[k] > q[k] ? dir : 0));
    const thead = el("thead", {}, el("tr", {}, ...cols.map(c => el("th", {
      class: (c.num ? "num " : "") + (c.key === k ? "sorted" + (sort.asc ? " asc" : "") : ""), textContent: c.label,
      onclick: () => { if (sort.key === c.key) sort.asc = !sort.asc; else { sort.key = c.key; sort.asc = !c.num; } rerender(); },
    }))));
    const tbody = el("tbody");
    for (const r of rows.slice(0, limit)) {
      const tr = el("tr", { onclick: () => onClick(r) });
      for (const c of cols) tr.append(el("td", { class: c.num ? "num" : "" }, ...[].concat(c.cell ? c.cell(r) : String(r[c.key]))));
      tbody.append(tr);
    }
    const foot = rows.length > limit ? el("tfoot", {}, el("tr", {}, el("td", { colSpan: cols.length, class: "muted" }, `… ${fmt(rows.length - limit)} more rows (refine the filter)`))) : null;
    tbl.replaceChildren(thead, tbody, ...(foot ? [foot] : []));
  }
  const regex = s => { try { return s ? new RegExp(s, "i") : null; } catch { return /$^/; } };
  const gdot = g => el("span", { class: "dot", style: { background: groupColor(g) } });

  function renderTop() {
    if (!R) return;
    const rx = regex($("#top-filter").value), a = R.arr, K = R.header.params.trials;
    const rows = [];
    for (let k = 0; k < a.active.length; k++) {
      const i = a.active[k], name = typeName(i);
      if (rx && !rx.test(name)) continue;
      rows.push({ i, name, group: M.group[i], nt: M.nt_names[M.nt[i]], rate: rate[i], n: a.active_n[k] / K });
    }
    table($("#top-table"), [
      { key: "name", label: "Neuron", cell: r => [r.name, M.sides[M.side[r.i]] ? ` ${M.sides[M.side[r.i]]}` : "", stimSet[r.i] ? el("span", { class: "tag stim", textContent: " stim" }) : null] },
      { key: "group", label: "Group", cell: r => [gdot(r.group), M.groups[r.group]] },
      { key: "nt", label: "NT" },
      { key: "rate", label: "Rate (Hz)", num: true, cell: r => [fmt(r.rate), el("span", { class: "bar", style: { width: `${Math.max(1, 48 * r.rate / rateMax)}px` } })] },
      { key: "n", label: "Spikes", num: true, cell: r => fmt(r.n) },
    ], rows, topSort, renderTop, r => inspect(r.i));
  }

  function renderTypes() {
    if (!R) return;
    const rx = regex($("#type-filter").value), a = R.arr;
    const agg = new Map();
    for (let k = 0; k < a.active.length; k++) {
      const i = a.active[k], t = M.type_id[i];
      let g = agg.get(t); if (!g) agg.set(t, g = { tid: t, act: 0, sum: 0, max: 0, group: M.group[i], stim: 0 });
      g.act++; g.sum += rate[i]; g.max = Math.max(g.max, rate[i]); g.stim += stimSet[i];
    }
    const rows = [];
    for (const g of agg.values()) {
      const name = M.types[g.tid] || "(untyped)";
      if (rx && !rx.test(name)) continue;
      rows.push({ ...g, name, n: typeCount[g.tid], frac: g.act / typeCount[g.tid], mean: g.sum / typeCount[g.tid] });
    }
    table($("#type-table"), [
      { key: "name", label: "Type", cell: r => [r.name, r.stim ? el("span", { class: "tag stim", textContent: " stim" }) : null] },
      { key: "group", label: "Group", cell: r => [gdot(r.group), M.groups[r.group]] },
      { key: "frac", label: "Active", num: true, cell: r => `${r.act} / ${r.n}` },
      { key: "mean", label: "Mean rate (Hz)", num: true, cell: r => fmt(r.mean) },
      { key: "max", label: "Max (Hz)", num: true, cell: r => fmt(r.max) },
    ], rows, typeSort, renderTypes, r => {
      $("#top-filter").value = `^${r.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`; renderTop();
      $("#top-filter").scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }

  function tableView(which) {
    if (!R) return;
    const h = R.header, a = R.arr, nb = h.nbins;
    const series = which === "readout"
      ? h.readouts.map((r, k) => [r.label, a.readouts.subarray(k * nb, (k + 1) * nb)])
      : M.groups.map((g, k) => [g, a.groups.subarray(k * nb, (k + 1) * nb)]).filter((_, k) => M.group_sizes[k]);
    $("#td-title").textContent = which === "readout" ? "Readouts — rate (Hz) per bin" : "Groups — rate (Hz) per bin";
    const t = $("#td-table");
    t.replaceChildren(
      el("thead", {}, el("tr", {}, el("th", { class: "num", textContent: "t (ms)" }), ...series.map(([l]) => el("th", { class: "num", textContent: l })))),
      el("tbody", {}, ...Array.from({ length: nb }, (_, j) => el("tr", {},
        el("td", { class: "num", textContent: `${fmt(j * h.bin_ms)}–${fmt((j + 1) * h.bin_ms)}` }),
        ...series.map(([, v]) => el("td", { class: "num", textContent: fmt(v[j]) }))))));
    $("#table-dialog").showModal();
  }

  /* ------------------------------------------------------------ inspector */
  async function inspect(i) {
    sel = i; brain.select(i);
    const r = await fetch(`/api/neuron/${i}`);
    if (!r.ok) return;
    const d = await r.json();
    if (sel !== i) return;
    const title = d.type || d.instance || `#${d.body}`;
    $("#insp-title").textContent = title;
    const kv = [
      ["Instance", d.instance || "–"], ["bodyId", String(d.body)], ["Superclass", d.superclass || "–"],
      ["Class", [d.klass, d.subclass].filter(Boolean).join(" · ") || "–"], ["Side", d.side || "–"],
      ["Neurotransmitter", d.nt || "–"], ["Group", d.group],
      ["Connections", `${fmt(d.n_in)} inputs (${fmt(d.syn_in)} syn.) · ${fmt(d.n_out)} outputs (${fmt(d.syn_out)} syn.)`],
      ["Position", d.pos_est ? "estimated (no annotated soma)" : "soma"],
    ];
    if (rate) kv.unshift(["Simulated rate", `${fmt(rate[i])} Hz${stimSet[i] ? " (stimulated)" : ""}`]);
    const dl = el("dl", { class: "kv" });
    for (const [k, v] of kv) dl.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
    const target = { query: String(d.body), field: "body", label: title };
    const actions = el("div", { class: "insp-actions" },
      el("button", { class: "ghost small", textContent: "Stimulate", onclick: () => hooks.addStim(target) }),
      el("button", { class: "ghost small", textContent: "Readout", onclick: () => hooks.addReadout(target) }),
      el("button", { class: "ghost small", textContent: "Silence", onclick: () => hooks.addSilence(d.body) }),
      el("button", { class: "ghost small", textContent: "Whole type", disabled: !d.type,
                      onclick: () => hooks.addStim({ query: `^${d.type.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`, field: "type", label: d.type }) }));
    const partners = (list, label, total) => {
      const t = el("table");
      t.append(el("thead", {}, el("tr", {}, el("th", { textContent: "Partner" }), el("th", { class: "num", textContent: "Syn." }),
        el("th", { textContent: "NT" }), el("th", { class: "num", textContent: "Rate" }))));
      const tb = el("tbody");
      for (const p of list) tb.append(el("tr", { onclick: () => inspect(p.idx) },
        el("td", { textContent: p.instance || p.type || `#${p.idx}` }),
        el("td", { class: "num", textContent: `${p.sign < 0 ? "−" : "+"}${p.n}` }),
        el("td", { textContent: p.nt || "?" }),
        el("td", { class: "num", textContent: rate ? fmt(rate[p.idx]) : "–" })));
      t.append(tb);
      return [el("h4", { textContent: `${label} (top ${fmt(list.length)} of ${fmt(total)})` }), t];
    };
    $("#insp-body").replaceChildren(dl, actions,
      ...partners(d.inputs, "Inputs", d.n_in), ...partners(d.outputs, "Outputs", d.n_out));
    $("#inspector").hidden = false;
  }

  return { init, render, inspect, redraw };
})();
