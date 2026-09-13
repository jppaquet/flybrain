"""
engine - loading of the MaleCNS connectome and configurable LIF simulation.

Two models:
  - "shiu"   : Shiu et al., Nature 2024. dv/dt = (g - v)/tau_m, dg/dt = -g/tau_syn,
               a presynaptic spike adds w to g after a delay (1.8 ms).
               Stimulation = Poisson input of weight f_poi * w_syn on v.
  - "simple" : the flysim.py model (instantaneous current, no delay,
               tau_m = 5 ms, forced spikes for stimulation).
The potential v is in mV above rest (threshold 7 mV = -45 mV for a rest at -52 mV).
"""
import os, re, time
import numpy as np
import pyarrow.ipc as ipc        # Feather v2 = Arrow's IPC file format

HERE = os.path.dirname(os.path.abspath(__file__))
ANN_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CACHE = "meta_cache.npz"

# The 22 superclasses grouped into 7 anatomical groups (+ "other").
GROUPS = ["Sensory", "Optic lobe", "Central brain", "Descending",
          "Ascending", "VNC", "Motor / efferent", "Other"]

def group_of(superclass):
    s = superclass or ""
    if "sensory" in s: return 0
    if s.startswith(("ol_", "visual_")): return 1
    if s == "cb_intrinsic": return 2
    if s.startswith("descending"): return 3
    if s.startswith("ascending"): return 4
    if s == "vnc_intrinsic": return 5
    if any(k in s for k in ("motor", "efferent", "endocrine")): return 6
    return 7

DEFAULTS = {
    "shiu":   dict(dt=0.1, tau_m=20.0, tau_syn=5.0, v_th=7.0, refrac=2.2,
                   delay=1.8, w_syn=0.275, f_poi=250.0),
    "simple": dict(dt=0.2, tau_m=5.0, tau_syn=0.0, v_th=7.0, refrac=2.2,
                   delay=0.0, w_syn=0.275, f_poi=0.0),
}


class Connectome:
    def __init__(self, d=HERE):
        t0 = time.time()
        g = np.load(os.path.join(d, "malecns_graph.npz"))
        self.indptr = g["indptr"]
        self.indices = g["indices"]
        self.nsyn = g["weight"].astype(np.float32)          # number of synapses
        self.sign = g["sign"].astype(np.float32)
        self.body = g["body"]
        self.N = len(self.indptr) - 1
        self.pre = np.repeat(np.arange(self.N, dtype=np.int32), np.diff(self.indptr))
        self.signed = self.nsyn * self.sign[self.pre]      # synapses x sign
        self._csc = None
        self._load_meta(d)
        self.body_to_idx = {int(b): i for i, b in enumerate(self.body)}
        print(f"connectome loaded in {time.time() - t0:.1f} s "
              f"({self.N:,} neurons, {len(self.indices):,} connections)")

    # ------------------------------------------------------------ metadata
    def _load_meta(self, d):
        cols = ["bodyId", "type", "instance", "superclass", "class", "subclass",
                "somaSide", "rootSide", "somaLocation"]
        t = ipc.open_file(os.path.join(d, ANN_FILE)).read_all().select(cols).to_pydict()
        pos_in = np.searchsorted(t["bodyId"], self.body)
        def get(c):
            v = t[c]
            return ["" if v[p] is None else str(v[p]) for p in pos_in]
        self.type, self.inst = get("type"), get("instance")
        self.superclass, self.klass = get("superclass"), get("class")
        self.subclass = get("subclass")
        ss, rs = get("somaSide"), get("rootSide")
        self.side = [a or (b if b != "unknown" else "") for a, b in zip(ss, rs)]
        self.group = np.array([group_of(s) for s in self.superclass], dtype=np.uint8)

        import csv
        self.nt = [""] * self.N
        with open(os.path.join(d, "neurons.csv"), newline="") as f:
            for r in csv.DictReader(f):
                self.nt[int(r["idx"])] = r["nt"]

        cache = os.path.join(d, CACHE)
        if os.path.exists(cache):
            c = np.load(cache)
            if len(c["pos"]) == self.N:
                self.pos, self.pos_est = c["pos"], c["pos_est"]
                return
        loc = [t["somaLocation"][p] for p in pos_in]
        known = np.array([l is not None for l in loc])
        pos = np.zeros((self.N, 3), dtype=np.float32)
        pos[known] = np.array([l for l in loc if l is not None], dtype=np.float32)
        self.pos, self.pos_est = self._estimate_positions(pos, known), ~known
        np.savez(cache, pos=self.pos, pos_est=self.pos_est)

    def _estimate_positions(self, pos, known):
        """Neurons without a soma (mostly sensory): weighted centroid of their positioned
        partners, iterated to spread to neighbours of neighbours."""
        pre, post, w = self.pre, self.indices, self.nsyn
        for _ in range(4):
            kpre, kpost = known[pre], known[post]
            acc = np.zeros((self.N, 3)); tot = np.zeros(self.N)
            for a, b, m in ((pre, post, kpost), (post, pre, kpre)):
                tot += np.bincount(a[m], weights=w[m], minlength=self.N)
                for k in range(3):
                    acc[:, k] += np.bincount(a[m], weights=w[m] * pos[b[m], k],
                                             minlength=self.N)
            new = (~known) & (tot > 0)
            if not new.any(): break
            pos[new] = (acc[new] / tot[new, None]).astype(np.float32)
            known = known | new
        pos[~known] = pos[known].mean(0)
        return pos

    # --------------------------------------------------------------- queries
    def find(self, query, field="any"):
        """Indices of the matching neurons. field: any|type|instance|
        superclass|class|body|idx (body/idx: comma-separated list)."""
        q = (query or "").strip()
        if not q: return np.zeros(0, np.int32)
        if field in ("body", "idx"):
            ids = [int(x) for x in re.split(r"[,\s]+", q) if x.strip().isdigit()]
            if field == "body":
                ids = [self.body_to_idx[b] for b in ids if b in self.body_to_idx]
            return np.array(sorted(i for i in ids if 0 <= i < self.N), dtype=np.int32)
        rx = re.compile(q, re.I)
        cols = {"type": (self.type,), "instance": (self.inst,),
                "superclass": (self.superclass,), "class": (self.klass, self.subclass),
                }.get(field, (self.type, self.inst, self.superclass, self.klass))
        out = [i for i in range(self.N) if any(rx.search(c[i]) for c in cols)]
        return np.array(out, dtype=np.int32)

    def csc(self):
        """Input index (for presynaptic partners)."""
        if self._csc is None:
            order = np.argsort(self.indices, kind="stable")
            ptr = np.zeros(self.N + 1, dtype=np.int64)
            np.cumsum(np.bincount(self.indices, minlength=self.N), out=ptr[1:])
            self._csc = (ptr, order)
        return self._csc


class Cancelled(Exception):
    pass


def resolve_params(p):
    model = p.get("model", "shiu")
    if model not in DEFAULTS: model = "shiu"
    q = dict(DEFAULTS[model])
    for k in q:
        if p.get(k) not in (None, ""): q[k] = float(p[k])
    q["model"] = model
    q["t_ms"] = min(max(float(p.get("t_ms", 500)), 10.0), 5000.0)
    q["trials"] = int(min(max(int(p.get("trials", 1)), 1), 10))
    q["seed"] = int(p.get("seed", 0))
    q["w_scale"] = float(p.get("w_scale", 1.0))
    # short-term synaptic depression (0 = off, as in Shiu et al.)
    q["std_u"] = min(max(float(p.get("std_u") or 0), 0.0), 0.9)
    q["std_tau"] = min(max(float(p.get("std_tau") or 300), 10.0), 5000.0)
    # continuous simulation: reset to rest if activity persists without a stimulus (0 = never)
    q["quench_ms"] = max(float(p.get("quench_ms") or 0), 0.0)
    q["dt"] = min(max(q["dt"], 0.05), 1.0)
    return q


class Stepper:
    """Persistent LIF state, advanced in slices (used by simulate and by live)."""
    def __init__(self, C, q, silence=None, seed=0):
        self.C, self.q, N, dt = C, q, C.N, q["dt"]
        self.shiu = q["model"] == "shiu"
        self.w = (C.signed * np.float32(q["w_syn"] * q["w_scale"])).astype(np.float32)
        self.D = int(round(q["delay"] / dt))
        self.refr = int(round(q["refrac"] / dt))
        self.dm = np.float32(np.exp(-dt / q["tau_m"]))
        self.dg = np.float32(np.exp(-dt / q["tau_syn"])) if q["tau_syn"] > 0 else np.float32(0)
        self.kv = np.float32(dt / q["tau_m"])
        self.kick = np.float32(q["f_poi"] * q["w_syn"])
        self.th = np.float32(q["v_th"])
        self.alive = np.ones(N, dtype=bool)
        if silence is not None and len(silence): self.alive[silence] = False
        # presynaptic resources: each spike uses a fraction U of them,
        # recovered with the time constant std_tau
        self.U = q.get("std_u", 0.0)
        self.krec = np.float32(dt / q.get("std_tau", 300.0))
        self.res = np.ones(N, np.float32) if self.U > 0 else None
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros(N, np.float32); self.g = np.zeros(N, np.float32)
        self.until = np.zeros(N, np.int32)
        self.buf = np.zeros((self.D, N), np.float32) if self.D > 0 else None
        self.pending = None
        self.s = 0

    @property
    def t_ms(self): return self.s * self.q["dt"]

    def quench(self):
        """Resets the network to rest without resetting the clock."""
        self.v[:] = 0; self.g[:] = 0; self.until[:] = 0
        if self.buf is not None: self.buf[:] = 0
        self.pending = None
        if self.res is not None: self.res[:] = 1

    def step(self, n, drive=(), record=None):
        """Advances n steps. drive: [(idx, hz)] Poisson inputs. Returns the indices of
        the neurons that fired; record receives (step, indices) pairs."""
        C, N, D, shiu = self.C, self.C.N, self.D, self.shiu
        v, g, until, buf, alive, rng = self.v, self.g, self.until, self.buf, self.alive, self.rng
        drive = [(idx[alive[idx]], hz * self.q["dt"] / 1000.0) for idx, hz in drive if len(idx)]
        out = []
        for _ in range(n):
            s = self.s
            inp = buf[s % D] if D > 0 else self.pending
            hits = [idx[rng.random(len(idx)) < p] for idx, p in drive]
            if self.res is not None: self.res += (1.0 - self.res) * self.krec
            if shiu:
                g *= self.dg
                if inp is not None: g += inp
                v += (g - v) * self.kv
                for h in hits: v[h] += self.kick          # PoissonInput on v (Shiu)
            else:
                v *= self.dm
                if inp is not None: v += inp
            v[until > s] = 0.0
            fmask = (v >= self.th) & alive
            if not shiu:
                for h in hits: fmask[h] = True
            fired = np.flatnonzero(fmask).astype(np.int32)
            if D > 0: buf[s % D] = 0.0
            self.pending = None
            if len(fired):
                v[fired] = 0.0
                until[fired] = s + 1 + self.refr
                out.append(fired)
                if record is not None: record.append((s, fired))
                st, en = C.indptr[fired], C.indptr[fired + 1]
                cnt = en - st
                tot = int(cnt.sum())
                if tot:
                    off = np.repeat(st, cnt) + (np.arange(tot, dtype=np.int64)
                                                - np.repeat(np.cumsum(cnt) - cnt, cnt))
                    wo = self.w[off]
                    if self.res is not None:
                        wo = wo * np.repeat(self.res[fired], cnt)
                        self.res[fired] *= np.float32(1.0 - self.U)
                    x = np.bincount(C.indices[off], weights=wo, minlength=N).astype(np.float32)
                    if D > 0: buf[s % D] += x
                    else: self.pending = x
            self.s += 1
        return np.concatenate(out) if out else np.zeros(0, np.int32)


def simulate(C, stims, q, silence=None, progress=None, cancel=None):
    """stims: list of dict(idx, hz, t_on, t_off). Returns (steps, idx) of every spike,
    one array pair per trial."""
    dt = q["dt"]
    steps = int(round(q["t_ms"] / dt))
    sched = [(s["idx"], s["hz"], int(s["t_on"] / dt), int(s["t_off"] / dt)) for s in stims]
    edges = sorted({e for _, _, a, b in sched for e in (a, b)})
    runs, total = [], steps * q["trials"]
    for tr in range(q["trials"]):
        st = Stepper(C, q, silence, seed=q["seed"] + tr)
        rec = []
        while st.s < steps:
            if cancel is not None and cancel.is_set(): raise Cancelled()
            s = st.s
            n = min(100, steps - s, *[e - s for e in edges if e > s])
            st.step(n, [(idx, hz) for idx, hz, a, b in sched if a <= s < b], record=rec)
            if progress is not None:
                progress((tr * steps + st.s) / total, sum(len(f) for _, f in rec))
        runs.append((np.concatenate([np.full(len(f), k, np.int32) for k, f in rec]) if rec else np.zeros(0, np.int32),
                     np.concatenate([f for _, f in rec]) if rec else np.zeros(0, np.int32)))
    return runs


def nice_bin(x):
    for m in (0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200):
        if m >= x: return float(m)
    return 200.0


def summarize(C, runs, q, stim_idx, readouts, raster_rows=1200, raster_cap=600_000):
    """Aggregates the spikes: rate per neuron, population rates per group and per readout,
    frames for the brain map, raster of one trial."""
    N, dt, T, K = C.N, q["dt"], q["t_ms"], q["trials"]
    steps = int(round(T / dt))
    bin_ms = nice_bin(T / 100)
    nb = int(np.ceil(T / bin_ms))
    spb = bin_ms / dt                                   # steps per bin
    allt = np.concatenate([r[0] for r in runs]); alli = np.concatenate([r[1] for r in runs])
    nspk = np.bincount(alli, minlength=N)
    bins = np.minimum((allt / spb).astype(np.int32), nb - 1)
    rate = nspk / K / (T / 1000.0)

    def pop(mask_idx):
        m = np.zeros(N, bool); m[mask_idx] = True
        sel = m[alli]
        c = np.bincount(bins[sel], minlength=nb)[:nb].astype(np.float64)
        return (c / max(len(mask_idx), 1) / K / (bin_ms / 1000.0)).astype(np.float32)

    groups = np.stack([pop(np.flatnonzero(C.group == k)) for k in range(len(GROUPS))])
    rd = np.stack([pop(r["idx"]) for r in readouts]) if readouts else np.zeros((0, nb), np.float32)

    active = np.flatnonzero(nspk > 0).astype(np.int32)
    rank = np.full(N, -1, np.int64); rank[active] = np.arange(len(active))
    frames = np.bincount(rank[alli] * nb + bins, minlength=len(active) * nb)
    frames = np.minimum(frames, 255).astype(np.uint8)

    # raster (trial 0): stimulated, readouts, then the most active
    t0, i0 = runs[0]
    stimset = np.zeros(N, bool); stimset[stim_idx] = True
    first = np.full(N, np.inf); np.minimum.at(first, i0, t0.astype(np.float64))
    rows, role = [], []
    def take(idx, r, cap):
        idx = np.asarray(idx, np.int32)
        if len(idx) > cap: idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
        for i in idx[np.argsort(first[idx], kind="stable")]:
            if i not in seen: seen.add(int(i)); rows.append(int(i)); role.append(r)
    seen = set()
    take(stim_idx, 0, 150)
    for k, r in enumerate(readouts): take(r["idx"], 1 + k, 100)
    others = np.flatnonzero((nspk > 0) & ~stimset)
    others = others[np.argsort(-nspk[others], kind="stable")][: max(raster_rows - len(rows), 0)]
    others = others[np.lexsort((first[others], C.group[others]))]
    for i in others:
        if int(i) not in seen: seen.add(int(i)); rows.append(int(i)); role.append(9)
    rows = np.array(rows, np.int32)
    row_of = np.full(N, -1, np.int32); row_of[rows] = np.arange(len(rows))
    m = row_of[i0] >= 0
    spk_row, spk_t = row_of[i0[m]], (t0[m] * dt).astype(np.float32)
    if len(spk_row) > raster_cap:
        keep = np.sort(np.random.default_rng(0).choice(len(spk_row), raster_cap, replace=False))
        spk_row, spk_t = spk_row[keep], spk_t[keep]

    arrays = dict(
        active=active, active_n=nspk[active].astype(np.int32), frames=frames,
        groups=groups.ravel(), readouts=rd.ravel().astype(np.float32),
        raster_rows=rows, raster_role=np.array(role, np.uint8),
        spk_row=spk_row.astype(np.uint16 if len(rows) < 65535 else np.int32), spk_t=spk_t,
    )
    header = dict(
        bin_ms=bin_ms, nbins=nb, steps=steps, total_spikes=int(nspk.sum()),
        n_active=int(len(active)), n_active_nonstim=int(((nspk > 0) & ~stimset).sum()),
        mean_rate_active=float(rate[active].mean()) if len(active) else 0.0,
        raster_truncated=bool(len(t0) and m.sum() > raster_cap),
    )
    return header, arrays
