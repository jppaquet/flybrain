#!/usr/bin/env python3
"""
flysim - simulateur "leaky integrate-and-fire" sur le connectome MaleCNS v1.0.

Modèle (d'après Shiu et al., Nature 2024, sur le connectome femelle) :
  - potentiel de membrane v en mV au-dessus du repos, fuite de constante tau
  - décharge quand v >= seuil, puis reset à 0 et période réfractaire
  - une synapse = 0.275 mV ; le poids d'une connexion = nb de synapses x signe
  - signe : acétylcholine -> excitateur ; GABA et glutamate -> inhibiteurs

Sous-commandes :
  python flysim.py types  [motif]      liste les types de neurones (regex)
  python flysim.py info   <motif>      détail des neurones qui matchent
  python flysim.py run --stim MOTIF [--readout MOTIF] [--ms 300] [--hz 150]
"""
import argparse, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------- paramètres
TAU_MS      = 5.0     # constante de temps membranaire
V_THRESH    = 7.0     # mV au-dessus du repos
REFRAC_MS   = 2.2
SYN_MV      = 0.275   # contribution d'une synapse
DT_MS       = 0.2


class Brain:
    def __init__(self, d=HERE):
        g = np.load(os.path.join(d, "malecns_graph.npz"))
        self.indptr  = g["indptr"]
        self.indices = g["indices"]
        self.sign    = g["sign"].astype(np.float32)
        self.body    = g["body"]
        # poids signé, en mV, porté par le neurone présynaptique
        self.w = (g["weight"].astype(np.float32) * SYN_MV)
        self.N = len(self.indptr) - 1
        pre_of_edge = np.repeat(np.arange(self.N, dtype=np.int32), np.diff(self.indptr))
        self.w *= self.sign[pre_of_edge]
        del pre_of_edge

        import csv
        self.type = [""] * self.N; self.inst = [""] * self.N
        self.klass = [""] * self.N; self.side = [""] * self.N; self.nt = [""] * self.N
        with open(os.path.join(d, "neurons.csv"), newline="") as f:
            for r in csv.DictReader(f):
                i = int(r["idx"])
                self.type[i] = r["type"]; self.inst[i] = r["instance"]
                self.klass[i] = r["class"]; self.side[i] = r["side"]; self.nt[i] = r["nt"]
        self.label = [f"{t or i or '?'}" for t, i in zip(self.type, self.inst)]

    def find(self, pattern, field="any"):
        rx = re.compile(pattern, re.I)
        out = []
        for i in range(self.N):
            hay = {"type": self.type[i], "instance": self.inst[i],
                   "class": self.klass[i]}.get(field)
            if hay is None:
                hay = f"{self.type[i]} {self.inst[i]} {self.klass[i]}"
            if rx.search(hay): out.append(i)
        return np.array(out, dtype=np.int32)

    # ------------------------------------------------------------- simulation
    def run(self, stim, ms=300.0, hz=150.0, stim_ms=None, seed=0, record=None):
        """stim : indices forcés à décharger à `hz` Hz pendant `stim_ms`.
        Retourne (n_spikes par neurone, raster {t: idx}, trace du readout)."""
        rng = np.random.default_rng(seed)
        steps = int(ms / DT_MS)
        stim_steps = steps if stim_ms is None else int(stim_ms / DT_MS)
        decay = np.exp(-DT_MS / TAU_MS).astype(np.float32)
        refrac_steps = int(REFRAC_MS / DT_MS)

        v = np.zeros(self.N, dtype=np.float32)
        until = np.zeros(self.N, dtype=np.int32)      # fin de période réfractaire
        nspk = np.zeros(self.N, dtype=np.int32)
        p_stim = hz * DT_MS / 1000.0
        raster_t, raster_i = [], []
        trace = np.zeros(steps, dtype=np.int32) if record is not None else None
        rec = np.zeros(self.N, dtype=bool)
        if record is not None and len(record): rec[record] = True

        t0 = time.time()
        for s in range(steps):
            fired = np.flatnonzero((v >= V_THRESH) & (until <= s))
            if s < stim_steps and len(stim):
                forced = stim[rng.random(len(stim)) < p_stim]
                fired = np.union1d(fired, forced) if len(forced) else fired
            if len(fired):
                v[fired] = 0.0
                until[fired] = s + refrac_steps
                nspk[fired] += 1
                raster_t.append(np.full(len(fired), s * DT_MS, dtype=np.float32))
                raster_i.append(fired.astype(np.int32))
                if trace is not None: trace[s] = int(rec[fired].sum())
                # propagation : concaténation des lignes CSR des neurones actifs
                st, en = self.indptr[fired], self.indptr[fired + 1]
                cnt = en - st
                tot = int(cnt.sum())
                if tot:
                    off = np.repeat(st, cnt) + (np.arange(tot, dtype=np.int64)
                                                - np.repeat(np.cumsum(cnt) - cnt, cnt))
                    v *= decay
                    v += np.bincount(self.indices[off], weights=self.w[off],
                                     minlength=self.N).astype(np.float32)
                else:
                    v *= decay
            else:
                v *= decay
            v[until > s] = 0.0                       # neurones en période réfractaire
        dur = time.time() - t0
        rt = np.concatenate(raster_t) if raster_t else np.zeros(0, np.float32)
        ri = np.concatenate(raster_i) if raster_i else np.zeros(0, np.int32)
        print(f"  ({steps} pas de {DT_MS} ms simulés en {dur:.1f} s, "
              f"{int(nspk.sum()):,} décharges)", file=sys.stderr)
        return nspk, (rt, ri), trace


def cmd_types(b, a):
    from collections import Counter
    rx = re.compile(a.pattern, re.I) if a.pattern else None
    c = Counter(t for t in b.type if t and (rx is None or rx.search(t)))
    for t, n in c.most_common(a.top):
        print(f"{n:6d}  {t}")
    print(f"\n{len(c):,} types distincts, {sum(c.values()):,} neurones")


def cmd_info(b, a):
    idx = b.find(a.pattern)
    print(f"{len(idx)} neurones pour /{a.pattern}/")
    for i in idx[: a.top]:
        deg_out = b.indptr[i + 1] - b.indptr[i]
        print(f"  [{i:6d}] body={b.body[i]:<12} type={b.type[i]:<20} "
              f"class={b.klass[i]:<14} side={b.side[i]:<6} nt={b.nt[i]:<8} "
              f"sortantes={deg_out}")


def cmd_run(b, a):
    stim = b.find(a.stim)
    if not len(stim): sys.exit(f"aucun neurone ne matche /{a.stim}/")
    read = b.find(a.readout) if a.readout else np.zeros(0, np.int32)
    print(f"Stimulation : {len(stim)} neurones /{a.stim}/ à {a.hz} Hz pendant {a.ms} ms")
    if len(read): print(f"Lecture     : {len(read)} neurones /{a.readout}/")
    nspk, (rt, ri), trace = b.run(stim, ms=a.ms, hz=a.hz, record=read)

    order = np.argsort(-nspk)
    print("\nNeurones les plus actifs (hors stimulation) :")
    shown = 0
    stimset = set(stim.tolist())
    for i in order:
        if i in stimset or nspk[i] == 0: continue
        print(f"  {nspk[i]:5d} décharges  {b.type[i] or b.inst[i]:<24} "
              f"{b.klass[i]:<14} {b.side[i]:<5} {b.nt[i]}")
        shown += 1
        if shown >= a.top: break
    if len(read):
        tot = int(nspk[read].sum())
        act = int((nspk[read] > 0).sum())
        print(f"\nReadout /{a.readout}/ : {tot:,} décharges, "
              f"{act}/{len(read)} neurones actifs")
    np.savez(os.path.join(HERE, "last_run.npz"), nspk=nspk, rt=rt, ri=ri,
             stim=stim, read=read, trace=trace if trace is not None else np.zeros(0))
    print("\n-> last_run.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("types"); q.add_argument("pattern", nargs="?")
    q.add_argument("--top", type=int, default=40); q.set_defaults(fn=cmd_types)
    q = sub.add_parser("info"); q.add_argument("pattern")
    q.add_argument("--top", type=int, default=30); q.set_defaults(fn=cmd_info)
    q = sub.add_parser("run"); q.add_argument("--stim", required=True)
    q.add_argument("--readout"); q.add_argument("--ms", type=float, default=300)
    q.add_argument("--hz", type=float, default=150); q.add_argument("--top", type=int, default=25)
    q.set_defaults(fn=cmd_run)
    a = p.parse_args()
    a.fn(Brain(), a)
