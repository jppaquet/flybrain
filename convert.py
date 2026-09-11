#!/usr/bin/env python3
"""
MaleCNS v1.0 -> graphe compact pour simulation spiking.

Lit les .feather du connectome plat (Janelia/Google Research, CC-BY) et produit :
  - malecns_graph.npz : CSR (indptr, indices, weight) du graphe neurone->neurone
  - neurons.csv       : idx, body, type, class, side, nt, sign  (une ligne par neurone)

Usage: python convert.py [dossier]
"""
import sys, os, csv
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc        # Feather v2 = format de fichier IPC d'Arrow

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
W_FILE  = os.path.join(D, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
A_FILE  = os.path.join(D, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
NT_FILE = os.path.join(D, "body-neurotransmitters-male-cns-v1.0.feather")

MIN_WEIGHT = int(os.environ.get("MIN_WEIGHT", "3"))   # nb minimal de synapses pour garder une connexion

def pick(cols, *cands, contains=None):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    if contains:
        for c in cols:
            if all(t in c.lower() for t in contains): return c
    return None

def show(name, tbl):
    print(f"\n[{name}] {tbl.num_rows:,} lignes")
    for f in tbl.schema:
        print(f"    {f.name:<28} {f.type}")

# ---------------------------------------------------------------- annotations
print("Lecture des annotations…")
ann = ipc.open_file(A_FILE).read_all()
show("body-annotations", ann)
acols = ann.schema.names
c_body  = pick(acols, "body", "bodyid", "body_id", contains=["body"])
c_type  = pick(acols, "type", "celltype", "cell_type")
c_class = pick(acols, "class", "superclass", "super_class")
c_side  = pick(acols, "side", "soma_side")
c_inst  = pick(acols, "instance")
print(f"  -> body={c_body} type={c_type} class={c_class} side={c_side}")

def col(tbl, name, default=""):
    if name is None: return None
    a = tbl.column(name)
    if pa.types.is_dictionary(a.type): a = a.cast(pa.string())
    return a.to_pylist()

bodies = np.asarray(ann.column(c_body).to_numpy(zero_copy_only=False), dtype=np.int64)
types  = col(ann, c_type)  or [""] * len(bodies)
klass  = col(ann, c_class) or [""] * len(bodies)
sides  = col(ann, c_side)  or [""] * len(bodies)
insts  = col(ann, c_inst)  or [""] * len(bodies)
types  = ["" if t is None else str(t) for t in types]
klass  = ["" if t is None else str(t) for t in klass]
sides  = ["" if t is None else str(t) for t in sides]
insts  = ["" if t is None else str(t) for t in insts]

# on garde les corps identifiés comme neurones (type ou classe renseignés)
keep = np.array([bool(t) or bool(k) for t, k in zip(types, klass)])
print(f"  {keep.sum():,} corps retenus comme neurones sur {len(bodies):,}")
bodies, types, klass, sides, insts = (
    bodies[keep],
    [t for t, k in zip(types, keep) if k],
    [t for t, k in zip(klass, keep) if k],
    [t for t, k in zip(sides, keep) if k],
    [t for t, k in zip(insts, keep) if k],
)
order  = np.argsort(bodies, kind="stable")
bodies = bodies[order]
types  = [types[i] for i in order]; klass = [klass[i] for i in order]
sides  = [sides[i] for i in order]; insts = [insts[i] for i in order]
N = len(bodies)

# ------------------------------------------------------------ neurotransmitter
nt = [""] * N
if os.path.exists(NT_FILE):
    print("Lecture des neurotransmetteurs…")
    ntt = ipc.open_file(NT_FILE).read_all()
    show("body-neurotransmitters", ntt)
    ncols = ntt.schema.names
    n_body = pick(ncols, "body", "bodyid", "body_id", contains=["body"])
    n_nt   = pick(ncols, "consensus_nt", "top_nt", "predicted_nt", "nt", contains=["nt"])
    if n_body and n_nt:
        nb = np.asarray(ntt.column(n_body).to_numpy(zero_copy_only=False), dtype=np.int64)
        nv = col(ntt, n_nt)
        pos = np.searchsorted(bodies, nb)
        ok = (pos < N) & (bodies[np.clip(pos, 0, N - 1)] == nb)
        for p, v in zip(pos[ok], (v for v, o in zip(nv, ok) if o)):
            nt[p] = "" if v is None else str(v).lower()

INHIB = ("gaba", "glut")            # GABA et glutamate = inhibiteurs chez la drosophile
sign = np.ones(N, dtype=np.int8)
for i, v in enumerate(nt):
    if any(v.startswith(x) for x in INHIB): sign[i] = -1
print(f"  {(sign < 0).sum():,} neurones inhibiteurs / {N:,}")

# ------------------------------------------------------------------- connexions
# lecture lot par lot, filtrée au fil de l'eau : ~1 Go de RAM au lieu de ~10 Go
print("Lecture du graphe de connexions (gros fichier, lu par lots)…")
with pa.memory_map(W_FILE) as src:
    rd = ipc.open_file(src)
    wcols = rd.schema.names
    print(f"\n[connectome-weights] {rd.num_record_batches:,} lots")
    for f in rd.schema:
        print(f"    {f.name:<28} {f.type}")
    c_pre  = pick(wcols, "body_pre", "bodyid_pre", "pre", "pre_id", contains=["pre"])
    c_post = pick(wcols, "body_post", "bodyid_post", "post", "post_id", contains=["post"])
    c_w    = pick(wcols, "weight", "count", "syn_count", contains=["weight"])
    print(f"  -> pre={c_pre} post={c_post} weight={c_w}")
    i_pre, i_post, i_w = (rd.schema.get_field_index(c) for c in (c_pre, c_post, c_w))
    parts, n_raw = [], 0
    for k in range(rd.num_record_batches):
        b = rd.get_batch(k)
        wb = b.column(i_w).to_numpy(zero_copy_only=False)
        n_raw += len(wb)
        m = wb >= MIN_WEIGHT
        if m.any():
            parts.append((np.asarray(b.column(i_pre).to_numpy(zero_copy_only=False)[m], dtype=np.int64),
                          np.asarray(b.column(i_post).to_numpy(zero_copy_only=False)[m], dtype=np.int64),
                          np.asarray(wb[m], dtype=np.int32)))
pre, post, wgt = (np.concatenate([p[j] for p in parts]) for j in range(3))
del parts
print(f"  {n_raw:,} paires brutes")
print(f"  {len(pre):,} paires avec weight >= {MIN_WEIGHT}")

ip = np.searchsorted(bodies, pre);  ip_ok = (ip < N) & (bodies[np.clip(ip, 0, N-1)] == pre)
iq = np.searchsorted(bodies, post); iq_ok = (iq < N) & (bodies[np.clip(iq, 0, N-1)] == post)
m = ip_ok & iq_ok
ip, iq, wgt = ip[m].astype(np.int32), iq[m].astype(np.int32), wgt[m].astype(np.int32)
del pre, post
print(f"  {len(ip):,} connexions entre neurones identifiés")

o = np.argsort(ip, kind="stable")
ip, iq, wgt = ip[o], iq[o], wgt[o]
indptr = np.zeros(N + 1, dtype=np.int64)
np.add.at(indptr, ip.astype(np.int64) + 1, 1)
np.cumsum(indptr, out=indptr)

out = os.path.join(D, "malecns_graph.npz")
np.savez(out, indptr=indptr, indices=iq, weight=wgt.astype(np.int32), sign=sign, body=bodies)
print(f"\nOK -> {out}  ({os.path.getsize(out)/1e6:.0f} Mo)")

with open(os.path.join(D, "neurons.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["idx", "body", "type", "instance", "class", "side", "nt", "sign"])
    for i in range(N):
        wr.writerow([i, bodies[i], types[i], insts[i], klass[i], sides[i], nt[i], sign[i]])
print(f"OK -> neurons.csv  ({N:,} neurones)")
