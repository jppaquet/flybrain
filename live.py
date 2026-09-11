"""
live - simulation en continu du connectome pour piloter la mouche 3D.

Comme Eon Systems (cerveau LIF -> corps simulé), mais la lecture descend jusqu'aux
motoneurones de la VNC, présents dans MaleCNS. Une session tourne dans un thread :
la dynamique avance par tranches de FRAME_MS avec les stimulations en cours
(événements brefs, ou entrées continues comme le son -> organe de Johnston). Pour
chaque tranche on calcule le taux de décharge de « canaux » :
  - quelques neurones descendants, qui pilotent un contrôleur de marche procédural
    (le LIF n'a pas de générateur de rythme : c'est la même limite que chez Eon) ;
  - des groupes de motoneurones (pattes par muscle, ailes, cou, abdomen,
    proboscis, antennes), traduits directement en posture.
"""
import re, threading, time
import numpy as np
import engine

FRAME_MS = 10.0

LEGS = [("fl", "T1"), ("ml", "T2"), ("hl", "T3")]
LEG_FUNCS = [  # (clé, libellé, motif sur le type du motoneurone)
    ("pro", "promoteurs coxa", r"Sternal anterior rotator|promotor"),
    ("rem", "remoteurs coxa", r"Sternal posterior rotator|remotor"),
    ("trx", "extenseurs trochanter", r"^Tr extensor|^Sternotrochanter|^Tergotr|^TTMn$|^STTMm$"),
    ("trf", "fléchisseurs trochanter", r"Tr flexor"),
    ("tix", "extenseurs tibia", r"^Ti extensor"),
    ("tif", "fléchisseurs tibia", r"Ti flexor"),
    ("tad", "dépresseurs tarse", r"^Ta depressor|^ltm"),
    ("tal", "releveurs tarse", r"^Ta levator"),
]

# Stimulations proposées : sensorielles, ou « optogénétiques » (activation directe d'un DN)
EVENTS = [
    dict(key="loomL", label="Looming à gauche", kind="sens", query=r"^LC4$", field="type", side="L", hz=150, ms=300),
    dict(key="loomR", label="Looming à droite", kind="sens", query=r"^LC4$", field="type", side="R", hz=150, ms=300),
    # GRN qui, dans ce connectome à w_scale 0,5, recrutent MN9 (crible sur les 60 types gustatifs)
    dict(key="sugar", label="Goût (LB3c + taste pegs)", kind="sens", query=r"^(LB3c|claw_tpGRN)$", field="type", hz=150, ms=800),
    dict(key="sound", label="Son (JO-C/E)", kind="sens", query=r"^JO-(C|E)", field="type", hz=150, ms=500),
    dict(key="cva", label="Odeur cVA (ORN DA1)", kind="sens", query=r"^ORN_DA1$", field="type", hz=150, ms=800),
    dict(key="gf", label="Fibre géante", kind="opto", query=r"^DNp01$", field="type", hz=200, ms=100),
    dict(key="p09", label="DNp09 : marche", kind="opto", query=r"^DNp09$", field="type", hz=150, ms=1500),
    dict(key="mdn", label="MDN : marche arrière", kind="opto", query=r"^MDN$", field="type", hz=150, ms=1500),
    dict(key="a02L", label="DNa02 gauche", kind="opto", query=r"^DNa0[12]$", field="type", side="L", hz=150, ms=1000),
    dict(key="a02R", label="DNa02 droite", kind="opto", query=r"^DNa0[12]$", field="type", side="R", hz=150, ms=1000),
    dict(key="pip10", label="pIP10 : chant", kind="opto", query=r"^pIP10$", field="type", hz=150, ms=1500),
    dict(key="mn9", label="MN9 : proboscis", kind="opto", query=r"^MN9$", field="type", hz=150, ms=800),
]
AUDIO_INPUT = dict(query=r"^JO-(C|E)", field="type")


def select(C, query, field="type", side=None):
    idx = C.find(query, field)
    if side: idx = idx[np.array([C.side[i] == side for i in idx], dtype=bool)] if len(idx) else idx
    return idx


def channels(C):
    """Canaux lus à chaque tranche. Mis en cache sur l'objet Connectome."""
    if getattr(C, "_live_channels", None) is not None: return C._live_channels
    M = [i for i in range(C.N) if C.superclass[i] in ("vnc_motor", "cb_motor")]
    ty, sub, side = C.type, C.subclass, C.side
    motor = lambda pred: np.array([i for i in M if pred(i)], dtype=np.int32)
    out = []
    def add(key, label, group, idx):
        if len(idx): out.append(dict(key=key, label=label, group=group, idx=np.asarray(idx, np.int32)))
    add("fwd", "DNp09 (marche)", "locomotion", select(C, r"^DNp09$"))
    add("back", "MDN (arrière)", "locomotion", select(C, r"^MDN$"))
    for s in "LR":
        add(f"turn{s}", f"DNa01/02 {s}", "locomotion", select(C, r"^DNa0[12]$", side=s))
    add("gf", "Fibre géante", "saut", select(C, r"^DNp01$"))
    add("ttmn", "TTMn (saut)", "saut", select(C, r"^TTMn$"))
    add("mn9", "MN9 (rostre)", "proboscis", select(C, r"^MN9$"))
    add("pm", "Autres MN proboscis", "proboscis", motor(lambda i: sub[i] == "pm" and ty[i] != "MN9"))
    for s in "LR":
        add(f"wpow{s}", f"Vol DLM/DVM {s}", "ailes", motor(lambda i: sub[i] == "wm" and re.match(r"^D[LV]Mn", ty[i]) and side[i] == s))
        add(f"wstr{s}", f"Pilotage aile {s}", "ailes", motor(lambda i: sub[i] == "wm" and not re.match(r"^D[LV]Mn", ty[i]) and side[i] == s))
        add(f"hal{s}", f"Haltère {s}", "ailes", motor(lambda i: sub[i] == "hm" and side[i] == s))
        add(f"neck{s}", f"Cou {s}", "tête", motor(lambda i: sub[i] == "nm" and side[i] == s))
        add(f"ant{s}", f"Antenne {s}", "tête", motor(lambda i: sub[i] == "am" and side[i] == s))
        add(f"abd{s}", f"Abdomen {s}", "abdomen", motor(lambda i: sub[i] == "ad" and side[i] == s))
        for code, leg in LEGS:
            for f, flabel, rx in LEG_FUNCS:
                add(f"{leg}{s}_{f}", f"{leg}{s} {flabel}", "pattes",
                    motor(lambda i: sub[i] == code and side[i] == s and re.search(rx, ty[i])))
    C._live_channels = out
    return out


class Session:
    def __init__(self, C, params):
        self.C, self.q = C, engine.resolve_params(params)
        self.sim = engine.Stepper(C, self.q)
        self.chans = channels(C)
        self.frames, self.base = [], 0
        self.inputs = {}                    # nom -> dict(idx, hz, until)
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.last_poll, self.speed, self.error = time.time(), 0.0, None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stimulate(self, name, idx, hz, ms=None):
        with self.lock:
            if hz <= 0 or not len(idx): self.inputs.pop(name, None); return
            until = None if ms is None else self.sim.t_ms + ms
            self.inputs[name] = dict(idx=idx, hz=float(hz), until=until)

    def _run(self):
        n = max(1, int(round(FRAME_MS / self.q["dt"])))
        w0, s0 = time.time(), self.sim.t_ms
        try:
            while not self.stop_ev.is_set():
                if time.time() - self.last_poll > 20: break        # plus personne ne regarde
                with self.lock:
                    now = self.sim.t_ms
                    for k in [k for k, e in self.inputs.items() if e["until"] is not None and e["until"] <= now]:
                        del self.inputs[k]
                    drive = [(e["idx"], e["hz"]) for e in self.inputs.values()]
                    names = sorted(self.inputs)
                t = time.time()
                fired = self.sim.step(n, drive)
                counts = np.bincount(fired, minlength=self.C.N)
                k = 1000.0 / FRAME_MS
                rates = [round(float(counts[c["idx"]].sum()) * k / len(c["idx"]), 1) for c in self.chans]
                frame = [round(self.sim.t_ms, 1), rates, int(len(fired)), int(np.count_nonzero(counts)), names]
                with self.lock:
                    self.frames.append(frame)
                    if len(self.frames) > 6000:
                        self.frames = self.frames[3000:]; self.base += 3000
                dt = time.time() - t
                self.speed = 0.9 * self.speed + 0.1 * (FRAME_MS / 1000.0 / max(dt, 1e-6))
                ahead = (self.sim.t_ms - s0) / 1000.0 - (time.time() - w0)
                if ahead > 0: time.sleep(ahead)                     # jamais plus vite que le temps réel
                else: w0, s0 = time.time(), self.sim.t_ms           # en retard : on ne rattrape pas
        except Exception as e:
            self.error = str(e)
            raise

    def read(self, since, limit=300):
        self.last_poll = time.time()
        with self.lock:
            i = min(max(0, since - self.base), len(self.frames))
            out = self.frames[i:i + limit]
            return dict(frames=out, next=self.base + i + len(out), speed=round(min(self.speed, 1.0), 3),
                        t_ms=round(self.sim.t_ms, 1), inputs=sorted(self.inputs), error=self.error,
                        alive=self.thread.is_alive())

    def stop(self):
        self.stop_ev.set()


class Live:
    """Une seule session à la fois (serveur local, un seul utilisateur)."""
    def __init__(self, C):
        self.C, self.session, self.sid = C, None, 0

    def start(self, params):
        if self.session: self.session.stop()
        self.sid += 1
        self.session = Session(self.C, params)
        return dict(id=self.sid, frame_ms=FRAME_MS, params=self.session.q,
                     channels=[dict(key=c["key"], label=c["label"], group=c["group"], n=int(len(c["idx"])))
                               for c in self.session.chans],
                     events=[{k: v for k, v in e.items()} | dict(n=int(len(select(self.C, e["query"], e["field"], e.get("side")))))
                             for e in EVENTS])

    def get(self, sid):
        if self.session is None or int(sid) != self.sid: raise KeyError("session inconnue ou remplacée")
        return self.session

    def event(self, sid, key):
        e = next(e for e in EVENTS if e["key"] == key)
        s = self.get(sid)
        s.stimulate(f"ev:{key}", select(self.C, e["query"], e["field"], e.get("side")), e["hz"], e["ms"])

    def audio(self, sid, level):
        """Son capté par la page -> organe de Johnston, taux proportionnel au niveau (0..1)."""
        s = self.get(sid)
        idx = getattr(self, "_jo", None)
        if idx is None: idx = self._jo = select(self.C, AUDIO_INPUT["query"], AUDIO_INPUT["field"])
        s.stimulate("audio", idx, 150.0 * min(max(float(level), 0.0), 1.0), 400)
