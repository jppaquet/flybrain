"""
world - the physical world around the fly: a kart on a circular track, driven by the
fly's legs. Shared by the live 3D page (live.py) and the learning environments (flyenv.py),
so a policy trained offline drives exactly the same kart in the browser.

Only motor neurons move the kart. Turning their rates into controls is the "body" of this
world (hand-written, like the tripod gait):
  - steering wheel: the front legs push its rim (coxa promotors); the left front leg turns
    it to the right, the right front leg to the left, and a spring recentres it;
  - accelerator: pressed by the right hind leg extending (trochanter extensors);
  - brake: a lever pulled by the left hind leg flexing (tibia flexors) - no descending
    neuron extends the left hind leg cleanly in this connectome, one flexes it.
"""
import math


def act(r, ref):
    """Rate (Hz) -> activation 0..1 (63 % at `ref`)."""
    return 1.0 - math.exp(-max(0.0, r) / ref)


# motor channels (keys of live.channels) read for each control; ref: rate giving 63 %
CONTROLS = dict(
    wheel_right=dict(read=["T1L_pro"], ref=12.0),   # left front leg pushes the rim
    wheel_left=dict(read=["T1R_pro"], ref=12.0),    # right front leg pushes the rim
    throttle=dict(read=["T3R_trx"], ref=12.0),      # right hind leg extends
    brake=dict(read=["T3L_tif"], ref=10.0),         # left hind leg flexes
)


def level(rates, c):
    return act(sum(rates.get(k, 0.0) for k in c["read"]) / len(c["read"]), c["ref"])


def controls(rates):
    """{channel key: Hz} -> dict(push -1..1 (+ turns right), throttle 0..1, brake 0..1)."""
    return dict(push=level(rates, CONTROLS["wheel_right"]) - level(rates, CONTROLS["wheel_left"]),
                throttle=level(rates, CONTROLS["throttle"]), brake=level(rates, CONTROLS["brake"]))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    """Circular road of radius R centred on (R, 0, 0): the kart starts at the origin
    heading +Z (the fly's own frame: +X is its left), and the road curves to the left."""
    def __init__(self, radius=24.0, half_width=4.0):
        self.R, self.W, self.cx, self.cz = radius, half_width, radius, 0.0

    def polar(self, x, z):
        dx, dz = x - self.cx, z - self.cz
        return math.hypot(dx, dz), math.atan2(dz, dx)

    def offset(self, x, z):
        """Signed distance from the centre line (+ = outside of the curve)."""
        return self.polar(x, z)[0] - self.R

    def heading_error(self, x, z, yaw):
        """Kart heading minus the road direction (rad, + = pointing to the left of the road)."""
        return wrap(yaw - (math.pi - self.polar(x, z)[1]))

    def to_dict(self):
        return dict(kind="circle", radius=self.R, half_width=self.W, cx=self.cx, cz=self.cz)


class Kart:
    """Kinematic bicycle model. Scene units (the fly is ~2.5 long), seconds."""
    WHEELBASE, MAX_STEER = 3.0, 0.5            # steer angle at full wheel (rad)
    ACCEL, BRAKE, DRAG, VMAX = 6.0, 12.0, 0.25, 14.0
    WHEEL_GAIN, WHEEL_SPRING = 3.0, 2.0        # wheel angle: d/dt = gain * push - spring * angle

    def __init__(self, track=None):
        self.track = track or Track()
        self.reset()

    def reset(self):
        self.x = self.z = self.yaw = self.v = self.wheel = 0.0
        self.push = self.throttle = self.brake = 0.0
        self.dist, self.theta = 0.0, self.track.polar(0.0, 0.0)[1]

    def update(self, ctl, dt):
        """Advances dt seconds with the controls from `controls()`; returns the distance
        gained along the road (negative when driving the wrong way)."""
        self.push, self.throttle, self.brake = ctl["push"], ctl["throttle"], ctl["brake"]
        self.wheel = min(1.0, max(-1.0, self.wheel + (self.WHEEL_GAIN * self.push - self.WHEEL_SPRING * self.wheel) * dt))
        a = self.ACCEL * self.throttle - (self.BRAKE * self.brake if self.v > 0 else 0.0) - self.DRAG * self.v
        self.v = min(max(self.v + a * dt, 0.0), self.VMAX)
        self.yaw = wrap(self.yaw - self.v * math.tan(self.wheel * self.MAX_STEER) / self.WHEELBASE * dt)
        self.x += math.sin(self.yaw) * self.v * dt
        self.z += math.cos(self.yaw) * self.v * dt
        th = self.track.polar(self.x, self.z)[1]
        gained = -wrap(th - self.theta) * self.track.R
        self.theta = th
        self.dist += gained
        return gained

    def offset(self):
        return self.track.offset(self.x, self.z)

    def state(self):
        return dict(x=self.x, z=self.z, yaw=self.yaw, speed=self.v, wheel=self.wheel, push=self.push,
                    throttle=self.throttle, brake=self.brake, offset=self.offset(),
                    heading_error=self.track.heading_error(self.x, self.z, self.yaw), distance=self.dist)

    FRAME_KEYS = ["x", "z", "yaw", "speed", "wheel", "throttle", "brake", "offset", "distance", "push", "heading_error"]

    def frame(self):
        s = self.state()
        return [round(s[k], 3) for k in self.FRAME_KEYS]


class Driver:
    """The fly's motor neurons -> kart, one 10 ms simulation frame at a time. live.py and
    flyenv.py both call frame() with the same per-frame channel rates, so the kart behaves
    identically in the browser and in training."""
    TAU_MS = 30.0                       # smoothing of the motor rates (as in the 3D page)
    KEYS = sorted({k for c in CONTROLS.values() for k in c["read"]})

    def __init__(self, track=None):
        self.kart = Kart(track)
        self.S = {}

    def reset(self):
        self.kart.reset()
        self.S = {}

    def frame(self, rates, dt_ms):
        """rates: {channel key: Hz} of this frame. Returns the distance gained along the road."""
        k = 1.0 - math.exp(-dt_ms / self.TAU_MS)
        for key in self.KEYS:
            s = self.S.get(key, 0.0)
            self.S[key] = s + (rates.get(key, 0.0) - s) * k
        return self.kart.update(controls(self.S), dt_ms / 1000.0)
