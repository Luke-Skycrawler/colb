import warp as wp 
from geometry import Soup
from BDF1 import BDFHistory
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from contact import ContactSolverBase, XConstraint
from rbd_simple import RbdComplex, Inertia

'''
reference: [1] Detailed Rigid Body Simulation with Extended Position Based Dynamics
'''

eps = 1e-6
gravity = -9.8

@wp.struct
class RBDDelta: 
    dx: wp.array(dtype = vec3)
    dq: wp.array(dtype = vec4)
    cnt: wp.array(dtype = int)



@wp.kernel
def add_dlam_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), deltas: RBDDelta, dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    o = scalar(1.0)
    z = scalar(0.0)

    l0 = c.l0
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]
    
    b0 = soup.body[i0]
    b1 = soup.body[i2]

    R0 = Rq(p[b0].nxt.q)
    R1 = Rq(p[b1].nxt.q)
    
    c0 = p[b0].nxt.c
    c1 = p[b1].nxt.c
    
    x0 = R0 @ soup.xcs[i0] + c0
    x1 = R0 @ soup.xcs[i1] + c0
    x2 = R1 @ soup.xcs[i2] + c1
    x3 = R1 @ soup.xcs[i3] + c1

    dab = wp.closest_point_edge_edge(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), eps)
    v0 = wp.lerp(x0, x1, scalar(dab[0]))
    v1 = wp.lerp(x2, x3, scalar(dab[1]))

    v10 = v0 - v1
    dist = scalar(dab[2])
    m1 = mass[b0].m

    if dist < l0: 
        # Eqs. (2) - (5)
        n = v10 / (dist + scalar(eps))
        cc = (dist - l0 + scalar(eps))
        
        r1 = v0 - c0 
        r2 = v1 - c1 
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)
        w1 = z
        w2 = z
        m2 = mass[b1].m
        if m1 > z:
            w1 = o / m1 + wp.dot(a1, (o / mass[b0].J) * a1)
        if m2 > z:
            w2 = o / m2 + wp.dot(a2, (o / mass[b1].J) * a2)

        lam = c.lam
        common = -cc - c.alpha / (dt * dt) * lam
        denom = w1 + w2 + c.alpha / (dt * dt)
        dlam = common / denom 

        # Eqs. (6) - (9)
        xconstraints[i].lam = dlam + lam
        pp = dlam * n 
        
        if m1 > z:
            dx1 = pp / m1
            r1xp = wp.cross(r1, pp) / mass[b0].J
            dq1 = scalar(0.5) * quat_mult(vec4(r1xp.x, r1xp.y, r1xp.z, scalar(0.0)), p[b0].nxt.q)
            wp.atomic_add(deltas.dx, b0, dx1)
            wp.atomic_add(deltas.dq, b0, dq1)
            wp.atomic_add(deltas.cnt, b0, 1)
        
        if m2 > z:
            dx2 = -pp / m2
            r2xp = wp.cross(r2, pp) / mass[b1].J
            dq2 = scalar(-0.5) * quat_mult(vec4(r2xp.x, r2xp.y, r2xp.z, scalar(0.0)), p[b1].nxt.q)


            wp.atomic_add(deltas.dx, b1, dx2)
            wp.atomic_add(deltas.dq, b1, dq2)

            wp.atomic_add(deltas.cnt, b1, 1)


@wp.kernel
def predict_position(p: wp.array(dtype = BDFHistory), dt: scalar, mass: wp.array(dtype = Inertia)):
    i = wp.tid()
    z = scalar(0.0)
    # p[i].nxt.v += dt * vec3(z, scalar(gravity), z)
    p[i].nxt.c += p[i].nxt.v * dt
    if mass[i].m > z:
        p[i].nxt.c += dt * vec3(z, scalar(gravity), z) * dt * scalar(0.5)
    p[i].nxt.q += scalar(0.5) * wp.transpose(Gq(p[i].nxt.q)) @ p[i].nxt.omega * dt
    p[i].nxt.q = wp.normalize(p[i].nxt.q)

@wp.kernel
def add_dx_kernel(p: wp.array(dtype = BDFHistory), deltas: RBDDelta):
    i = wp.tid()
    if deltas.cnt[i] > 0:
        p[i].nxt.c += deltas.dx[i] / scalar(deltas.cnt[i])
        p[i].nxt.q += deltas.dq[i] / scalar(deltas.cnt[i])
        p[i].nxt.q = wp.normalize(p[i].nxt.q)

    deltas.dx[i] = vec3(scalar(0.0))
    deltas.dq[i] = vec4(scalar(0.0))
    deltas.cnt[i] = 0

@wp.kernel
def initialize_lam(contacts: wp.array(dtype = XConstraint)):
    i = wp.tid()
    contacts[i].lam = scalar(0.0)

@wp.kernel
def forward_states(history: wp.array(dtype = BDFHistory), dt: scalar):
    i = wp.tid()
    history[i].nxt.v = (history[i].nxt.c - history[i].now.c) / dt

    q_prev = history[i].now.q
    q_prev_inv = vec4(-q_prev.x, -q_prev.y, -q_prev.z, q_prev.w)
    dq = quat_mult(history[i].nxt.q, q_prev_inv)
    
    omega = scalar(2.0) * vec3(dq.x, dq.y, dq.z) / dt
    if dq.w < scalar(0.0):
        omega = -omega

    history[i].nxt.omega = omega
    history[i].now = history[i].nxt


class XPBDRbd(RbdComplex, ContactSolverBase):
    def __init__(self, h, meshes_filename):
        RbdComplex.__init__(self, h, meshes_filename)
        deltas = RBDDelta()

        deltas.dx = wp.zeros(self.n_bodies, dtype = vec3)
        deltas.dq = wp.zeros(self.n_bodies, dtype = vec4)
        deltas.cnt = wp.zeros((self.n_bodies,), dtype = int)

        self.deltas = deltas

        ContactSolverBase.__init__(self)

    def predict_position(self): 
        wp.launch(predict_position, self.n_bodies, inputs = [self.history, self.dt, self.inertia])

    def add_dlambda(self):
        wp.launch(add_dlam_kernel, (self.n_contacts, ), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.deltas, self.dt])

    def add_dx(self):
        wp.launch(add_dx_kernel, self.n_bodies, inputs = [self.history, self.deltas])

    def forward_states(self):
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1

    def initialize_multiplier(self):
        wp.launch(initialize_lam, (self.n_contacts, ), inputs = [self.contacts_new.list])

    def step(self):
        for ss in range(10):
            # substeps
            with wp.ScopedTimer("step"):
                self.predict_position()
                self.detect_collision()
                self.initialize_multiplier()

                for iter in range(2):
                    # xpbd iters 
                    self.deltas.dx.zero_()
                    self.deltas.dq.zero_()
                    self.deltas.cnt.zero_()

                    self.add_dlambda()
                    self.add_dx()
                self.forward_states()