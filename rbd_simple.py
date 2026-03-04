import numpy as np 
import warp as wp
from BDF1 import BDFHistory, cdot, qdot, vdot, wdot, dcdot_dc, dqdot_dq, forward_states
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq
from geometry import OBJComplex
from xpbd_contact import RBDDelta, XConstraint, Inertia, XPBDContact, add_dlam_kernel
from utils.scene import JSONComplex
# gravity = -9.8
gravity = 0.
length = scalar(0.2)

@wp.kernel
def compute_V(xcs: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), n_nodes_per_body: int, V: wp.array(dtype = vec3)):
    i = wp.tid()
    bd = i // n_nodes_per_body
    qi = history[bd].now.q
    ci = history[bd].now.c
    R = Rq(qi)
    v0 = xcs[i]
    v1 = (R @ v0) + ci
    V[i] = v1

@wp.kernel
def init(history: wp.array(dtype = BDFHistory), p: wp.array(dtype = wp.vec3)):    
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    # history[i].now.c = vec3(z, z, z)
    history[i].now.c = vec3(scalar(p[i].x), scalar(p[i].y), scalar(p[i].z))
    history[i].now.q = vec4(z, z, z, o)
    history[i].now.v = vec3(z, z, z)
    history[i].now.w = vec4(z, z, z, z)
    history[i].now.omega = vec3(z, o, o)

    history[i].nxt = history[i].now

@wp.kernel
def _set_inertia_kernel(meta: wp.array(dtype = Inertia)):
    i = wp.tid()
    meta[i].m = scalar(1.0)
    # meta[i].J = mat33(scalar(0.4) * meta[i].m * length * length)
    meta[i].J = scalar(0.4) * meta[i].m * length * length

class RigidBodyBase:
    def __init__(self, *args):
        super().__init__(*args)
        n_bodies = self.n_bodies

        self.history = wp.zeros(n_bodies, dtype = BDFHistory)
        self.inertia = wp.zeros(n_bodies, dtype = Inertia)
        self.set_inertia()

    def set_inertia(self): 
        wp.launch(_set_inertia_kernel, self.n_bodies, inputs = [self.inertia])

class RbdComplex(RigidBodyBase, JSONComplex):
    '''
    NOTE: need to have self.meshes_filename and self.transforms predefined before calling super().__init__()
    '''
    def __init__(self, h, config_file):
        super().__init__(config_file)
        self.V0 = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)

        self.V = np.copy(self.V0)
        self.dt = h
        self.frame: int = 0
        self.n_nodes_per_body = self.n_nodes // self.n_bodies
        self.reset()
        
    def reset(self):
        p = np.array([obj.p for obj in self.kinetic_objects])
        pwp = wp.array(p, dtype = wp.vec3)
        # print(f"p = {p}")
        wp.launch(init, self.n_bodies, inputs = [self.history, pwp])
        self.frame: int = 0
        self.compute_V()

    def compute_V(self):
        # V = wp.zeros_like(self.xcs)
        V = self.soup.x_transformed
        wp.launch(compute_V, self.n_nodes, inputs = [self.xcs, self.history, self.n_nodes_per_body, V])
        self.V = V.numpy()
        return self.V


@wp.kernel
def predict_position(p: wp.array(dtype = BDFHistory), dt: scalar):
    i = wp.tid()
    z = scalar(0.0)
    p[i].nxt.v += dt * vec3(z, scalar(gravity), z)
    p[i].nxt.c += p[i].nxt.v * dt
    p[i].nxt.q += scalar(0.5) * wp.transpose(Gq(p[i].nxt.q)) @ p[i].nxt.omega * dt

@wp.kernel
def add_dx_kernel(p: wp.array(dtype = BDFHistory), deltas: RBDDelta, delta_counts: wp.array(dtype = int)):
    i = wp.tid()
    if delta_counts[i] > 0:
        p[i].nxt.c += deltas.dx[i] / scalar(delta_counts[i])
        p[i].nxt.q += deltas.dq[i] / scalar(delta_counts[i])
        p[i].nxt.q = wp.normalize(p[i].nxt.q)

        deltas.dx[i] = vec3(scalar(0.0))
        deltas.dq[i] = vec4(scalar(0.0))
        delta_counts[i] = 0

class XPBDRbd(RbdComplex, XPBDContact):
    def __init__(self, h, meshes_filename):
        RbdComplex.__init__(self, h, meshes_filename)
        deltas = RBDDelta()

        deltas.dx = wp.zeros(self.n_bodies, dtype = vec3)
        deltas.dq = wp.zeros(self.n_bodies, dtype = vec4)
        self.deltas = deltas

        self.delta_counts = wp.zeros((self.n_bodies,), dtype = int)
        XPBDContact.__init__(self)       

    def predict_position(self): 
        wp.launch(predict_position, self.n_bodies, inputs = [self.history, self.dt])

    def add_dlambda(self):
        wp.launch(add_dlam_kernel, (self.n_contacts, ), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.deltas, self.delta_counts, self.dt])

    def add_dx(self):
        wp.launch(add_dx_kernel, self.n_bodies, inputs = [self.history, self.deltas, self.delta_counts])

    def step(self):
        for ss in range(1):
            # substeps

            self.predict_position()
            self.detect_collision()
            self.initialize_multiplier()

            for iter in range(10):
                # xpbd iters 
                self.deltas.dx.zero_()
                self.deltas.dq.zero_()

                self.add_dlambda()
                self.add_dx()

            wp.launch(forward_states, self.n_bodies, inputs = [self.history])
            # wp.copy(self.states_now, self.states_nxt)
            # print(f'   states_nxt q = {self.history.numpy()["nxt"]["q"]}')        
            self.frame += 1

    def initialize_multiplier(self):
        self.deltas.dx.zero_()
        self.deltas.dq.zero_()
        self.delta_counts.zero_()
        
        