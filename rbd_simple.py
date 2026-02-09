import numpy as np 
import warp as wp
from BDF1 import BDFHistory, cdot, qdot, vdot, wdot, dcdot_dc, dqdot_dq, forward_states
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq
from geometry import OBJComplex
# gravity = -9.8
gravity = 0.
eps = 1e-6
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
def init(history: wp.array(dtype = BDFHistory)):    
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    history[i].now.c = vec3(z, z, z)
    history[i].now.q = vec4(z, z, z, o)
    history[i].now.v = vec3(z, z, z)
    history[i].now.w = vec4(z, z, z, z)
    history[i].now.omega = vec3(z, o, o)

    history[i].nxt = history[i].now


class RigidBodyBase:
    def __init__(self):
        super().__init__()
        self.n_bodies = len(self.meshes_filename)
        n_bodies = self.n_bodies

        self.history = wp.zeros(n_bodies, dtype = BDFHistory)

class RbdComplex(RigidBodyBase, OBJComplex):
    '''
    NOTE: need to have self.meshes_filename and self.transforms predefined before calling super().__init__()
    '''
    def __init__(self, h, meshes_filename):
        self.meshes_filename = meshes_filename
        super().__init__()
        self.V0 = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)

        self.V = np.copy(self.V0)
        self.dt = h
        self.frame: int = 0
        self.n_nodes_per_body = self.n_nodes // self.n_bodies
        self.reset()
        
    def reset(self):
        wp.launch(init, self.n_bodies, inputs = [self.history])
        self.frame: int = 0

    def compute_V(self):
        V = wp.zeros_like(self.xcs)
        wp.launch(compute_V, self.n_nodes, inputs = [self.xcs, self.history, self.n_nodes_per_body, V])
        self.V = V.numpy()
        return self.V

@wp.struct
class RBDDelta: 
    dx: vec3
    dq: vec4

@wp.struct 
class XConstraint: 
    e0: int 
    e1: int 
    alpha: scalar
    lam: scalar

@wp.kernel
def add_dlam_kernel(p: wp.array(dtype = BDFHistory), xconstraints: wp.array(dtype = XConstraint), deltas: wp.array(dtype = wp.vec3), delta_counts: wp.array(dtype = int)):
    i = wp.tid()
    c = xconstraints[i]

    # l0 = c.l0
    # v10 = p[c.v0].x - p[c.v1].x
    # dist = wp.length(v10)
    # w0 = p[c.v0].w
    # w1 = p[c.v1].w

    # denom = w0 + w1 + c.alpha
    # common = -(dist - l0) - c.alpha * c.lam
    # dlam = common / denom

    # c.lam += dlam
    # gradient = v10 / (dist + eps)
    # dx0 = w0 * dlam * gradient
    # dx1 = -w1 * dlam * gradient

    # wp.atomic_add(deltas, c.v0, dx0)
    # wp.atomic_add(deltas, c.v1, dx1)


    # wp.atomic_add(delta_counts, c.e0, 1)
    # wp.atomic_add(delta_counts, c.e1, 1)


@wp.kernel
def predict_position(p: wp.array(dtype = BDFHistory), dt: scalar):
    i = wp.tid()
    z = scalar(0.0)
    p[i].nxt.v += dt * vec3(z, scalar(gravity), z)
    p[i].nxt.c += p[i].nxt.v * dt
    p[i].nxt.q += scalar(0.5) * wp.transpose(Gq(p[i].nxt.q)) @ p[i].nxt.omega * dt

@wp.kernel
def add_dx_kernel(p: wp.array(dtype = BDFHistory), deltas: wp.array(dtype = RBDDelta), delta_counts: wp.array(dtype = int)):
    i = wp.tid()
    if delta_counts[i] > 0:
        p[i].nxt.c += deltas[i].dx / scalar(delta_counts[i])
        p[i].nxt.q += deltas[i].dq / scalar(delta_counts[i])
        p[i].nxt.q = wp.normalize(p[i].nxt.q)

        deltas[i].dx = vec3(scalar(0.0))
        deltas[i].dq = vec4(scalar(0.0))
        delta_counts[i] = 0

class XPBDRbd(RbdComplex):
    def __init__(self, h, meshes_filename):
        super().__init__(h, meshes_filename)
        self.deltas = wp.zeros(self.n_bodies, dtype = RBDDelta)
        self.delta_counts = wp.zeros((self.n_bodies,), dtype = int)

    def predict_position(self): 
        wp.launch(predict_position, self.n_bodies, inputs = [self.history, self.dt])
    def initialize_multiplier(self):
        pass 

    def add_dlambda(self):
        # wp.launch(add_dlam_kernel, self.n_bodies, inputs = [self.history, self.deltas, self.delta_counts])
        pass
    
    def add_dx(self):
        wp.launch(add_dx_kernel, self.n_bodies, inputs = [self.history, self.deltas, self.delta_counts])

    def step(self):
        for ss in range(1):
            # substeps

            self.predict_position()
            self.initialize_multiplier()
            for iter in range(10):
                # xpbd iters 
                self.deltas.zero_()
                self.add_dlambda()
                self.add_dx()

            wp.launch(forward_states, self.n_bodies, inputs = [self.history])
            # wp.copy(self.states_now, self.states_nxt)
            # print(f'   states_nxt q = {self.history.numpy()["nxt"]["q"]}')        
            self.frame += 1