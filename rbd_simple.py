import numpy as np 
import warp as wp
from BDF1 import BDFHistory
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq
from utils.scene import JSONComplex
from geometry import Soup
length = scalar(0.4)

@wp.struct 
class Inertia: 
    m: scalar
    J: scalar

@wp.kernel
def compute_V(geo: Soup, history: wp.array(dtype = BDFHistory)):
    i = wp.tid()
    bd = geo.body[i]
    qi = history[bd].now.q
    ci = history[bd].now.c
    R = Rq(qi)
    v0 = geo.xcs[i]
    v1 = (R @ v0) + ci
    geo.x_transformed[i] = v1

@wp.kernel
def init(history: wp.array(dtype = BDFHistory), p: wp.array(dtype = wp.vec3), mass: wp.array(dtype = Inertia)):    
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    t = scalar(3.0)
    # history[i].now.c = vec3(z, z, z)
    history[i].now.c = vec3(scalar(p[i].x), scalar(p[i].y), scalar(p[i].z))
    history[i].now.q = vec4(z, z, z, o)
    history[i].now.v = vec3(z, z, z)
    history[i].now.w = vec4(z, z, z, z)
    if mass[i].m > scalar(0.0): 
        history[i].now.omega = vec3(z, z, t)
    else:
        history[i].now.omega = vec3(z)
    history[i].nxt = history[i].now

@wp.kernel
def _set_inertia_kernel(meta: wp.array(dtype = Inertia), m: wp.array(dtype = scalar)):
    i = wp.tid()
    meta[i].m = wp.max(m[i], scalar(0.0))
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
        m = wp.zeros((self.n_bodies, ), dtype = scalar)
        mnp = np.array([obj.mass for obj in self.kinetic_objects], dtype = float)
        m.assign(mnp / 1000.0)
        wp.launch(_set_inertia_kernel, self.n_bodies, inputs = [self.inertia, m])

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
        wp.launch(init, self.n_bodies, inputs = [self.history, pwp, self.inertia])
        self.frame: int = 0
        self.compute_V()

    def compute_V(self, ret = True):
        V = self.soup.x_transformed
        wp.launch(compute_V, self.n_nodes, inputs = [self.soup, self.history])
        if ret:
            self.V = V.numpy()
            return self.V
        else:
            return None
