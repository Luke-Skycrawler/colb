import numpy as np 
import warp as wp 
from scalar_types import *
from BDF1 import BDFAffine, AffineState
from rbd_simple import Inertia, _set_inertia_kernel
from geometry import Soup
from utils.scene import JSONComplex

class AffineBodyBase: 
    def __init__(self, *args):
        super().__init__(*args)
        n_bodies = self.n_bodies

        self.history = wp.zeros(n_bodies, dtype = BDFAffine)
        self.inertia = wp.zeros(n_bodies, dtype = Inertia)
        self.set_inertia()
        
    def set_inertia(self): 
        m = wp.zeros((self.n_bodies, ), dtype = scalar)
        mnp = np.array([obj.mass for obj in self.kinetic_objects], dtype = float)
        m.assign(mnp / 1000.0)
        wp.launch(_set_inertia_kernel, self.n_bodies, inputs = [self.inertia, m])



@wp.kernel
def compute_V(geo: Soup, history: wp.array(dtype = BDFAffine)):
    i = wp.tid()
    bd = geo.body[i]
    qi = history[bd].now.q
    ci = history[bd].now.c
    R = qi
    v0 = geo.xcs[i]
    v1 = (R @ v0) + ci
    geo.x_transformed[i] = v1

@wp.kernel
def init(history: wp.array(dtype = BDFAffine), p: wp.array(dtype = vec3), mass: wp.array(dtype = Inertia)):    
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(3.0)
    history[i].now.c = vec3(scalar(p[i].x), scalar(p[i].y), scalar(p[i].z))
    history[i].now.q = wp.identity(3, dtype = scalar)
    history[i].now.v = vec3(z, z, z)
    history[i].now.qdot = mat33(z, o, z, -o, z, z, z, z, z)

    history[i].nxt = history[i].now

    
class AbdComplex(AffineBodyBase, JSONComplex):
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
        pwp = wp.array(p, dtype = vec3)
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
