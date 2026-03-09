import warp as wp 
import numpy as np 
from primal import PrimalRbd, XConstraint, stiffness, eps, gravity, compute_u_minus_utilde
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from xpbd_contact import fetch_b0b1,  fetch_dist_n_r0r1
from BDF1 import BDFHistory
from rbd_simple import Inertia
from geometry import Soup

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar, color: int, colors: wp.array(dtype = int)):
    i = wp.tid()
    c = colors[i]
    # n_bodies = history.shape[0]
    if c == color:
        history[i].nxt.v -= alpha * du[i * 2] 
        history[i].nxt.c = history[i].now.c + dt * history[i].nxt.v
        
        history[i].nxt.omega -= alpha * du[i * 2 + 1]
        history[i].nxt.q = history[i].now.q + scalar(0.5) * wp.transpose(Gq(history[i].nxt.q)) @ history[i].nxt.omega * dt
        history[i].nxt.q = wp.normalize(history[i].nxt.q)


class VBDRbd(PrimalRbd):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)
        self.define_color()
        self.alpha = 1.0
    
    def define_color(self):
        angles = np.array([
            [90., 0., 90.],
            [0., 0., 90.],
            [90., 90., 0.],
        ])
        colors = []
        for obj in self.kinetic_objects:
            err = np.abs(obj.euler - angles)
            color = np.argmin(np.sum(err, axis=1))
            colors.append(color)

        colors = np.array(colors)
        self.colors = wp.array(colors, dtype = int)

    def add_du(self, alpha, color): 
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt, color, self.colors])


    def step(self):
        for ss in range(10):
            newton = True
            iter = 0
            max_iter = 8
            self.detect_collision()
            while newton: 
                for color in range(3):
                    self.compute_preconditioner()
                    self.compute_rhs()
                    
                    du_norm = self.compute_du() 
                    self.add_du(self.alpha, color)
                iter += 1
                # print(f"    iter: {iter}, du norm: {du_norm}")
                newton = not (du_norm < 1e-5 or iter >= max_iter)
        
            self.forward_states()
