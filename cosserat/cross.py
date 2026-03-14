import warp as wp 
from scalar_types import * 
from .cosserat import Node, Seg, n_fixed, from_to
from viewer import PSViewer
import polyscope as ps
import numpy as np
from .cosserat import StableCosserat
from .rod_contact import RodContact, PrimalRod

n_segs_per_thread = 40
o = scalar(0.0)
z = scalar(1.0)
@wp.kernel
def reset(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg)):
    i = wp.tid()
    e3 = wp.vec3(0.0, 0.0, 1.0)
    ii = i
    mid = wp.vec3(0.0, 0.0, 0.0)
    vy = scalar(1.0)
    if i >= n_segs_per_thread:
        e3 = wp.vec3(1.0, 0.0, 0.0)
        ii -= n_segs_per_thread
        mid += wp.vec3(0.0, 0.4, 0.0)
        vy = scalar(-1.0)

    li = 10.0 / float(n_segs_per_thread)
    n_nodes = x.shape[0]
    x[i].x = vec3(e3 * (float(ii) * li - 5.0) + mid)
    x[i].x0 = x[i].x
    x[i].v = vec3(o, vy, o)
    x[i].v0 = vec3(o, vy, o)
    if ii > n_fixed and ii < n_segs_per_thread - 1:
        x[i].mass = scalar(1.0)
    else: 
        x[i].mass = scalar(-1.0)

    seg[i].q = from_to(vec3(o, o, z), vec3(e3))
    seg[i].q0 = seg[i].q
    seg[i].w = wp.quaternion(o, o, o, o, scalar)
    seg[i].q_rest = wp.quat_identity(scalar)
    if ii < n_segs_per_thread - 1:
        seg[i].l = scalar(li)
    else:
        # segment does not exist
        seg[i].l = scalar(-1.0)

@wp.kernel
def update_prescribed(prescribed_motion: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    if i < n_segs_per_thread: 
        prescribed_motion[i] = vec3(o, z, o) * dt
    else:
        prescribed_motion[i] = -vec3(o, z, o) * dt

class ContactTest(StableCosserat):
    def __init__(self, n_nodes, dt):
        super().__init__(n_nodes, dt)
    def reset(self):
        wp.launch(update_prescribed, self.n_nodes, inputs = [self.prescribed_motion, self.dt])
        wp.launch(reset, self.n_nodes, inputs = [self.nodes, self.segs])
        self.frame = 0

class ContactTestPrimal(PrimalRod):
    def __init__(self, n_nodes, dt):
        super().__init__(n_nodes, dt)  

    def reset(self):
        wp.launch(update_prescribed, self.n_nodes, inputs = [self.prescribed_motion, self.dt])
        wp.launch(reset, self.n_nodes, inputs = [self.nodes, self.segs])
        self.frame = 0

if __name__ == '__main__':
    wp.config.max_unroll = 0
    wp.init()
    ps.init()
    dt = 2e-3
    # rod = ContactTest(n_segs_per_thread * 2, dt)
    rod = ContactTestPrimal(n_segs_per_thread * 2, dt)
    viewer = PSViewer(rod)
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()