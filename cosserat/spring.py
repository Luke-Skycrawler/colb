import warp as wp 
from viewer import PSViewer
import polyscope as ps
from .cosserat import StableCosserat, Node, Seg, from_to, n_fixed, scalar, vec3, mat33, quat, e3
from .rod_contact import PrimalRod

n_partition = 20
n_rings = 20
spring_length = scalar(10.0)
spring_r = scalar(1.0)
@wp.kernel
def reset_nodes(x: wp.array(dtype = Node)):
    n_nodes = x.shape[0]
    dy = spring_length / scalar(n_nodes - 1)
    i = wp.tid()

    # set node position
    theta = scalar(wp.pi * 2.0) * scalar(i) / scalar(n_partition)
    x[i].x = vec3(wp.cos(theta) * spring_r, wp.sin(theta) * spring_r, scalar(i) * dy)
    x[i].x0 = x[i].x
    x[i].mass = scalar(1.0)
    if i <= n_fixed: 
        x[i].mass = scalar(-1.0)

    z3 = vec3(scalar(0.0), scalar(0.0), scalar(0.0))
    x[i].v = z3
    x[i].v0 = z3
    x[i].last = i - 1

@wp.kernel
def reset_segs(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg)):
    # n_segs = seg.shape[0]
    # for i in range(n_segs):
    i = wp.tid()

    p0 = x[i].x
    if i < x.shape[0] - 1: 
        p1 = x[i + 1].x
    
        seg[i].l = wp.length(p1 - p0)
        seg[i].w = wp.quaternion(scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar)
        seg[i].q = wp.normalize(from_to(e3, wp.normalize(p1 - p0)))
        seg[i].q0 = seg[i].q
        seg[i].nxt = i + 1
    else: 
        seg[i].nxt = -1

@wp.kernel
def init_rest_frame(seg: wp.array(dtype = Seg)):
    i = wp.tid()
    if i > 0 and seg[i].nxt >= 0:
        seg[i].q_rest = wp.quat_inverse(seg[i - 1].q) * seg[i].q
    else: 
        seg[i].q_rest = wp.quat_identity(scalar)
        
    
class CosseratHelix(PrimalRod):
    def __init__(self, dt):
        n_nodes = n_partition * n_rings
        super().__init__(n_nodes, dt)

    def reset(self):
        wp.launch(reset_nodes, self.n_nodes, inputs = [self.nodes])
        wp.launch(reset_segs, self.n_segs, inputs = [self.nodes, self.segs])
        wp.launch(init_rest_frame, self.n_segs, inputs = [self.segs])

        # print(f"position = {self.nodes}")
        # print(f"quats = {self.segs.numpy()['q']}")
        
        self.frame = 0


if __name__ == '__main__':
    wp.init()
    ps.init()
    dt = 1e-3
    sim = CosseratHelix(dt)
    # sim = PrimalRod(10, dt)
    viewer = PSViewer(sim)
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()