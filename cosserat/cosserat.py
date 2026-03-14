import warp as wp
from viewer import PSViewer
import polyscope as ps
from scalar_types import *

o = scalar(1.0)
z = scalar(0.0)

g = vec3(0.0, -10.0, 0.0)
e3 = vec3(0.0, 0.0, 1.0)

# kss = scalar(1.0e8)
# kbt = scalar(1.0e8)
kss = scalar(1.0e6)
kbt = scalar(1.0e4)

n_fixed = 1

length = scalar(10.0)
'''
segment ei = (ei0, nxt)
nxt == -1 means the segment does not exist
'''
@wp.struct
class Seg: 
    q: quat
    l: scalar 
    # kss: scalar
    q_rest: quat 
    # kbt: scalar
    w: quat
    q0: quat
    # lam: scalar
    nxt: int

@wp.struct 
class Node: 
    x: vec3
    x0: vec3
    v: vec3
    v0: vec3
    dx: vec3
    Hi: mat33
    fi: vec3
    mass: scalar
    last: int

@wp.func 
def compute_y(x: Node, dt: scalar):
    '''
    inertia
    '''
    return x.x0 + dt * x.v + dt * dt * g

@wp.func
def compute_w(seg: Seg, dt: scalar):
    w = seg.w * dt + seg.q0
    return w / wp.length(w)

@wp.kernel
def init_positions(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), dt: scalar, prescribed_motion: wp.array(dtype = vec3)): 
    i = wp.tid()
    if x[i].mass > 0.0:
        eg = vec3(z, scalar(-1.0), z)
        at = wp.dot(eg, (x[i].v - x[i].v0) / dt)
        a_tilde = at / scalar(10.0)
        a_tilde = wp.clamp(a_tilde, z, o)
        a = g * a_tilde
        y = x[i].x0 + dt * x[i].v + dt * dt * g
        x[i].x = y
        if seg[i].nxt >= 0:
            seg[i].q = compute_w(seg[i], dt)
    else: 
        x[i].x += prescribed_motion[i]

@wp.func
def compute_Css(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), i: int):
    '''
    shear and stretch constraint 
    did not check nxt == -1 case, avoid calling this func when seg[i] not exist
    '''
    # x1 = x[i + 1].x
    x0 = x[i].x
    nxt = seg[i].nxt
    x1 = x[nxt].x 
    li = seg[i].l
    return (x1 - x0) / li - wp.quat_rotate(seg[i].q, e3)

@wp.kernel
def position_update(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), dt: scalar, odd: int):
    '''
    Eq.6 
    x = argmin 1/2h^2 ||x - y||_M^2 + Ess_i(x)

    xi = yi - h^2 kss / mi (Css_{i-1} / l_{i-1} - Css_i / l_i)
    '''
    i = wp.tid()
    if i % 2 == odd and x[i].mass > 0.0:
        last = x[i].last
        lhs = z
        c0 = vec3(z)
        if last >= 0:
            lhs = x[i].mass / (dt * dt) + kss / seg[last].l
            c0 = compute_Css(x, seg, last)
        # rhs = mass / (dt * dt) * (yi - x[i].x) + kss * c0
        rhs = kss * c0 - g * x[i].mass
        if i < x.shape[0] - 1:
            c1 = compute_Css(x, seg, i)
            rhs -= kss * c1
            lhs += kss / seg[i].l
        
        x[i].x += -rhs / lhs

# @wp.func
# def phi(seg: wp.array(dtype = Seg), i: int) -> scalar:
#     '''
#     phi in Eq.4
#     '''
#     ret = 1.0
#     qbar = wp.quat_inverse(seg[i].q)
#     q1 = qbar * seg[i + 1].q
#     q2 = seg[i].q_rest
#     cond_1 = wp.dot(vec3(q1.x, q1.y, q1.z), vec3(q2.x, q2.y, q2.z)) > 0.0
#     if not cond_1: 
#         ret = -1.0
#     # cond = wp.length_sq(q1 - q2) > wp.length_sq(q1 + q2)
#     # if cond:
#         # ret = -1.0
#     return ret

@wp.func
def from_to(f: vec3, t: vec3) -> quat:
    h = scalar(0.5) * (f + t)
    l = wp.length_sq(h)
    if l > 0: 
        h *= o / wp.sqrt(l)
    
    return quat(wp.cross(f, h), wp.dot(f, h))

@wp.kernel
def quat_update(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), dt: scalar, odd: int):
    i = wp.tid()
    if i % 2 == odd: 
        v = -scalar(2.0) * kss * (x[i + 1].x - x[i].x) 
        b = wp.quaternion(z, z, z, z, scalar)
        start = max(0, i - 1)
        end = min(i + 1, seg.shape[0] - 1)
        for s in range(start, end):
            oid = i - 1 
            if s == i:
                oid = i + 1
            if seg[oid].l < 0.0 or seg[s].l < 0.0:
                continue
            oq = seg[oid].q
            qq = wp.quat_inverse(oq) * seg[i].q
            if i == s:
                qq = wp.quat_inverse(qq)
            
            phii = scalar(-1.0)
            ang = seg[s].q_rest
            if wp.dot(wp.vec4d(qq.x, qq.y, qq.z, qq.w), wp.vec4d(ang.x, ang.y, ang.z, ang.w)) > 0.0:
                phii = o
            if s == i:
                ang = wp.quat_inverse(ang)
            li = seg[s].l
            b += (scalar(4.0) * kbt / li) * (phii * oq * ang)
        if start == end:
            seg[i].q = from_to(e3, wp.normalize(-v))
        else: 
            lam = wp.length(v) + wp.length(b)
            v_quat = wp.quaternion(v, z, scalar)
            e3q = wp.quaternion(e3, z, scalar)
            vbe3 = v_quat * b * e3q
            qi = (vbe3 + lam * b)
            seg[i].q = qi / wp.length(qi)
        
@wp.kernel
def update_v(x: wp.array(dtype = Node), segs: wp.array(dtype = Seg), dt: scalar):
    i = wp.tid()
    x[i].v0 = x[i].v
    x[i].v = (x[i].x - x[i].x0) / dt
    x[i].x0 = x[i].x 
    if i < segs.shape[0]:
        segs[i].w = (segs[i].q - segs[i].q0) / dt
        segs[i].q0 = segs[i].q
    

@wp.kernel
def reset(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg)):
    i = wp.tid()
    li = length / scalar(x.shape[0] - 1)
    n_nodes = x.shape[0]
    x[i].x = e3 * scalar(i) * li
    x[i].x0 = x[i].x
    x[i].v = vec3(z)
    x[i].v0 = vec3(z)
    x[i].last = i - 1
    if i > n_fixed:
        x[i].mass = scalar(1.0)
    else: 
        x[i].mass = scalar(-1.0)
    if i < n_nodes - 1:
        seg[i].l = li
        seg[i].q = wp.quat_identity(scalar)
        seg[i].q0 = wp.quat_identity(scalar)
        seg[i].w = wp.quaternion(z, z, z, z, scalar)
        seg[i].q_rest = wp.quat_identity(scalar)
        seg[i].nxt = i + 1
    # else:
    #     seg[i].nxt = -1

class StableCosserat:
    def __init__(self, n_nodes, dt): 
        self.max_iters = 4
        self.dt = dt
        self.n_nodes = n_nodes
        self.n_segs = self.n_nodes - 1

        self.nodes = wp.zeros((self.n_nodes), dtype = Node)
        self.segs = wp.zeros((self.n_segs), dtype = Seg)
        self.frame = 0
        self.prescribed_motion = wp.zeros((self.n_nodes), dtype = vec3)
        self.reset()

    def reset(self):
        wp.launch(reset, self.n_nodes, inputs = [self.nodes, self.segs])
        self.frame = 0

    def vbd_step_position(self):
        for odd in range(2):
            wp.launch(position_update, self.n_nodes, inputs = [self.nodes, self.segs, self.dt, odd])
    
    def prestep(self): 
        # adaptive initialization
        wp.launch(init_positions, self.n_nodes, inputs = [self.nodes, self.segs, self.dt, self.prescribed_motion])

    def step(self):
        n_substeps = 1
        for ss in range(n_substeps): 
            with wp.ScopedTimer(f"step"):
                self.prestep()
                for it in range(self.max_iters): 
                    self.vbd_step_position()
                    for odd in range(2): 
                        wp.launch(quat_update, self.n_segs, inputs = [self.nodes, self.segs, self.dt, odd])

                wp.launch(update_v, self.n_nodes, inputs = [self.nodes, self.segs, self.dt])
                self.frame += 1
    

