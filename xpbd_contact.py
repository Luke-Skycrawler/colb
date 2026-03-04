import warp as wp 
from geometry import Soup
from BDF1 import BDFHistory
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState
from contact import ContactSolverBase, XConstraint

'''
reference: [1] Detailed Rigid Body Simulation with Extended Position Based Dynamics
'''
eps = 1e-6

@wp.struct
class RBDDelta: 
    dx: wp.array(dtype = vec3)
    dq: wp.array(dtype = vec4)

# @wp.struct 
# class XConstraint: 
#     e0: int 
#     e1: int 
#     l0: scalar
#     alpha: scalar
#     lam: scalar

@wp.struct 
class Inertia: 
    m: scalar
    J: scalar

@wp.func 
def quat_mult(q1: vec4, q2: vec4) -> vec4:
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w
    return vec4(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )

@wp.kernel
def add_dlam_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), deltas: RBDDelta, delta_counts: wp.array(dtype = int), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    o = scalar(1.0)

    l0 = c.l0
    # i0 = soup.edges[c.e0 * 2]
    # i1 = soup.edges[c.e0 * 2 + 1]
    # i2 = soup.edges[c.e1 * 2]
    # i3 = soup.edges[c.e1 * 2 + 1]
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
    v0 = x0 + scalar(dab[0]) * (x1 - x0)
    v1 = x2 + scalar(dab[1]) * (x3 - x2)
    v10 = v0 - v1
    dist = wp.length(v10)

    if dist < l0: 
        # Eqs. (2) - (5)
        n = v10 / (dist + scalar(eps))
        cc = (dist + scalar(eps))
        
        r1 = v0 - c0 
        r2 = v1 - c1 
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)
        w1 = o / mass[b0].m + wp.dot(a1, (o / mass[b0].J) * a1)
        w2 = o / mass[b1].m + wp.dot(a1, (o / mass[b1].J) * a2)

        common = -cc - c.alpha / (dt * dt)
        denom = w1 + w2 + c.alpha / (dt * dt)
        dlam = common / denom 

        # Eqs. (6) - (9)
        c.lam += dlam 
        pp = dlam * n 
        dx1 = pp / mass[b0].m
        dx2 = -pp / mass[b1].m
        
        r1xp = wp.cross(r1, pp) / mass[b0].J
        dq1 = scalar(0.5) * quat_mult(vec4(r1xp.x, r1xp.y, r1xp.z, scalar(0.0)), p[b0].nxt.q)
        
        r2xp = wp.cross(r2, pp) / mass[b1].J
        dq2 = scalar(-0.5) * quat_mult(vec4(r2xp.x, r2xp.y, r2xp.z, scalar(0.0)), p[b1].nxt.q)

        wp.atomic_add(deltas.dx, b0, dx1)
        wp.atomic_add(deltas.dq, b0, dq1)

        wp.atomic_add(deltas.dx, b1, dx2)
        wp.atomic_add(deltas.dq, b1, dq2)

        wp.atomic_add(delta_counts, b0, 1)
        wp.atomic_add(delta_counts, b1, 1)


class XPBDContact(ContactSolverBase): 
    def __init__(self): 
        ContactSolverBase.__init__(self)
    
    def initialize_multiplier(self):
        pass
