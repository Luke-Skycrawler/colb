import warp as wp 
from contact import ContactSolverBase, XConstraint, thickness, contact_volume, buffer, eps, ContactRet
from .cosserat import Node, Seg
from scalar_types import *
import numpy as np

o = scalar(1.0)
z = scalar(0.0)
stiffness = 4e4
@wp.func 
def fetch_dist_n_r0r1(p: wp.array(dtype = Node), c: XConstraint): 
    l0 = c.l0 
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]


    x0 = p[i0].x
    x1 = p[i1].x
    x2 = p[i2].x
    x3 = p[i3].x

    dab = wp.closest_point_edge_edge(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), eps)
    v0 = wp.lerp(x0, x1, scalar(dab[0]))
    v1 = wp.lerp(x2, x3, scalar(dab[1]))
    v10 = v0 - v1 
    dist = scalar(dab[2])
    n = v10 / dist
    
    return dab, n, v0, v1

    
@wp.kernel 
def get_contact_points(p: wp.array(dtype = Node), xconstraints: wp.array(dtype = XConstraint), contact_ret: ContactRet):
    i = wp.tid() 
    c = xconstraints[i]
    abd, n, r1, r2 = fetch_dist_n_r0r1(p, c)
    dist = abd[2]
    # alpha, beta, dist 

    contact_ret.points[i] = (r1 + r2) * scalar(0.5)
    contact_ret.dists[i] = dist


class NodeContact(ContactSolverBase): 
    def __init__(self): 
        super().__init__()

    def get_contact_points(self):
        wp.launch(get_contact_points, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.contact_ret])
        # filter d > 2 * thickness
        dists = self.contact_ret.dists.numpy()[:self.n_contacts]
        points = self.contact_ret.points.numpy()[:self.n_contacts]
        
        valid = dists < thickness * 2.0
        magnitudes = np.abs(dists[valid] - thickness * 2.0)
        return points[valid], magnitudes

@wp.kernel
def preconditioner_diag_kernel(p: wp.array(dtype = Node), xconstarints: wp.array(dtype = XConstraint), precond: wp.array(dtype = mat33), rhs: wp.array(dtype = vec3), dt: scalar): 
    i = wp.tid()
    c = xconstarints[i]
    
    k = scalar(stiffness)
    
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]

    abd, n, r1, r2 = fetch_dist_n_r0r1(p, c)
    l0 = c.l0 
    dist = abd[2]
    
    v10 = n * dist 
    if dist < l0: 
        j1 = n 
        j2 = -n 
        
        dj1 = k * wp.outer(j1, j1)
        dj2 = k * wp.outer(j2, j2)

        alpha = scalar(abd[0])
        beta = scalar(abd[1])

        wp.atomic_add(precond, i0, dj1 * (o - alpha))
        wp.atomic_add(precond, i1, dj1 * alpha)
        
        wp.atomic_add(precond, i2, dj2 * (o - beta))
        wp.atomic_add(precond, i3, dj2 * beta)

        f1 = k * (l0 - dist) * n * dt 
        f2 = -f1 
        
        wp.atomic_add(rhs, i0, -f1 * (o - alpha))
        wp.atomic_add(rhs, i1, -f1 * alpha)

        wp.atomic_add(rhs, i2, -f2 * (o - beta))
        wp.atomic_add(rhs, i3, -f2 * beta)

@wp.kernel
def compute_u_minus_utilde(pi)
        
        