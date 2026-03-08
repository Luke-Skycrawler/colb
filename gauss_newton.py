import warp as wp 
import numpy as np 
from primal import PrimalRbd, XConstraint, stiffness, eps, gravity, add_du_kernel, compute_u_minus_utilde, apply_du
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from BDF1 import BDFHistory
from xpbd_contact import RbdComplex, ContactSolverBase, forward_states, fetch_dist_n_r0r1, fetch_b0b1
from contact import contact_volume
from rbd_simple import Inertia
from geometry import Soup

@wp.struct 
class Triplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    vals: wp.array(dtype = wp.mat33)


@wp.kernel
def uTMu_kernel(history: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), dt: scalar, E: wp.array(dtype = scalar)):
    i = wp.tid()
    du, domega = compute_u_minus_utilde(history[i], dt)
    mi = mass[i].m
    Ji = mass[i].J

    de = scalar(0.5) * mi * wp.dot(du, du) + scalar(0.5) * Ji * wp.dot(domega, domega)
    wp.atomic_add(E, 0, de)

@wp.kernel
def uTMu_batch(history: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), dt: scalar, E: wp.array(dtype = scalar), du_arr: wp.array(dtype = vec3)):
    i, pow2i = wp.tid()
    n_bodies = history.shape[0]

    pow2 = scalar(pow2i)
    alpha = wp.pow(scalar(0.5), pow2)

    history_i = apply_du(du_arr[i], du_arr[i + n_bodies], history[i], alpha, dt)

    du, domega = compute_u_minus_utilde(history_i, dt)
    mi = mass[i].m
    Ji = mass[i].J

    de = scalar(0.5) * mi * wp.dot(du, du) + scalar(0.5) * Ji * wp.dot(domega, domega)
    wp.atomic_add(E, pow2i, de)

@wp.kernel
def contact_energy_kernel(p: wp.array(dtype = BDFHistory), soup: Soup, xconstraints: wp.array(dtype = XConstraint), E: wp.array(dtype = scalar)):
    i = wp.tid()
    c = xconstraints[i]
    k = scalar(stiffness)
    l0 = c.l0

    b0, b1 = fetch_b0b1(c, soup)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p[b0], p[b1], soup, c)

    if dist < l0: 
        de = scalar(0.5) * k * (l0 - dist) * (l0 - dist)
        wp.atomic_add(E, 0, de)

@wp.kernel
def contact_energy_batch(p: wp.array(dtype = BDFHistory), soup: Soup, xconstraints: wp.array(dtype = XConstraint), E: wp.array(dtype = scalar), du: wp.array(dtype = vec3), dt: scalar):
    n_bodies = p.shape[0]
    i, pow2i = wp.tid()

    pow2 = scalar(pow2i)
    alpha = wp.pow(scalar(0.5), pow2)
    
    c = xconstraints[i]
    k = scalar(stiffness)
    l0 = c.l0
    b0, b1 = fetch_b0b1(c, soup)
    
    p0 = apply_du(du[b0], du[b0 + n_bodies], p[b0], alpha, dt)
    p1 = apply_du(du[b1], du[b1 + n_bodies], p[b1], alpha, dt)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p0, p1, soup, c)
    
    dl = wp.max(scalar(0.0), l0 - dist)
    de = scalar(0.5) * k * dl * dl
    wp.atomic_add(E, pow2i, de)
    
class LineSearchInterface: 
    def __init__(self, h, config_file):
        super().__init__(h, config_file)
        self.n_batches = 8
        self.g = wp.zeros((1, ), dtype = scalar)
        self.g_batch = wp.zeros((self.n_batches, ), dtype = scalar)

        self.pows = np.arange(self.n_batches, dtype = float)
        self.alphas = np.power(0.5, self.pows)
        

    def line_search_iterative(self):
        alpha = 1.0
        backup = wp.clone(self.history)
        E0 = self.compute_g()

        while True:
            wp.copy(self.history, backup)    
            self.add_du(alpha)
            E1 = self.compute_g()
            if E1 < E0:
                break 
            elif alpha * 0.5 < 0.1: 
                break 
            else: 
                alpha *= 0.5 

        return alpha

    def line_search_batch(self):
        self.g_batch.zero_()
        wp.launch(uTMu_batch, (self.n_bodies, self.n_batches), inputs = [self.history, self.inertia, self.dt, self.g_batch, self.du])

        wp.launch(contact_energy_batch, (self.n_contacts, self.n_batches), inputs = [self.history, self.soup, self.contacts_new.list, self.g_batch, self.du, self.dt])

        g = self.g_batch.numpy()
        idx = np.argmin(g)
        alpha = self.alphas[idx]
        self.add_du(alpha)
        return alpha

    def line_search(self):
        # self.line_search_iterative()
        self.line_search_batch()

    def compute_g(self):
        '''
        Eq. (2) in [2]
        g(u) = 1/2 (u - u_tilde)^T M (u - u_tilde) + sum_j 1/2 k Cj(q)^2
        '''

        self.g.zero_()
        wp.launch(uTMu_kernel, (self.n_bodies,), inputs = [self.history, self.inertia, self.dt, self.g])

        wp.launch(contact_energy_kernel, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.g])

        return self.g.numpy()[0]

class LineSearchGDRbd(LineSearchInterface, PrimalRbd):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)

class GaussNewtonRbd(LineSearchInterface, RbdComplex, ContactSolverBase):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)
        triplets = Triplets()
        triplets.rows = wp.zeros((self.n_bodies * 4 + contact_volume * 8,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((self.n_bodies * 4 + contact_volume * 8,), dtype = wp.mat33)

        self.rhs = wp.zeros((self.n_bodies * 2,), dtype = vec3)
        self.du = wp.zeros_like(self.rhs)

    def compute_contact_gh(self):
        # contact hessian and gradient
        pass

    def compute_inertia_gh(self):
        # inertia hessian and gradient
        pass

    def compute_du(self):
        pass

    def add_du(self, alpha):
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])

    def forward_states(self): 
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1


    def step(self):
        for ss in range(10):
            newton = True
            iter = 0
            max_iter = 4
            self.detect_collision()
            while newton: 
                self.compute_contact_gh()
                self.compute_inertia_gh()
                
                du_norm = self.compute_du() 
                alpha = self.line_search()
                # alpha = self.line_search_batch()
                iter += 1
                # print(f"    iter: {iter}, du norm: {du_norm}")
                newton = not (du_norm < 1e-5 or iter >= max_iter)
        
            self.forward_states()

    
        