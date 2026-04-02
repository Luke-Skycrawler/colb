import warp as wp 
import numpy as np 
from primal import PrimalRbd, XConstraint, stiffness, eps, gravity, compute_u_minus_utilde, apply_du
from scalar_types import *
from BDF1 import BDFHistory
from xpbd_contact import RbdComplex, ContactSolverBase, forward_states, fetch_dist_n_r0r1, fetch_b0b1
from contact import contact_volume
from rbd_simple import Inertia
from geometry import Soup
from warp.optim.linear import cg
from warp.sparse import bsr_from_triplets, bsr_set_from_triplets
@wp.struct 
class Triplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    vals: wp.array(dtype = mat6)    

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

    history_i = apply_du(du_arr[i * 2], du_arr[i * 2 + 1], history[i], alpha, dt)

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
    
    p0 = apply_du(du[b0 * 2], du[b0 * 2 + 1], p[b0], alpha, dt)
    p1 = apply_du(du[b1 * 2], du[b1 * 2 + 1], p[b1], alpha, dt)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p0, p1, soup, c)
    
    dl = wp.max(scalar(0.0), l0 - dist)
    de = scalar(0.5) * k * dl * dl
    wp.atomic_add(E, pow2i, de)
    
class LineSearchInterface: 
    def __init__(self):
        # super().__init__(h, config_file)
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
            # elif alpha * 0.5 < 0.1: 
            elif alpha * 0.5 < 1e-2: 
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
        return self.line_search_iterative()
        # return self.line_search_batch()

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
        PrimalRbd.__init__(h, config_file)
        LineSearchInterface.__init__(self)



@wp.kernel
def compute_contact_gh_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), triplets: Triplets, rhs: wp.array(dtype = vec6), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    k = scalar(stiffness)
    n_bodies = p.shape[0]

    b0, b1 = fetch_b0b1(c, soup)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p[b0], p[b1], soup, c)
    l0 = c.l0

    v10 = n * dist
    z = scalar(0.0)

    if dist < l0:         
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)


        j1 = make_vec6(n, a1)
        j2 = make_vec6(-n, -a2)

        dt2 = dt * dt

        dj11 = k * wp.outer(j1, j1) * dt2
        dj22 = k * wp.outer(j2, j2) * dt2
        dj12 = k * wp.outer(j1, j2) * dt2

        # if mass[b0].m < z:
        #     dj11 *= z
        #     dj12 *= z
        # if mass[b1].m < z:
        #     dj22 *= z
        #     dj12 *= z

        # wp.atomic_add(precond, b0 * 2, dpx12 * k)
        # wp.atomic_add(precond, b1 * 2, dpx12 * k)
        # wp.atomic_add(precond, b0 * 2 + 1, dpq1 * k)
        # wp.atomic_add(precond, b1 * 2 + 1, dpq2 * k)

        trip_offset = i * 4 + n_bodies
        # first n_bodies entries are for inertia

        if mass[b0].m > z:
            triplets.rows[trip_offset + 0] = b0 
            triplets.cols[trip_offset + 0] = b0
            triplets.vals[trip_offset + 0] = dj11
        
        if mass[b1].m > z:
            triplets.rows[trip_offset + 1] = b1
            triplets.cols[trip_offset + 1] = b1
            triplets.vals[trip_offset + 1] = dj22

        if mass[b0].m > z and mass[b1].m > z:
            triplets.rows[trip_offset + 2] = b0
            triplets.cols[trip_offset + 2] = b1
            triplets.vals[trip_offset + 2] = dj12

            triplets.rows[trip_offset + 3] = b1
            triplets.cols[trip_offset + 3] = b0
            triplets.vals[trip_offset + 3] = wp.transpose(dj12)
        
        f1 = k * (l0 - dist) * n * dt
        tau1 = wp.cross(r1, f1) 
        tau2 = wp.cross(r2, -f1)

        db1 = make_vec6(-f1, -tau1)
        db2 = make_vec6(f1, -tau2)
        
        if mass[b0].m > z:
            wp.atomic_add(rhs, b0, db1)
        if mass[b1].m > z:
            wp.atomic_add(rhs, b1, db2)
        # wp.atomic_add(rhs, b0 * 2, -f1)
        # wp.atomic_add(rhs, b1 * 2, f1)
        # wp.atomic_add(rhs, b0 * 2 + 1, -tau1)
        # wp.atomic_add(rhs, b1 * 2 + 1, -tau2)

@wp.func 
def make_vec6(v: vec3, w: vec3) -> vec6:
    return vec6(v[0], v[1], v[2], w[0], w[1], w[2]) 

@wp.kernel
def compute_inertia_gh_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), triplets: Triplets, rhs: wp.array(dtype = vec6), dt: scalar):
    i = wp.tid()
    o = scalar(1.0)
    z = scalar(0.0)
    mi = mass[i].m

    triplets.rows[i] = i
    triplets.cols[i] = i

    if mi > z: 
        Ji = mass[i].J
        du, domega = compute_u_minus_utilde(p[i], dt)
        rhs[i] += make_vec6(mi * du, Ji * domega)
        triplets.vals[i] = wp.diag(vec6(mi, mi, mi, Ji, Ji, Ji))
    else:
        triplets.vals[i] = wp.diag(vec6(o, o, o, o, o, o))
        rhs[i] = vec6(z)


@wp.kernel
def add_vec6_du_kernel(du: wp.array(dtype = vec6), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar):
    i = wp.tid()
    # n_bodies = history.shape[0]
    dui = du[i]
    u = vec3(dui[0], dui[1], dui[2])
    omega = vec3(dui[3], dui[4], dui[5])
    history[i] = apply_du(u, omega, history[i], alpha, dt)


class GaussNewtonRbd(LineSearchInterface, RbdComplex, ContactSolverBase):
    def __init__(self, h, config_file):
        RbdComplex.__init__(self, h, config_file)        
        ContactSolverBase.__init__(self)
        LineSearchInterface.__init__(self)

        triplets = Triplets()
        nnz = self.n_bodies + contact_volume * 4
        triplets.rows = wp.zeros((nnz,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((nnz,), dtype = mat6)

        self.rhs = wp.zeros((self.n_bodies,), dtype = vec6)
        self.du = wp.zeros_like(self.rhs)
        self.triplets = triplets

    def compute_contact_gh(self):
        # contact hessian and gradient
        self.rhs.zero_()
        self.triplets.rows.zero_()
        self.triplets.cols.zero_()  
        self.triplets.vals.zero_()
        wp.launch(compute_contact_gh_kernel, (self.n_contacts,), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.triplets, self.rhs, self.dt])
        

    def compute_inertia_gh(self):
        # inertia hessian and gradient
        wp.launch(compute_inertia_gh_kernel, (self.n_bodies,), inputs = [self.history, self.inertia, self.triplets, self.rhs, self.dt])
        

    def compute_du(self):
        self.du.zero_()
        A = bsr_from_triplets(self.n_bodies, self.n_bodies, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        cg(A, self.rhs, self.du, tol = 1e-5)
        du_norm = np.max(np.abs(self.du.numpy()))
        return du_norm

    def add_du(self, alpha):
        wp.launch(add_vec6_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])

    def forward_states(self): 
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1


    def step(self):
        for ss in range(1):
            with wp.ScopedTimer("gauss newton step"):
                newton = True
                iter = 0
                max_iter = 8
                self.detect_collision()
                while newton: 
                    self.compute_contact_gh()
                    self.compute_inertia_gh()
                    
                    du_norm = self.compute_du() 
                    alpha = self.line_search()
                    # alpha = self.line_search_batch()
                    iter += 1
                    print(f"    iter: {iter}, du norm: {du_norm}, alpha = {alpha}")
                    newton = not (du_norm < 1e-5 or iter >= max_iter)
            
                self.forward_states()

    
        