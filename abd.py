import warp as wp 
from rbd_simple import Inertia
from scalar_types import *
from abd_simple import AbdComplex
from contact import ContactSolverBase, contact_volume, XConstraint, fetch_b0b1
from gauss_newton import LineSearchInterface, TripletsCSR
from scipy.sparse import csr_matrix
from warp.sparse import bsr_from_triplets
from warp.optim.linear import cg, bicgstab
from BDF1 import BDFAffine, AffineState
import dxslv
import numpy as np 
from geometry import Soup
from ortho import energy_ortho, grad_ortho, hessian_ortho
@wp.struct
class Triplets:
    rows: wp.array(dtype=int)
    cols: wp.array(dtype=int)
    vals: wp.array(dtype=mat33)

solver_config = "cg"
stiffness = 4e4
gravity = scalar(0.0)

wp.config.max_unroll = 1
wp.config.enable_backward = False
def ptr(arr):
    return arr.__cuda_array_interface__['data'][0]
    

@wp.kernel
def bsr2csr(triplets_BSR: Triplets, triplets_CSR: TripletsCSR):
    i = wp.tid() 

    r = triplets_BSR.rows[i] * 3
    c = triplets_BSR.cols[i] * 3
    v = triplets_BSR.vals[i]

    for ii in range(3):
        for jj in range(3):
            triplets_CSR.rows[i * 9 + ii * 3 + jj] = r + ii
            triplets_CSR.cols[i * 9 + ii * 3 + jj] = c + jj
            triplets_CSR.vals[i * 9 + ii * 3 + jj] = v[ii][jj]


@wp.kernel
def energy_inertia(states: wp.array(dtype = BDFAffine), e: wp.array(dtype = scalar), inertia: wp.array(dtype = Inertia), dt: scalar):
    i = wp.tid()
    state = states[i]

    
    A_tilde = tildeA(state.now.q, state.now.qdot, dt)
    p_tilde = tildep(state.now.c, state.now.v, dt)
    
    dqTMdq = norm_M(inertia[i], state.nxt.q, state.nxt.c, A_tilde, p_tilde)
    de = energy_ortho(state.nxt.q) * dt * dt + scalar(0.5) * dqTMdq
    wp.atomic_add(e, 0, de)

@wp.func
def norm_M(inertia: Inertia, A: mat33, p: vec3, A_tilde: mat33, p_tilde: vec3) -> scalar:
    dq0 = p - p_tilde
    dq1 = A[0] - A_tilde[0]
    dq2 = A[1] - A_tilde[1]
    dq3 = A[2] - A_tilde[2]
    mass = inertia.m
    I0 = inertia.J
    return wp.dot(dq0, dq0) * mass + (wp.dot(dq1, dq1) + wp.dot(dq2, dq2) + wp.dot(dq3, dq3)) * I0


@wp.kernel
def bsr_hessian_inertia(triplets: Triplets, states: wp.array(dtype = BDFAffine), inertia: wp.array(dtype = Inertia), dt: scalar):
    i = wp.tid()
    # os = offset(i, i, bsr)
    os = i * 16
    mass = inertia[i].m
    I0 = inertia[i].J
    for ii in range(4):
        for jj in range(4):
            m = wp.where(ii == jj, wp.where(ii == 0, mass, I0), scalar(0.0))
            I = vec3(m)
            dh =wp.diag(I)
            if ii > 0 and jj > 0:
                dh += hessian_ortho(ii - 1, jj - 1, states[i].nxt.q) * dt * dt

            triplets.rows[os + ii + jj * 4] = i * 4 + ii
            triplets.cols[os + ii + jj * 4] = i * 4 + jj
            triplets.vals[os + ii + jj * 4] = dh


@wp.kernel
def inertia_grad_hess(g: wp.array(dtype = vec3), triplets: Triplets, states: wp.array(dtype = BDFAffine), inertia: wp.array(dtype = Inertia), dt: scalar):
    i = wp.tid()
    os = i * 16
    mass = inertia[i].m
    I0 = inertia[i].J

    if mass > scalar(0.0):
        state = states[i]
        for ii in range(1, 4):
            g[ii + i * 4] = dt * dt * grad_ortho(ii - 1, state.nxt.q)

        A_tilde = tildeA(state.now.q, state.now.qdot, dt)
        p_tilde = tildep(state.now.c, state.now.v, dt)
        q0, q1, q2, q3 = Mdq(state.nxt.q, state.nxt.c, A_tilde, p_tilde, inertia[i])

        g[0 + i * 4] += q0
        g[1 + i * 4] += q1
        g[2 + i * 4] += q2
        g[3 + i * 4] += q3

        for ii in range(4):
            for jj in range(4):
                m = wp.where(ii == jj, wp.where(ii == 0, mass, I0), scalar(0.0))
                I = vec3(m)
                dh =wp.diag(I)
                if ii > 0 and jj > 0:
                    dh += hessian_ortho(ii - 1, jj - 1, states[i].nxt.q) * dt * dt

                triplets.rows[os + ii + jj * 4] = i * 4 + ii
                triplets.cols[os + ii + jj * 4] = i * 4 + jj
                triplets.vals[os + ii + jj * 4] = dh
    # else: 
    #     for ii in range(4):
    #         for jj in range(4):
    #             triplets.rows[os + ii + jj * 4] = i * 4 + ii
    #             triplets.cols[os + ii + jj * 4] = i * 4 + jj
    #             if ii == jj:
    #                 triplets.vals[os + ii + jj * 4] = wp.identity(3, dtype = scalar) * scalar(100.0)
    #             else:
    #                 triplets.vals[os + ii + jj * 4] = mat33(scalar(0.0))
    #         g[ii + i * 4] = vec3(scalar(0.0))
@wp.func
def Mdq(A: mat33, p: vec3, A_tilde: mat33, p_tilde: vec3, inertia: Inertia):
    q0 = p - p_tilde
    q1 = A[0] - A_tilde[0]
    q2 = A[1] - A_tilde[1]
    q3 = A[2] - A_tilde[2]
    mass = inertia.m
    I0 = inertia.J
    return q0 * mass, q1 * I0, q2 * I0, q3 * I0

@wp.func
def tildeA(A0: mat33, Adot: mat33, dt: scalar) -> mat33:
    return A0 + dt * Adot

@wp.func
def tildep(p0: vec3, pdot: vec3, dt: scalar) -> vec3:
    return p0 + dt * pdot + dt * dt * vec3(scalar(0.0), gravity, scalar(0.0))


# @wp.kernel
# def compute_contact_gh_kernel(p: wp.array(dtype = BDFAffine), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), triplets: Triplets, rhs: wp.array(dtype = vec12), dt: scalar):
#     i = wp.tid()
#     c = xconstraints[i]
#     k = scalar(stiffness)
#     n_bodies = p.shape[0]

#     b0, b1 = fetch_b0b1(c, soup)
#     dist, n, r1, r2 = fetch_dist_n_r0r1(p[b0], p[b1], soup, c)
#     l0 = c.l0

#     v10 = n * dist
#     z = scalar(0.0)

#     if dist < l0:         
#         a1 = wp.cross(r1, n)
#         a2 = wp.cross(r2, n)


#         j1 = make_vec6(n, a1)
#         j2 = make_vec6(-n, -a2)

#         dt2 = dt * dt

#         dj11 = k * wp.outer(j1, j1) * dt2
#         dj22 = k * wp.outer(j2, j2) * dt2
#         dj12 = k * wp.outer(j1, j2) * dt2

#         # if mass[b0].m < z:
#         #     dj11 *= z
#         #     dj12 *= z
#         # if mass[b1].m < z:
#         #     dj22 *= z
#         #     dj12 *= z

#         # wp.atomic_add(precond, b0 * 2, dpx12 * k)
#         # wp.atomic_add(precond, b1 * 2, dpx12 * k)
#         # wp.atomic_add(precond, b0 * 2 + 1, dpq1 * k)
#         # wp.atomic_add(precond, b1 * 2 + 1, dpq2 * k)

#         trip_offset = i * 4 + n_bodies
#         # first n_bodies entries are for inertia

#         if mass[b0].m > z:
#             triplets.rows[trip_offset + 0] = b0 
#             triplets.cols[trip_offset + 0] = b0
#             triplets.vals[trip_offset + 0] = dj11
        
#         if mass[b1].m > z:
#             triplets.rows[trip_offset + 1] = b1
#             triplets.cols[trip_offset + 1] = b1
#             triplets.vals[trip_offset + 1] = dj22

#         if mass[b0].m > z and mass[b1].m > z:
#             triplets.rows[trip_offset + 2] = b0
#             triplets.cols[trip_offset + 2] = b1
#             triplets.vals[trip_offset + 2] = dj12

#             triplets.rows[trip_offset + 3] = b1
#             triplets.cols[trip_offset + 3] = b0
#             triplets.vals[trip_offset + 3] = wp.transpose(dj12)
        
#         f1 = k * (l0 - dist) * n * dt
#         tau1 = wp.cross(r1, f1) 
#         tau2 = wp.cross(r2, -f1)

#         db1 = make_vec6(-f1, -tau1)
#         db2 = make_vec6(f1, -tau2)
        
#         if mass[b0].m > z:
#             wp.atomic_add(rhs, b0, db1)
#         if mass[b1].m > z:
#             wp.atomic_add(rhs, b1, db2)


@wp.kernel
def forward_states(history: wp.array(dtype = BDFAffine), dt: scalar):
    i = wp.tid()
    history[i].nxt.v = (history[i].nxt.c - history[i].now.c) / dt
    q_dot = (history[i].nxt.q - history[i].now.q) / dt
    
    history[i].nxt.qdot = q_dot
    history[i].now = history[i].nxt


class NewtonAbd(LineSearchInterface, AbdComplex, ContactSolverBase):
    def __init__(self, h, config_file):
        AbdComplex.__init__(self, h, config_file)        
        ContactSolverBase.__init__(self)
        LineSearchInterface.__init__(self)

        triplets = Triplets()
        nnz = (self.n_bodies + contact_volume * 4) * 16
        triplets.rows = wp.zeros((nnz,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((nnz,), dtype = mat33)

        self.rhs = wp.zeros((self.n_bodies * 4,), dtype = vec3)
        self.du = wp.zeros_like(self.rhs)
        self.triplets = triplets

    def compute_contact_gh(self):
        return 
        self.rhs.zero_()
        self.triplets.rows.zero_()
        self.triplets.cols.zero_()
        self.triplets.vals.zero_()
        wp.launch(compute_contact_gh_kernel, (self.n_contacts,), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.triplets, self.rhs, self.dt])


    def to_csr(self, triplets: Triplets): 
        a = TripletsCSR() 
        nnz = (self.n_bodies) * 9
        a.rows = wp.zeros((nnz,), dtype = int)
        a.cols = wp.zeros_like(a.rows)
        a.vals = wp.zeros((nnz,), dtype = scalar)

        wp.launch(bsr2csr, dim = nnz // 9, inputs = [triplets, a])

        # prune numerical zeros off is necessary, because the direct solver will directly copy the first nnz values
        return bsr_from_triplets(self.n_bodies * 4, self.n_bodies * 4, a.rows, a.cols, a.vals, prune_numerical_zeros=False)

    def to_scipy_csr(self, mat):
        ii = mat.offsets.numpy()
        jj = mat.columns.numpy()
        values = mat.values.numpy()
        shape = (mat.nrow, mat.ncol) 
        print(f"shape = {shape}, values = {values.shape}, ii = {ii.shape}, jj = {jj.shape}")
        csr = csr_matrix((values, jj, ii), shape = shape)
        return csr

    def compute_du(self, iter):
        self.du.zero_()
        if solver_config == "cg": 
            A = bsr_from_triplets(self.n_bodies * 4, self.n_bodies * 4, self.triplets.rows, self.triplets.cols, self.triplets.vals)
            # cg(A, self.rhs, self.du, tol = 1e-5)
            bicgstab(A, self.rhs, self.du, tol = 1e-5)
        else: 
            A_csr = self.to_csr(self.triplets)
            if iter == 0: 
                with wp.ScopedTimer("build solver"):
                    A_scipy = self.to_scipy_csr(A_csr)
                    self.solver = dxslv.CUSolver(A_scipy)
                with wp.ScopedTimer("analyze + factorize"):
                    self.solver.analyze_pattern()
                    self.solver.factorize()
            else:
                values = A_csr.values
                self.solver.refactor_cuda(ptr(values))
            self.solver.solve_cuda(ptr(self.rhs), ptr(self.du))
        du_norm = np.max(np.abs(self.du.numpy()))
        return du_norm


    def forward_states(self): 
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1

    def step(self):
        for ss in range(1):
            with wp.ScopedTimer("newton step"):
                newton = True
                iter = 0
                max_iter = 10
                # self.detect_collision()
                while newton: 
                    self.triplets.rows.zero_()
                    self.triplets.cols.zero_()
                    self.triplets.vals.zero_()
                    self.rhs.zero_()
                    wp.launch(inertia_grad_hess, self.n_bodies, inputs = [self.rhs, self.triplets, self.history, self.inertia, self.dt])
                    # wp.launch(bsr_hessian_inertia, self.n_bodies, inputs = [self.triplets, self.history, self.inertia, self.dt])
                    
                    dq_norm = self.compute_du(iter) 
                    alpha = self.line_search()
                    # alpha = self.line_search_batch()
                    iter += 1
                    print(f"    iter: {iter}, dq norm: {dq_norm}, alpha = {alpha}")
                    newton = not (dq_norm < 1e-5 or iter >= max_iter)
                    
                self.forward_states()

    def line_search_iterative(self):
        alpha = 1.0
        backup = wp.clone(self.history)
        # E0 = self.compute_g()

        while True:
            wp.copy(self.history, backup)    
            self.add_du(alpha)
            break

        return alpha

    # def compute_dq(self, iter): 
    #     hess = bsr_from_triplets(self.n_bodies * 4, self.n_bodies * 4, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        
    #     bicgstab(hess, self.rhs, self.du, tol = 1e-5)
    #     return np.max(np.abs(self.du.numpy()))
        

    def add_du(self, alpha):
        # wp.launch(add_vec12_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])

    def get_contact_points(self):
        # wp.launch(get_contact_points, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.contact_ret])
        # # filter d > 2 * thickness
        # dists = self.contact_ret.dists.numpy()[:self.n_contacts]
        # points = self.contact_ret.points.numpy()[:self.n_contacts]
        
        # valid = dists < thickness * 2.0
        # magnitudes = np.abs(dists[valid] - thickness * 2.0)
        return np.zeros((0, 3)), np.zeros((0,))
        return points[valid], magnitudes

@wp.func
def apply_du(du: vec3, dw: mat33, _state: BDFAffine, alpha: scalar, dt: scalar): 
    state = BDFAffine()
    state = _state
    state.nxt.c -= alpha * du 
    
    state.nxt.q -= alpha * dw
    
    return state



@wp.kernel 
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFAffine), alpha: scalar, dt: scalar):
    i = wp.tid()
    dui = du[i]
    dq = wp.matrix_from_rows(du[i * 4 + 1], du[i * 4 + 2], du[i * 4 + 3])
    history[i] = apply_du(dui, dq, history[i], alpha, dt)
