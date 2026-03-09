import warp as wp 
import numpy as np 
from rbd_simple import RbdComplex, Inertia 
from contact import ContactSolverBase, XConstraint
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult, vec6, mat6
from BDF1 import BDFHistory
from xpbd_contact import forward_states, fetch_dist_n_r0r1, fetch_b0b1
from geometry import Soup

'''
differentiable primal solver from [1]

reference
[1]: Primal/Dual Descent Methods for Dynamics.
'''

eps = 1e-6
stiffness = 1e4
gravity = -9.8

@wp.kernel
def preconditioner_diag_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), precond: wp.array(dtype = mat6), rhs: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    k = scalar(stiffness)
    # n_bodies = p.shape[0]

    b0, b1 = fetch_b0b1(c, soup)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p[b0], p[b1], soup, c)
    l0 = c.l0

    v10 = n * dist
    if dist < l0:         
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)

        # dpx12 = wp.cw_mul(n, n)
        
        # dpq1 = wp.cw_mul(a1, a1)
        # dpq2 = wp.cw_mul(a2, a2)
        # dpx12 = wp.outer(n, n)
        # dpq1 = wp.outer(a1, a1)
        # dpq2 = wp.outer(a2, a2)

        j1 = vec6(
            n[0], n[1], n[2], a1[0], a1[1], a1[2]
        )
        j2 = vec6(
            -n[0], -n[1], -n[2], -a2[0], -a2[1], -a2[2]
        )
        
        dj1 = k * wp.outer(j1, j1)
        dj2 = k * wp.outer(j2, j2)

        # wp.atomic_add(precond, b0 * 2, dpx12 * k)
        # wp.atomic_add(precond, b1 * 2, dpx12 * k)
        # wp.atomic_add(precond, b0 * 2 + 1, dpq1 * k)
        # wp.atomic_add(precond, b1 * 2 + 1, dpq2 * k)

        wp.atomic_add(precond, b0, dj1)
        wp.atomic_add(precond, b1, dj2)
        
        f1 = k * (l0 - dist) * n * dt
        tau1 = wp.cross(r1, f1) 
        tau2 = wp.cross(r2, -f1)
        wp.atomic_add(rhs, b0 * 2, -f1)
        wp.atomic_add(rhs, b1 * 2, f1)
        wp.atomic_add(rhs, b0 * 2 + 1, -tau1)
        wp.atomic_add(rhs, b1 * 2 + 1, -tau2)
        
@wp.func 
def compute_u_minus_utilde(pi: BDFHistory, dt: scalar): 
    z = scalar(0.0)
    u_tilde = pi.now.v + dt * vec3(z, scalar(gravity), z)
    du = pi.nxt.v - u_tilde

    domega = pi.nxt.omega - pi.now.omega

    return du, domega

@wp.kernel
def rhs_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), precond: wp.array(dtype = mat6), rhs: wp.array(dtype = vec3), dt: scalar):
    '''
    adds the mass term of Eq. (4)
    rhs should contain the force term before this kernel is called
    '''
    i = wp.tid()
    # n_bodies = p.shape[0]
    o = scalar(1.0)
    z = scalar(0.0)
    mi = mass[i].m
    if mi > z: 
        Ji = mass[i].J
        du, domega = compute_u_minus_utilde(p[i], dt)
        rhs[i * 2] += mi * du
        rhs[i * 2 + 1] += Ji * domega

        # precond[i] = precond[i] * dt * dt + vec3(mi, mi, mi)
        # precond[i + n_bodies] = precond[i + n_bodies] * dt * dt + vec3(Ji, Ji, Ji)
        # precond[i * 2] = precond[i * 2] * dt * dt + wp.diag(vec3(mi))
        # precond[i * 2 + 1] = precond[i * 2 + 1] * dt * dt + wp.diag(vec3(Ji))
        precond[i] = precond[i] * dt * dt + wp.diag(vec6(mi, mi, mi, Ji, Ji, Ji))
    else:
        # precond[i] = vec3(scalar(1.0))
        # precond[i + n_bodies] = vec3(scalar(1.0))
        # precond[i * 2] = wp.diag(vec3(scalar(1.0)))
        # precond[i * 2 + 1] = wp.diag(vec3(scalar(1.0)))
        precond[i] = wp.diag(vec6(o, o, o, o, o, o))

        rhs[i * 2] = vec3(z)
        rhs[i * 2 + 1] = vec3(z)
        
        
@wp.func 
def ldlt(A: wp.spatial_matrixd, b: wp.spatial_vectord):
    '''
    computes Ax = b using LDLT factorization 
    args:   A: 6x6 matrix
            b: 6D vector
    '''

    z = scalar(0.0)
    L = wp.matrix(z, shape = (6, 6), dtype = scalar)

    # --- Cholesky factorization ---
    for i in range(6):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                L[i, j] = wp.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    # --- Forward substitution Ly = b ---
    y = wp.vector(z, length = 6, dtype = scalar)

    for i in range(6):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    # --- Back substitution Lᵀx = y ---
    x = wp.vector(z, length = 6, dtype = scalar)

    for i in range(5, -1, -1):
        s = y[i]
        for j in range(i + 1, 6):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]

    return x

@wp.kernel
def du_kernel(precond: wp.array(dtype = mat6), rhs: wp.array(dtype = vec3), du: wp.array(dtype = vec3)):
    i = wp.tid()
    # du[i] = wp.cw_div(rhs[i], precond[i])
    # du[i] = wp.inverse(precond[i]) @ rhs[i]
    y = vec6(rhs[i * 2][0], rhs[i * 2][1], rhs[i * 2][2], rhs[i * 2 + 1][0], rhs[i * 2 + 1][1], rhs[i * 2 + 1][2])
    x = ldlt(precond[i], y)
    du[i * 2] = vec3(x[0], x[1], x[2])
    du[i * 2 + 1] = vec3(x[3], x[4], x[5])
    

@wp.func
def apply_du(du: vec3, dw: vec3, _state: BDFHistory, alpha: scalar, dt: scalar): 
    state = BDFHistory()
    state = _state
    state.nxt.v -= alpha * du 
    state.nxt.c = state.now.c + dt * state.nxt.v
    
    state.nxt.omega -= alpha * dw
    state.nxt.q = state.now.q + scalar(0.5) * wp.transpose(Gq(state.nxt.q)) @ state.nxt.omega * dt
    state.nxt.q = wp.normalize(state.nxt.q)

    return state

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar):
    i = wp.tid()
    # n_bodies = history.shape[0]
    
    history[i] = apply_du(du[i * 2], du[i * 2 + 1], history[i], alpha, dt)


class PrimalRbd(RbdComplex, ContactSolverBase): 
    def __init__(self, h, config_file): 
        RbdComplex.__init__(self, h, config_file)        
        ContactSolverBase.__init__(self)
        self.alpha = 0.5
        # self.precond = wp.zeros((self.n_bodies * 2,), dtype = mat33)
        self.precond = wp.zeros((self.n_bodies,), dtype = mat6)
        self.rhs = wp.zeros((self.n_bodies * 2,), dtype = vec3)
        self.du = wp.zeros_like(self.rhs)

    def compute_J(self):
        pass

    def compute_preconditioner(self):
        '''
        Eq. (12)
        P^GN = (M + dt ^ 2 k J^T J)^-1 
        J = (r1 cross n, n)
        '''
        self.precond.zero_()
        self.rhs.zero_()
        wp.launch(preconditioner_diag_kernel, dim = (self.n_contacts, ), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.precond, self.rhs, self.dt])
        
    def compute_rhs(self):
        wp.launch(rhs_kernel, dim = (self.n_bodies, ), inputs = [self.history, self.inertia, self.precond, self.rhs, self.dt])

    def compute_du(self): 
        wp.launch(du_kernel, dim = self.n_bodies, inputs = [self.precond, self.rhs, self.du])

        # return np.max(np.abs(self.du.numpy()))
        return 1.0

    def add_du(self, alpha): 
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])
    
    def forward_states(self): 
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1

    def line_search(self):
        self.add_du(self.alpha)
        return self.alpha

    def step(self):
        for ss in range(10):
            with wp.ScopedTimer("step"):
                newton = True
                iter = 0
                max_iter = 8
                self.detect_collision()
                while newton: 
                    self.compute_preconditioner()
                    self.compute_rhs()
                    
                    du_norm = self.compute_du() 
                    alpha = self.line_search()
                    iter += 1
                    # print(f"    iter: {iter}, du norm: {du_norm}, alpha = {alpha}")
                    newton = not (du_norm < 1e-5 or iter >= max_iter)
        
            self.forward_states()
            
