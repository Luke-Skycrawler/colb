import warp as wp 
import numpy as np 
from rbd_simple import RbdComplex, Inertia 
from contact import ContactSolverBase, XConstraint
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from BDF1 import BDFHistory
from xpbd_contact import forward_states, fetch_dist_n_r0r1_b0b1
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
def preconditioner_diag_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), precond: wp.array(dtype = vec3), rhs: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    k = scalar(stiffness)
    n_bodies = p.shape[0]

    dist, n, r1, r2, b0, b1 = fetch_dist_n_r0r1_b0b1(p, soup, c)
    l0 = c.l0

    v10 = n * dist
    if dist < l0:         
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)

        dpx12 = wp.cw_mul(n, n)
        
        dpq1 = wp.cw_mul(a1, a1)
        dpq2 = wp.cw_mul(a2, a2)

        wp.atomic_add(precond, b0, dpx12 * k)
        wp.atomic_add(precond, b1, dpx12 * k)
        wp.atomic_add(precond, b0 + n_bodies, dpq1 * k)
        wp.atomic_add(precond, b1 + n_bodies, dpq2 * k)
        
        f1 = k * (l0 - dist) * n * dt
        tau1 = wp.cross(r1, f1) 
        tau2 = wp.cross(r2, -f1)
        wp.atomic_add(rhs, b0, -f1)
        wp.atomic_add(rhs, b1, f1)
        wp.atomic_add(rhs, b0 + n_bodies, -tau1)
        wp.atomic_add(rhs, b1 + n_bodies, -tau2)
        
@wp.func 
def compute_u_minus_utilde(pi: BDFHistory, dt: scalar): 
    z = scalar(0.0)
    u_tilde = pi.now.v + dt * vec3(z, scalar(gravity), z)
    du = pi.nxt.v - u_tilde

    domega = pi.nxt.omega - pi.now.omega

    return du, domega

@wp.kernel
def rhs_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), precond: wp.array(dtype = vec3), rhs: wp.array(dtype = vec3), dt: scalar):
    '''
    adds the mass term of Eq. (4)
    rhs should contain the force term before this kernel is called
    '''
    i = wp.tid()
    n_bodies = p.shape[0]
    z = scalar(0.0)
    mi = mass[i].m
    if mi > z: 
        Ji = mass[i].J
        du, domega = compute_u_minus_utilde(p[i], dt)
        rhs[i] += mi * du
        rhs[i + n_bodies] += Ji * domega

        precond[i] = precond[i] * dt * dt + vec3(mi, mi, mi)
        precond[i + n_bodies] = precond[i + n_bodies] * dt * dt + vec3(Ji, Ji, Ji)

    else:
        precond[i] = vec3(scalar(1.0))
        precond[i + n_bodies] = vec3(scalar(1.0))

        rhs[i] = vec3(z)
        rhs[i + n_bodies] = vec3(z)
        
        
    
@wp.kernel
def du_kernel(precond: wp.array(dtype = vec3), rhs: wp.array(dtype = vec3), du: wp.array(dtype = vec3)):
    i = wp.tid()
    du[i] = wp.cw_div(rhs[i], precond[i])

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar):
    i = wp.tid()
    n_bodies = history.shape[0]
    history[i].nxt.v -= alpha * du[i] 
    history[i].nxt.c = history[i].now.c + dt * history[i].nxt.v
    
    history[i].nxt.omega -= alpha * du[i + n_bodies]
    history[i].nxt.q = history[i].now.q + scalar(0.5) * wp.transpose(Gq(history[i].nxt.q)) @ history[i].nxt.omega * dt
    history[i].nxt.q = wp.normalize(history[i].nxt.q)


class PrimalRbd(RbdComplex, ContactSolverBase): 
    def __init__(self, h, config_file): 
        RbdComplex.__init__(self, h, config_file)        
        ContactSolverBase.__init__(self)
        self.alpha = 0.5
        self.precond = wp.zeros((self.n_bodies * 2,), dtype = vec3)
        self.rhs = wp.zeros_like(self.precond)
        self.du = wp.zeros_like(self.precond)

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
        wp.launch(du_kernel, dim = self.n_bodies * 2, inputs = [self.precond, self.rhs, self.du])

        return np.max(np.abs(self.du.numpy()))

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
                max_iter = 4
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
            
