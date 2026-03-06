import warp as wp 
import numpy as np 
from rbd_simple import RbdComplex, Inertia 
from contact import ContactSolverBase, XConstraint
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from BDF1 import BDFHistory
from xpbd_contact import forward_states
from geometry import Soup

'''
differentiable primal solver from [1]

reference
[1]: Primal/Dual Descent Methods for Dynamics.
'''

eps = 1e-6
stiffness = 1e5
gravity = -9.8

@wp.kernel
def preconditioner_diag_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), precond: wp.array(dtype = vec3), rhs: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    o = scalar(1.0)
    z = scalar(0.0)
    k = scalar(stiffness)
    n_bodies = p.shape[0]
    l0 = c.l0
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
    v0 = wp.lerp(x0, x1, scalar(dab[0]))
    v1 = wp.lerp(x2, x3, scalar(dab[1]))

    v10 = v0 - v1
    dist = scalar(dab[2])
    m1 = mass[b0].m

    if dist < l0: 
        n = v10 / (dist + scalar(eps))
        
        r1 = v0 - c0 
        r2 = v1 - c1
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
        u_tilde = p[i].now.v + dt * vec3(z, scalar(gravity), z)
        du = p[i].nxt.v - u_tilde
        rhs[i] += mi * du

        rhs[i + n_bodies] += Ji * (p[i].nxt.omega - p[i].now.omega)

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
    du[i] = wp.cw_div(rhs[i], precond[i] + vec3(scalar(eps)))

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar):
    i = wp.tid()
    n_bodies = history.shape[0]
    history[i].nxt.v -= alpha * du[i] 
    history[i].nxt.c = history[i].now.c - dt * history[i].nxt.v
    
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

    def step(self):
        newton = True
        iter = 0
        max_iter = 8
        self.detect_collision()
        while newton: 
            self.compute_preconditioner()
            self.compute_rhs()
            
            du_norm = self.compute_du() 
            self.add_du(self.alpha)
            iter += 1
            print(f"iter: {iter}, du norm: {du_norm}")
            newton = not (du_norm < 1e-5 or iter >= max_iter)
    
        self.forward_states()
        
