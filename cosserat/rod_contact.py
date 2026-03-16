import warp as wp 
from contact import ContactSolverBase, XConstraint, thickness, contact_volume, buffer, eps, ContactRet
from .cosserat import Node, Seg, StableCosserat, kss, kbt, compute_Css, g
from scalar_types import *
import numpy as np
from geometry import Soup

'''
Contact handling for Cosserat rod, using primal preconditioned gradient descent.
'''
o = scalar(1.0)
z = scalar(0.0)

stiffness = 1e9
gravity = -9.8
debug = False

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
    dist = scalar(abd[2])
    # alpha, beta, dist 

    contact_ret.points[i] = (r1 + r2) * scalar(0.5)
    contact_ret.dists[i] = dist

@wp.kernel 
def copy_x(x: wp.array(dtype = vec3), nodes: wp.array(dtype = Node)):
    i = wp.tid()
    x[i] = nodes[i].x

class RodContact(ContactSolverBase, StableCosserat): 
    def __init__(self, n_nodes, dt): 
        StableCosserat.__init__(self, n_nodes, dt)
        self.define_contact_viewer_interface()
        ContactSolverBase.__init__(self)

    def define_contact_viewer_interface(self): 
        '''
        Bridging cosserat rod with contact solver by defining `self.soup: Soup`
        '''
        geom = Soup()
        geom.xcs = wp.zeros((self.n_nodes,), dtype = vec3)
        geom.triangles = wp.zeros((1,), dtype = int)

        nxt = self.segs.numpy()["nxt"]
        select = nxt >= 0
        e_start = np.arange(self.n_nodes)[select]
        e_end = e_start + 1
        self.E = np.hstack((e_start.reshape(-1, 1), e_end.reshape(-1, 1)))
        
        geom.edges = wp.array(self.E.reshape(-1), dtype = int)
        geom.body = wp.zeros((1, ), dtype = int)
        geom.x_transformed = wp.zeros((self.n_nodes,), dtype = vec3)

        wp.launch(copy_x, (self.n_nodes, ), inputs = [geom.xcs, self.nodes])
        wp.copy(geom.x_transformed, geom.xcs)
        
        self.soup = geom
    
    def compute_V(self, ret = True):
        wp.launch(copy_x, (self.n_nodes, ), inputs = [self.soup.x_transformed, self.nodes])
        if ret:
            return self.soup.x_transformed.numpy()
        return None

    def get_contact_points(self):
        wp.launch(get_contact_points, (self.n_contacts,), inputs = [self.nodes, self.contacts_new.list, self.contact_ret])
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
    dist = scalar(abd[2])
    
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

@wp.func
def compute_u_minus_utilde(pi: Node, dt: scalar):
    u_tilde = pi.v0 + dt * vec3(z, scalar(gravity), z)
    du = pi.v - u_tilde
    return du 

@wp.kernel
def rhs_kernel(p: wp.array(dtype = Node), precond: wp.array(dtype = mat33), rhs: wp.array(dtype = vec3), dt: scalar): 
    i = wp.tid()
    mi = p[i].mass
    if mi > z:
        du = compute_u_minus_utilde(p[i], dt)
        rhs[i] += mi * du
        precond[i] = precond[i] * dt * dt + wp.diag(vec3(mi, mi, mi))
    else:
        precond[i] = mat33(o)
        rhs[i] = vec3(z)

@wp.kernel
def du_kernel(precond: wp.array(dtype = mat33), rhs: wp.array(dtype = vec3), du: wp.array(dtype = vec3)):
    i = wp.tid()
    du[i] = wp.inverse(precond[i]) @ rhs[i]
        
@wp.func
def apply_du(du: vec3, p: Node, alpha: scalar, dt: scalar):
    state = Node()
    state = p

    state.v -= alpha * du 
    state.x = state.x0 + dt * state.v

    return state

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), p: wp.array(dtype = Node), alpha: scalar, dt: scalar):
    i = wp.tid()
    if p[i].mass > z: 
        p[i] = apply_du(du[i], p[i], alpha, dt)

@wp.kernel
def elastics_precond(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), precond: wp.array(dtype = mat33), b: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    if x[i].mass > z:
        last = x[i].last
        lhs = mat33(z)
        c0 = vec3(z)
        if last >= 0 and seg[last].nxt == i:
            lhs = wp.diag(vec3(kss / seg[last].l))
            c0 = compute_Css(x, seg, last)
        # rhs = mass / (dt * dt) * (yi - x[i].x) + kss * c0
        rhs = kss * c0
        if seg[i].nxt >= 0:
            c1 = compute_Css(x, seg, i)
            rhs -= kss * c1
            lhs += wp.diag(vec3(kss / seg[i].l))
        
        b[i] += rhs * dt
        precond[i] += lhs

@wp.kernel
def _precond_J_kernel(precond: wp.array(dtype = mat33), J: wp.array(dtype = scalar)):
    i = wp.tid()
    J[i] = wp.determinant(precond[i])

class PrimalRod(RodContact):
    def __init__(self, n_nodes, dt):
        RodContact.__init__(self, n_nodes, dt)

        self.alpha = 0.5
        self.precond = wp.zeros((self.n_nodes,), dtype = mat33)
        self.rhs = wp.zeros((self.n_nodes,), dtype = vec3)
        self.du = wp.zeros_like(self.rhs)

    def compute_contact_preconditioner_rhs(self):
        self.precond.zero_()
        self.rhs.zero_()

        wp.launch(preconditioner_diag_kernel, dim = (self.n_contacts, ), inputs = [self.nodes, self.contacts_new.list, self.precond, self.rhs, self.dt])

    def compute_mass_preconditioner_rhs(self):
        wp.launch(rhs_kernel, dim = (self.n_nodes,), inputs = [self.nodes, self.precond, self.rhs, self.dt])

    def compute_du(self):
        wp.launch(du_kernel, dim = (self.n_nodes,), inputs = [self.precond, self.rhs, self.du])
        J = wp.zeros((self.n_nodes,), dtype = scalar)
        if debug: 
            wp.launch(_precond_J_kernel, dim = (self.n_nodes,), inputs = [self.precond, J])
            du = self.du.numpy()
            Jnp = J.numpy()
            print("rhs norm = ", np.linalg.norm(self.rhs.numpy(), axis = 1).max())
            print("du norm = ", np.linalg.norm(du, axis = 1).max())
            print("J norm min = ", Jnp.min())

    def add_du(self, alpha):
        wp.launch(add_du_kernel, dim = (self.n_nodes,), inputs = [self.du, self.nodes, alpha, self.dt])
    
    def forward_states(self):
        pass

    def line_search(self):
        self.add_du(self.alpha)
        return self.alpha

    def prestep(self):
        super().prestep()
        self.detect_collision()
    
    def compute_elastics_preconditioner_rhs(self):
        if debug: 
            rhs_norm_before = np.linalg.norm(self.rhs.numpy(), axis = 1).max()
        wp.launch(elastics_precond, dim = (self.n_nodes,), inputs = [self.nodes, self.segs, self.precond, self.rhs, self.dt])
        if debug:
            rhs_after = self.rhs.numpy()
            rhs_norm_after = np.linalg.norm(rhs_after, axis = 1).max()
            print(f"rhs norm before elastics = {rhs_norm_before}, after elastics = {rhs_norm_after}")
            idx = np.isnan(rhs_after[:, 0]) | np.isnan(rhs_after[:, 1]) | np.isnan(rhs_after[:, 2])
            idx = np.arange(self.n_nodes)[idx]
            print(f"nan index = {idx}")
            nxt = self.segs.numpy()["nxt"]
            print(f"nxt at nan index last = {nxt[idx - 1]}")
            print(f"nxt at nan index = {nxt[idx]}")

    def vbd_step_position(self):
        '''
        override the position update
        '''
        for it in range(2):
            self.compute_contact_preconditioner_rhs()
            self.compute_elastics_preconditioner_rhs()
            self.compute_mass_preconditioner_rhs()
            self.compute_du()

            self.line_search()