import warp as wp 
import numpy as np 
from primal import PrimalRbd, XConstraint, stiffness, eps, gravity, compute_u_minus_utilde
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from xpbd_contact import fetch_b0b1,  fetch_dist_n_r0r1
from BDF1 import BDFHistory
from rbd_simple import Inertia
from geometry import Soup

@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar, color: int, colors: wp.array(dtype = int)):
    i = wp.tid()
    c = colors[i]
    # n_bodies = history.shape[0]
    if c == color:
        history[i].nxt.v -= alpha * du[i * 2] 
        history[i].nxt.c = history[i].now.c + dt * history[i].nxt.v
        
        history[i].nxt.omega -= alpha * du[i * 2 + 1]
        history[i].nxt.q = history[i].now.q + scalar(0.5) * wp.transpose(Gq(history[i].nxt.q)) @ history[i].nxt.omega * dt
        history[i].nxt.q = wp.normalize(history[i].nxt.q)


class GSPrimalRbd(PrimalRbd):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)
        self.define_color()
        self.alpha = 1.0
    
    def define_color(self):
        angles = np.array([
            [90., 0., 90.],
            [0., 0., 90.],
            [90., 90., 0.],
        ])
        colors = []
        for obj in self.kinetic_objects:
            err = np.abs(obj.euler - angles)
            color = np.argmin(np.sum(err, axis=1))
            colors.append(color)

        colors = np.array(colors)
        self.colors = wp.array(colors, dtype = int)

    def add_du(self, alpha, color): 
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt, color, self.colors])


    def step(self):
        for ss in range(10):
            newton = True
            iter = 0
            max_iter = 4
            self.detect_collision()
            while newton: 
                for color in range(3):
                    self.compute_preconditioner()
                    self.compute_rhs()
                    
                    du_norm = self.compute_du() 
                    self.add_du(self.alpha, color)
                iter += 1
                # print(f"    iter: {iter}, du norm: {du_norm}")
                newton = not (du_norm < 1e-5 or iter >= max_iter)
        
            self.forward_states()

@wp.kernel
def rhs_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), precond: wp.array3d(dtype = scalar), rhs: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    n_bodies = p.shape[0]
    z = scalar(0.0)
    mi = mass[i].m

    if mi > z: 
        Ji = mass[i].J
        du, domega = compute_u_minus_utilde(p[i], dt)
        rhs[i * 2] += mi * du
        rhs[i * 2 + 1] += Ji * domega

        for ii in range(6):
            for jj in range(6):
                dd = precond[i, ii, jj] * dt * dt
                if ii == jj:
                    if ii < 3:
                        dd += scalar(mi)
                    else: 
                        dd += scalar(Ji)
                precond[i, ii, jj] = dd

    else:
        for ii in range(6):
            for jj in range(6):
                dd = z
                if ii == jj:
                    dd = scalar(1.0)
                precond[i, ii, jj] = dd

        rhs[i * 2] = vec3(z)
        rhs[i * 2 + 1] = vec3(z)

@wp.kernel
def preconditioner_diag_kernel(p: wp.array(dtype = BDFHistory), mass: wp.array(dtype = Inertia), soup: Soup, xconstraints: wp.array(dtype = XConstraint), precond: wp.array3d(dtype = scalar), rhs: wp.array(dtype = vec3), dt: scalar):
    i = wp.tid()
    c = xconstraints[i]
    k = scalar(stiffness)

    b0, b1 = fetch_b0b1(c, soup)
    dist, n, r1, r2 = fetch_dist_n_r0r1(p[b0], p[b1], soup, c)
    l0 = c.l0

    v10 = n * dist
    if dist < l0:         
        a1 = wp.cross(r1, n)
        a2 = wp.cross(r2, n)

        J1 = wp.vector(
            n[0], n[1], n[2], a1[0], a1[1], a1[2],
            length = 6
        )
        J2 = wp.vector(
            -n[0], -n[1], -n[2], -a2[0], -a2[1], -a2[2],
            length = 6
        )

        # for ii in range(6):
        #     for jj in range(6):
        #         wp.atomic_add(precond, b0, ii, jj, J1[ii] * J1[jj] * k)
        #         wp.atomic_add(precond, b1, ii, jj, J2[ii] * J2[jj] * k)

        H1 = wp.tile(wp.outer(J1, J1) * k)
        wp.tile_atomic_add(precond, H1, (b0, 0, 0))

        H2 = wp.tile(wp.outer(J2, J2) * k)
        wp.tile_atomic_add(precond, H2, (b1, 0, 0))
        
        f1 = k * (l0 - dist) * n * dt
        tau1 = wp.cross(r1, f1) 
        tau2 = wp.cross(r2, -f1)
        wp.atomic_add(rhs, b0 * 2, -f1)
        wp.atomic_add(rhs, b1 * 2, f1)
        wp.atomic_add(rhs, b0 * 2 + 1, -tau1)
        wp.atomic_add(rhs, b1 * 2 + 1, -tau2)

@wp.kernel
def du_kernel(precond: wp.array3d(dtype = scalar), rhs: wp.array(dtype = vec3), du: wp.array(dtype = vec3)):
    i = wp.tid()
    hii = wp.tile_load(precond[i], (6, 6), (0, 0))
    Lii = wp.tile_cholesky(hii)
    yi = wp.tile(wp.vector(
        rhs[i * 2][0], rhs[i * 2][1], rhs[i * 2][2], rhs[i * 2 + 1][0], rhs[i * 2 + 1][1], rhs[i * 2 + 1][2],
        length = 6
    ))
    du_i = wp.tile_cholesky_solve(Lii, yi)
    
    # dux = vec3(du_i[0], du_i[1], du_i[2])
    # duomega = vec3(du_i[3], du_i[4], du_i[5])

    dux = vec3(du_i[0, 0], du_i[1, 0], du_i[2, 0])
    duomega = vec3(du_i[3, 0], du_i[4, 0], du_i[5, 0])

    du[i * 2] = dux
    du[i * 2 + 1] = duomega

class VBDRbd(GSPrimalRbd):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)

        # overwrites the preconditioner array defined in primal.py
        self.precond = wp.zeros((self.n_bodies, 6, 6), dtype = scalar)


    def compute_preconditioner(self):
        wp.launch(preconditioner_diag_kernel, dim = (self.n_contacts, ), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.precond, self.rhs, self.dt])

    def compute_rhs(self):
        wp.launch(rhs_kernel, dim = (self.n_bodies, ), inputs = [self.history, self.inertia, self.precond, self.rhs, self.dt])

    def compute_du(self):
        wp.launch_tiled(du_kernel, dim = self.n_bodies * 2, inputs = [self.precond, self.rhs, self.du], block_dim = 1)
        
