import warp as wp 
from rbd_simple import Inertia
from scalar_types import *
from abd_simple import AbdComplex
from contact import ContactSolverBase, contact_volume, XConstraint, fetch_b0b1, thickness, ContactRet, CSRTriplets
from gauss_newton import LineSearchInterface, TripletsCSR
from scipy.sparse import csr_matrix
from warp.sparse import bsr_from_triplets, bsr_axpy
from warp.optim.linear import cg, bicgstab
from BDF1 import BDFAffine, AffineState
import dxslv
import numpy as np 
from geometry import Soup
from ortho import energy_ortho, grad_ortho, hessian_ortho
from ipctk_wp.distance.edge_edge import x_to_grad_psd_hess_ee
from ipctk_wp.distance.point_edge import point_edge_distance, point_edge_distance_gradient_hessian
from ipctk_wp.distance.point_point import point_point_distance_gradient_hessian
from barrier import barrier_energy, barrier_derivative, barrier_derivative2
from utils.data_collector import Profiller

@wp.struct
class Triplets:
    rows: wp.array(dtype=int)
    cols: wp.array(dtype=int)
    vals: wp.array(dtype=mat33)

solver_config = "cg"
gravity = scalar(-10.0)
eps = 1e-6
contact_stiffness = scalar(4e8)

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

@wp.kernel
def energy_contact(states: wp.array(dtype = BDFAffine), soup: Soup, contacts: wp.array(dtype = XConstraint), e: wp.array(dtype = scalar), dt: scalar):
    i = wp.tid()
    c = contacts[i]
    
    dab, v0, v1 = fetch_dist_v0v1(states, soup, c)
    dist = dab[2]
    if dist < c.l0:
        de = barrier_energy(dist * dist) * dt * dt * contact_stiffness
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
    else: 
        for ii in range(4):
            for jj in range(4):
                triplets.rows[os + ii + jj * 4] = i * 4 + ii
                triplets.cols[os + ii + jj * 4] = i * 4 + jj
                if ii == jj:
                    triplets.vals[os + ii + jj * 4] = wp.identity(3, dtype = scalar) * scalar(100.0)
                else:
                    triplets.vals[os + ii + jj * 4] = mat33(scalar(0.0))
            g[ii + i * 4] = vec3(scalar(0.0))
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

@wp.func 
def fetch_dist_v0v1(p: wp.array(dtype = BDFAffine), soup: Soup, c: XConstraint):
    l0 = c.l0
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]
    
    b0 = soup.body[i0]
    b1 = soup.body[i2]

    R0 = wp.transpose(p[b0].nxt.q)
    R1 = wp.transpose(p[b1].nxt.q)

    c0 = p[b0].nxt.c
    c1 = p[b1].nxt.c    

    x0 = R0 @ soup.xcs[i0] + c0
    x1 = R0 @ soup.xcs[i1] + c0
    x2 = R1 @ soup.xcs[i2] + c1
    x3 = R1 @ soup.xcs[i3] + c1

    dab = wp.closest_point_edge_edge(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), eps)
    v0 = wp.lerp(x0, x1, scalar(dab[0]))
    v1 = wp.lerp(x2, x3, scalar(dab[1]))

    dist = scalar(dab[2])

    return vec3(dab), v0, v1

@wp.kernel
def contact_hessian_ee(p: wp.array(dtype = BDFAffine), soup: Soup, contacts: wp.array(dtype = XConstraint), triplets: Triplets, b: wp.array(dtype = vec3), inertia: wp.array(dtype = Inertia), dt: scalar):
    i = wp.tid()
    o = scalar(1.0)
    z = scalar(0.0)
    c = contacts[i]
    
    dab, v0, v1 = fetch_dist_v0v1(p, soup, c)
    dist = dab[2]
    alpha = dab[0]
    beta = dab[1]

    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]

    h = wp.zeros((4, 4), dtype = mat33)
    hess_ei = wp.zeros((4, 4), dtype = mat33)
    hess_ej =  wp.zeros((4, 4), dtype = mat33)
    hess_cross = wp.zeros((4, 4), dtype = mat33)

    a0 = alpha == z
    a1 = alpha == o
    beta0 = beta == z 
    beta1 = beta == o
    point_point = (a0 or a1) and (beta0 or beta1)
    point_edge = a0 or a1 or beta0 or beta1
    if dist < c.l0 and not point_edge:
    # if dist < c.l0:
        b0 = soup.body[i0]
        b1 = soup.body[i2]

        R0 = wp.transpose(p[b0].nxt.q)
        R1 = wp.transpose(p[b1].nxt.q)

        c0 = p[b0].nxt.c
        c1 = p[b1].nxt.c    

        x0 = R0 @ soup.xcs[i0] + c0
        x1 = R0 @ soup.xcs[i1] + c0
        x2 = R1 @ soup.xcs[i2] + c1
        x3 = R1 @ soup.xcs[i3] + c1

        x0_tilde = soup.xcs[i0]
        x1_tilde = soup.xcs[i1]
        x2_tilde = soup.xcs[i2]
        x3_tilde = soup.xcs[i3]

        grad_dist, hess_dist = x_to_grad_psd_hess_ee(x0, x1, x2, x3)
        B_ = barrier_derivative(dist * dist)
        B__ = barrier_derivative2(dist * dist)

        hess_dist = B_ * hess_dist + B__ * wp.outer(grad_dist, grad_dist)
        grad_dist *= contact_stiffness * B_ * dt * dt
        hess_dist *= contact_stiffness * dt * dt

        
        Jei = wp.matrix_from_rows(
            vec4(o, x0_tilde.x, x0_tilde.y, x0_tilde.z),
            vec4(o, x1_tilde.x, x1_tilde.y, x1_tilde.z),
        )
        Jej = wp.matrix_from_rows(
            vec4(o, x2_tilde.x, x2_tilde.y, x2_tilde.z),
            vec4(o, x3_tilde.x, x3_tilde.y, x3_tilde.z),
        )

        g0 = vec3(grad_dist[0], grad_dist[1], grad_dist[2])
        g1 = vec3(grad_dist[3], grad_dist[4], grad_dist[5])
        g2 = vec3(grad_dist[6], grad_dist[7], grad_dist[8])
        g3 = vec3(grad_dist[9], grad_dist[10], grad_dist[11])

        
        for ii in range(4):
            for jj in range(4):
                h[ii, jj] = mat33(
                    hess_dist[ii * 3 + 0, jj * 3 + 0], hess_dist[ii * 3 + 0, jj * 3 + 1], hess_dist[ii * 3 + 0, jj * 3 + 2],
                    hess_dist[ii * 3 + 1, jj * 3 + 0], hess_dist[ii * 3 + 1, jj * 3 + 1], hess_dist[ii * 3 + 1, jj * 3 + 2],
                    hess_dist[ii * 3 + 2, jj * 3 + 0], hess_dist[ii * 3 + 2, jj * 3 + 1], hess_dist[ii * 3 + 2, jj * 3 + 2]
                )
        
        for ii in range(4):
            for jj in range(4):
                block_ei = mat33()
                block_ej = mat33()
                block_cross = mat33()
                
                for k in range(2):
                    for l in range(2):
                        block_ei += Jei[k, ii] * Jei[l, jj] * h[k, l]
                        block_ej += Jej[k, ii] * Jej[l, jj] * h[k + 2, l + 2]
                        block_cross += Jei[k, ii] * Jej[l, jj] * h[k, l + 2]

                hess_ei[ii, jj] = block_ei
                hess_ej[ii, jj] = block_ej
                hess_cross[ii, jj] = block_cross
                    
                        
                    
        for ii in range(4):
            gei = g0 * Jei[0, ii] + g1 * Jei[1, ii]
            gej = g2 * Jej[0, ii] + g3 * Jej[1, ii]
            if inertia[b0].m > z:
                wp.atomic_add(b, b0 * 4 + ii, gei)
            if inertia[b1].m > z:
                wp.atomic_add(b, b1 * 4 + ii, gej)

        for ii in range(4):
            for jj in range(4):
                m0 = inertia[b0].m
                m1 = inertia[b1].m
                if m0 > z:
                    # hess ei
                    iei = i * 64 + ii * 4 + jj
                    triplets.rows[iei] = b0 * 4 + ii 
                    triplets.cols[iei] = b0 * 4 + jj
                    triplets.vals[iei] = hess_ei[ii, jj]

                if m1 > z:
                    # hess ej 
                    iej = i * 64 + 16 + ii * 4 + jj
                    triplets.rows[iej] = b1 * 4 + ii
                    triplets.cols[iej] = b1 * 4 + jj
                    triplets.vals[iej] = hess_ej[ii, jj]

                if m0 > z and m1 > z:
                    # cross terms 
                    ic1 = i * 64 + 32 + ii * 4 + jj
                    ic2 = i * 64 + 48 + ii * 4 + jj
                    
                    triplets.rows[ic1] = b0 * 4 + ii
                    triplets.cols[ic1] = b1 * 4 + jj
                    triplets.vals[ic1] = hess_cross[ii, jj]

                    triplets.rows[ic2] = b1 * 4 + ii
                    triplets.cols[ic2] = b0 * 4 + jj
                    triplets.vals[ic2] = wp.transpose(hess_cross[jj, ii])

    elif dist < c.l0:
    # elif False:
        # _, da0 = point_edge_distance(x0, x2, x3)
        # _, da1 = point_edge_distance(x1, x2, x3)
        # _, db0 = point_edge_distance(x2, x0, x1)
        # _, db1 = point_edge_distance(x3, x0, x1)
        
        # a0 = da0 < c.l0
        # a1 = da1 < c.l0
        # b0 = db0 < c.l0
        # b1 = db1 < c.l0


        # point-point cases first:
        
        pe_stencil = wp.vec3i()
        if point_point:
            pp_stencil = wp.vec2i(0, 0)
            if a0 and beta0:
                pp_stencil = wp.vec2i(i0, i2)
            elif a0 and beta1:
                pp_stencil = wp.vec2i(i0, i3)
            elif a1 and beta0:
                pp_stencil = wp.vec2i(i1, i2)
            elif a1 and beta1: 
                pp_stencil = wp.vec2i(i1, i3)

            ii0 = pp_stencil[0]
            ii1 = pp_stencil[1]
            b0 = soup.body[i0]
            b1 = soup.body[i2]
            
            R0 = wp.transpose(p[b0].nxt.q)
            R1 = wp.transpose(p[b1].nxt.q)
            
            x0_tilde = soup.xcs[ii0]
            x1_tilde = soup.xcs[ii1]
            x0 = R0 @ x0_tilde + p[b0].nxt.c
            x1 = R1 @ x1_tilde + p[b1].nxt.c

            grad_pp, hess_pp = point_point_distance_gradient_hessian(x0, x1)
            
            B_ = barrier_derivative(dist * dist)
            B__ = barrier_derivative2(dist * dist)
            
            # hess_pp = B_ * hess_pp + B__ * wp.outer(grad_pp, grad_pp)
            hess_pp = B__ * wp.outer(grad_pp, grad_pp)
            
            grad_pp *= contact_stiffness * B_ * dt * dt
            hess_pp *= contact_stiffness * dt * dt


            Jpi = vec4(o, x0_tilde.x, x0_tilde.y, x0_tilde.z)
            Jpj = vec4(o, x1_tilde.x, x1_tilde.y, x1_tilde.z)
            
            g0 = vec3(grad_pp[0], grad_pp[1], grad_pp[2])
            g1 = vec3(grad_pp[3], grad_pp[4], grad_pp[5])

            
            for ii in range(2):
                for jj in range(2):
                    h[ii, jj] = mat33(
                        hess_pp[ii * 3 + 0, jj * 3 + 0], hess_pp[ii * 3 + 0, jj * 3 + 1], hess_pp[ii * 3 + 0, jj * 3 + 2],
                        hess_pp[ii * 3 + 1, jj * 3 + 0], hess_pp[ii * 3 + 1, jj * 3 + 1], hess_pp[ii * 3 + 1, jj * 3 + 2],
                        hess_pp[ii * 3 + 2, jj * 3 + 0], hess_pp[ii * 3 + 2, jj * 3 + 1], hess_pp[ii * 3 + 2, jj * 3 + 2]
                    )
            
            for ii in range(4):
                for jj in range(4):                    
                    block_ei = Jpi[ii] * Jpi[jj] * h[0, 0]
                    block_ej = Jpj[ii] * Jpj[jj] * h[1, 1]
                    block_cross = Jpi[ii] * Jpj[jj] * h[0, 1]
                    

                    hess_ei[ii, jj] = block_ei
                    hess_ej[ii, jj] = block_ej
                    hess_cross[ii, jj] = block_cross
                        
                            
                        
            for ii in range(4):
                gei = g0 * Jpi[ii]
                gej = g1 * Jpj[ii]
                if inertia[b0].m > z:
                    wp.atomic_add(b, b0 * 4 + ii, gei)
                if inertia[b1].m > z:
                    wp.atomic_add(b, b1 * 4 + ii, gej)

            for ii in range(4):
                for jj in range(4):
                    m0 = inertia[b0].m
                    m1 = inertia[b1].m
                    if m0 > z:
                        # hess ei
                        iei = i * 64 + ii * 4 + jj
                        triplets.rows[iei] = b0 * 4 + ii 
                        triplets.cols[iei] = b0 * 4 + jj
                        triplets.vals[iei] = hess_ei[ii, jj]

                    if m1 > z:
                        # hess ej 
                        iej = i * 64 + 16 + ii * 4 + jj
                        triplets.rows[iej] = b1 * 4 + ii
                        triplets.cols[iej] = b1 * 4 + jj
                        triplets.vals[iej] = hess_ej[ii, jj]

                    if m0 > z and m1 > z:
                        # cross terms 
                        ic1 = i * 64 + 32 + ii * 4 + jj
                        ic2 = i * 64 + 48 + ii * 4 + jj
                        
                        triplets.rows[ic1] = b0 * 4 + ii
                        triplets.cols[ic1] = b1 * 4 + jj
                        triplets.vals[ic1] = hess_cross[ii, jj]

                        triplets.rows[ic2] = b1 * 4 + ii
                        triplets.cols[ic2] = b0 * 4 + jj
                        triplets.vals[ic2] = wp.transpose(hess_cross[jj, ii])
            

                
        elif a0:
            pe_stencil = wp.vec3i(i0, i2, i3)
        elif a1: 
            pe_stencil = wp.vec3i(i1, i2, i3)
        elif beta0: 
            pe_stencil = wp.vec3i(i2, i0, i1)
        elif beta1: 
            pe_stencil = wp.vec3i(i3, i0, i1)
        
        if not point_point:
        # if False:
            ii0 = pe_stencil[0] 
            ii1 = pe_stencil[1]
            ii2 = pe_stencil[2]

            b0 = soup.body[ii0]
            b1 = soup.body[ii1]
            
            R0 = wp.transpose(p[b0].nxt.q)  
            R1 = wp.transpose(p[b1].nxt.q)

            x0_tilde = soup.xcs[ii0]
            x1_tilde = soup.xcs[ii1]
            x2_tilde = soup.xcs[ii2]
            x0 = R0 @ x0_tilde + p[b0].nxt.c
            x1 = R1 @ x1_tilde + p[b1].nxt.c
            x2 = R1 @ x2_tilde + p[b1].nxt.c

            _grad_pe, hess_pe = point_edge_distance_gradient_hessian(x0, x1, x2)
            grad_pe = vec9(_grad_pe[0,0], _grad_pe[0,1], _grad_pe[0,2], _grad_pe[1,0], _grad_pe[1,1], _grad_pe[1,2], _grad_pe[2,0], _grad_pe[2,1], _grad_pe[2,2])
            
            B_ = barrier_derivative(dist * dist)
            B__ = barrier_derivative2(dist * dist)

            hess_pe = B_ * hess_pe + B__ * wp.outer(grad_pe, grad_pe)
            grad_pe *= contact_stiffness * B_ * dt * dt
            hess_pe *= contact_stiffness * dt * dt

            
            Jp = vec4(o, x0_tilde.x, x0_tilde.y, x0_tilde.z)
            Jej = wp.matrix_from_rows(
                vec4(o, x1_tilde.x, x1_tilde.y, x1_tilde.z),
                vec4(o, x2_tilde.x, x2_tilde.y, x2_tilde.z),
            )

            g0 = vec3(grad_pe[0], grad_pe[1], grad_pe[2])
            g1 = vec3(grad_pe[3], grad_pe[4], grad_pe[5])
            g2 = vec3(grad_pe[6], grad_pe[7], grad_pe[8])

            
            for ii in range(3):
                for jj in range(3):
                    h[ii, jj] = mat33(
                        hess_pe[ii * 3 + 0, jj * 3 + 0], hess_pe[ii * 3 + 0, jj * 3 + 1], hess_pe[ii * 3 + 0, jj * 3 + 2],
                        hess_pe[ii * 3 + 1, jj * 3 + 0], hess_pe[ii * 3 + 1, jj * 3 + 1], hess_pe[ii * 3 + 1, jj * 3 + 2],
                        hess_pe[ii * 3 + 2, jj * 3 + 0], hess_pe[ii * 3 + 2, jj * 3 + 1], hess_pe[ii * 3 + 2, jj * 3 + 2]
                    )
            
            for ii in range(4):
                for jj in range(4):
                    block_ej = mat33()
                    block_cross = mat33()
                    
                    block_ei = Jp[ii] * Jp[jj] * h[0, 0]
                    
                    for l in range(2):
                        block_cross += Jp[ii] * Jej[l, jj] * h[0, l + 1]
                    for k in range(2):
                        for l in range(2):
                            block_ej += Jej[k, ii] * Jej[l, jj] * h[k + 1, l + 1]

                    hess_ei[ii, jj] = block_ei
                    hess_ej[ii, jj] = block_ej
                    hess_cross[ii, jj] = block_cross
                        
                            
                        
            for ii in range(4):
                gei = g0 * Jp[ii]
                gej = g1 * Jej[0, ii] + g2 * Jej[1, ii]
                if inertia[b0].m > z:
                    wp.atomic_add(b, b0 * 4 + ii, gei)
                if inertia[b1].m > z:
                    wp.atomic_add(b, b1 * 4 + ii, gej)

            for ii in range(4):
                for jj in range(4):
                    m0 = inertia[b0].m
                    m1 = inertia[b1].m
                    if m0 > z:
                        # hess ei
                        iei = i * 64 + ii * 4 + jj
                        triplets.rows[iei] = b0 * 4 + ii 
                        triplets.cols[iei] = b0 * 4 + jj
                        triplets.vals[iei] = hess_ei[ii, jj]

                    if m1 > z:
                        # hess ej 
                        iej = i * 64 + 16 + ii * 4 + jj
                        triplets.rows[iej] = b1 * 4 + ii
                        triplets.cols[iej] = b1 * 4 + jj
                        triplets.vals[iej] = hess_ej[ii, jj]

                    if m0 > z and m1 > z:
                        # cross terms 
                        ic1 = i * 64 + 32 + ii * 4 + jj
                        ic2 = i * 64 + 48 + ii * 4 + jj
                        
                        triplets.rows[ic1] = b0 * 4 + ii
                        triplets.cols[ic1] = b1 * 4 + jj
                        triplets.vals[ic1] = hess_cross[ii, jj]

                        triplets.rows[ic2] = b1 * 4 + ii
                        triplets.cols[ic2] = b0 * 4 + jj
                        triplets.vals[ic2] = wp.transpose(hess_cross[jj, ii])

             
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
        self.pattern_old = None 
        self.profiler = Profiller(max_iters = 16, frames = 50)

    def compute_contact_gh(self):
        return 
        self.rhs.zero_()
        self.triplets.rows.zero_()
        self.triplets.cols.zero_()
        self.triplets.vals.zero_()
        wp.launch(compute_contact_gh_kernel, (self.n_contacts,), inputs = [self.history, self.inertia, self.soup, self.contacts_new.list, self.triplets, self.rhs, self.dt])


    def to_csr(self, triplets: Triplets): 
        a = TripletsCSR() 
        nnz = (self.n_bodies + self.n_contacts * 4) * 144
        a.rows = wp.zeros((nnz,), dtype = int)
        a.cols = wp.zeros_like(a.rows)
        a.vals = wp.zeros((nnz,), dtype = scalar)

        wp.launch(bsr2csr, dim = (nnz // 9,), inputs = [triplets, a])

        # prune numerical zeros off is necessary, because the direct solver will directly copy the first nnz values
        return bsr_from_triplets(self.n_bodies * 12, self.n_bodies * 12, a.rows, a.cols, a.vals, prune_numerical_zeros=False)

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
            # if iter == 0: 
            new_pattern = self.bodywise_actual_contact()
            cond = self.pattern_old is None or self.pattern_diff(new_pattern, self.pattern_old)
            self.pattern_old = new_pattern
            # if cond:
            if True:
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

        residue = np.linalg.norm(self.rhs.numpy())
        du_norm = np.max(np.abs(self.du.numpy()))
        return residue

    def pattern_diff(self, a, b):
        a_columns = a.columns.numpy()
        b_columns = b.columns.numpy()
        a_offsets = a.offsets.numpy()
        b_offsets = b.offsets.numpy()

        # if a_columns.shape[0] == b_columns.shape[0] and np.max(np.abs(a_columns - b_columns)) == 0 and np.max(np.abs(a_offsets - b_offsets)) == 0:
        #     return False
        return True

    def forward_states(self): 
        wp.launch(forward_states, self.n_bodies, inputs = [self.history, self.dt])
        self.frame += 1

    def step(self):
        for ss in range(1):
            with wp.ScopedTimer("newton step"):
                newton = True
                iter = 0
                max_iter = 16
                while newton: 
                    self.detect_collision()
                    self.triplets.rows.zero_()
                    self.triplets.cols.zero_()
                    self.triplets.vals.zero_()
                    self.rhs.zero_()
                    wp.launch(inertia_grad_hess, self.n_bodies, inputs = [self.rhs, self.triplets, self.history, self.inertia, self.dt])

                    wp.launch(contact_hessian_ee, dim = (self.n_contacts, ), inputs = [self.history, self.soup, self.contacts_new.list, self.triplets, self.rhs, self.inertia, self.dt])
                    
                    residue = self.compute_du(iter) 
                    alpha = self.line_search()
                    # alpha = self.line_search_batch()
                    iter += 1

                    self.profiler.convergence[self.frame, iter - 1] = residue
                    self.profiler.alphas[self.frame, iter - 1] = alpha

                    print(f"    iter: {iter}, residue: {residue: .2e}, alpha = {alpha: .2e}")
                    newton = not (residue < 1e-5 or iter >= max_iter)
                    
                self.forward_states()
                self.profiler.convergence[iter: max_iter] = self.profiler.convergence[self.frame, iter - 1]

    def bodywise_actual_contact(self):
        triplets = CSRTriplets()
        triplets.rows = wp.zeros((self.n_contacts * 2,), dtype = int)
        triplets.cols = wp.zeros((self.n_contacts * 2,), dtype = int)
        triplets.vals = wp.ones((self.n_contacts * 2,), dtype = int)
        wp.launch(bodywise_actual_contact, dim = (self.n_contacts, ), inputs = [self.history, self.contacts_new.list, triplets, self.soup])
        adjacency = bsr_from_triplets(self.n_bodies, self.n_bodies, triplets.rows, triplets.cols, triplets.vals)
        return adjacency

    # def line_search_iterative(self):
    #     alpha = 1.0
    #     backup = wp.clone(self.history)
    #     # E0 = self.compute_g()

    #     while True:
    #         wp.copy(self.history, backup)    
    #         self.add_du(alpha)
    #         break

    #     return alpha

    def compute_g(self, update_contact = True):
        '''
        g = h ^ 2 * (E_ortho + B) + 1/2 |q - \tilde{q}|_M^2
        '''
        self.g.zero_()
        wp.launch(energy_inertia, (self.n_bodies,), inputs = [self.history, self.g, self.inertia, self.dt])
        if update_contact:
            self.detect_collision()
        wp.launch(energy_contact, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.g, self.dt])

        return self.g.numpy()[0]

    # def compute_dq(self, iter): 
    #     hess = bsr_from_triplets(self.n_bodies * 4, self.n_bodies * 4, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        
    #     bicgstab(hess, self.rhs, self.du, tol = 1e-5)
    #     return np.max(np.abs(self.du.numpy()))
        

    def add_du(self, alpha):
        # wp.launch(add_vec12_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt])

    def get_contact_points(self):
        wp.launch(get_contact_points, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.contact_ret])
        # filter d > 2 * thickness
        dists = self.contact_ret.dists.numpy()[:self.n_contacts]
        points = self.contact_ret.points.numpy()[:self.n_contacts]
        
        valid = dists < thickness * 2.0
        magnitudes = np.abs(dists[valid] - thickness * 2.0)
        # return np.zeros((0, 3)), np.zeros((0,))
        return points[valid], magnitudes

@wp.kernel
def get_contact_points(p: wp.array(dtype = BDFAffine), soup: Soup, xconstraints: wp.array(dtype = XConstraint), contact_ret: ContactRet):
    i = wp.tid()
    c = xconstraints[i]
    dab, v0, v1 = fetch_dist_v0v1(p, soup, c)
    dist = dab[2]
    contact_ret.points[i] = (v0 + v1) * scalar(0.5)
    contact_ret.dists[i] = dist

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
    dui = du[i * 4]
    dq = wp.matrix_from_rows(du[i * 4 + 1], du[i * 4 + 2], du[i * 4 + 3])
    history[i] = apply_du(dui, dq, history[i], alpha, dt)

@wp.kernel
def bodywise_actual_contact(history: wp.array(dtype = BDFAffine), contacts: wp.array(dtype = XConstraint), triplets: CSRTriplets, soup: Soup):
    i = wp.tid()
    c = contacts[i]
    b0, b1 = fetch_b0b1(c, soup)
    dab, v0, v1 = fetch_dist_v0v1(history, soup, c)
    dist = dab[2]
    if b0 != b1 and dist < c.l0: 
        triplets.rows[i * 2 + 0] = b0
        triplets.cols[i * 2 + 0] = b1
        triplets.rows[i * 2 + 1] = b1
        triplets.cols[i * 2 + 1] = b0

