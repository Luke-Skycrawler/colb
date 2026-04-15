from scalar_types import *
from BDF1 import BDFAffine, AffineState

wp.config.max_unroll = 1
wp.config.enable_backward = False
@wp.struct
class Triplets:
    rows: wp.array(dtype=int)
    cols: wp.array(dtype=int)
    vals: wp.array(dtype=mat33)

vec5i = wp.types.vector(length = 5, dtype = int)
dt = scalar(1e-2)
mass = scalar(1e3)
# I0 = scalar(1e2)
I0 = scalar(64)
gravity = scalar(-9.8)
stiffness = scalar(1e9)



class InertialEnergy:
    def __init__(self):
        self.e = wp.zeros(1, dtype = scalar)

    def energy(self, states):
        e = self.e
        e.zero_()
        wp.launch(energy_inertia, states.shape, inputs = [states, e])
        return self.e.numpy()[0]
    
    def gradient(self, g, states):
        g.zero_()
        wp.launch(flattened_gradient_inertia, states.shape, inputs = [g, states])

    def hessian(self, triplets, states):
        wp.launch(bsr_hessian_inertia, states.shape, inputs = [triplets, states])

@wp.kernel
def energy_inertia(states: wp.array(dtype = BDFAffine), e: wp.array(dtype = scalar)):
    i = wp.tid()
    state = states[i]

    
    A_tilde = tildeA(state.now.q, state.now.qdot)
    p_tilde = tildep(state.now.c, state.now.v)
    
    dqTMdq = norm_M(state.nxt.q, state.nxt.c, A_tilde, p_tilde)
    de = energy_ortho(state.nxt.q) * dt * dt + scalar(0.5) * dqTMdq
    wp.atomic_add(e, 0, de)

@wp.func
def norm_M(A: mat33, p: vec3, A_tilde: mat33, p_tilde: vec3) -> scalar:
    dq0 = p - p_tilde
    dq1 = A[0] - A_tilde[0]
    dq2 = A[1] - A_tilde[1]
    dq3 = A[2] - A_tilde[2]

    return wp.dot(dq0, dq0) * mass + (wp.dot(dq1, dq1) + wp.dot(dq2, dq2) + wp.dot(dq3, dq3)) * I0


@wp.func
def energy_ortho(A: mat33) -> scalar:
    e = scalar(0.0)
    for i in range(3):
        for j in range(3):
            term = wp.dot(A[i], A[j]) - scalar(wp.where(i == j, 1.0, 0.0))
            e += term * term
    return e * stiffness
            
            

@wp.func
def grad_ortho(i: int, A: mat33) -> vec3:
    grad = -A[i]
    for j in range(3):
        grad += wp.dot(A[i], A[j]) * A[j]

    return grad * scalar(4.0) * stiffness

@wp.func
def hessian_ortho(i: int, j: int, A: mat33) -> mat33:
    hess = mat33(scalar(0.0))
    if i == j:
        qiqiT = wp.outer(A[i], A[i]) 
        qiTqi = wp.dot(A[i], A[i]) - scalar(1.0)
        term2 = wp.diag(vec3(qiTqi))

        for k in range(3):
            hess += wp.outer(A[k], A[k])
        hess += qiqiT + term2
    else:
        hess = wp.outer(A[j], A[i]) + wp.diag(vec3(wp.dot(A[j], A[i])))
    return hess * scalar(4.0) * stiffness


@wp.kernel
def bsr_hessian_inertia(triplets: Triplets, states: wp.array(dtype = BDFAffine)):
    i = wp.tid()
    # os = offset(i, i, bsr)
    os = i * 16
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
def flattened_gradient_inertia(g: wp.array(dtype = vec3), states: wp.array(dtype = BDFAffine)):
    i = wp.tid()

    state = states[i]
    for ii in range(1, 4):
        g[ii + i * 4] = dt * dt * grad_ortho(ii - 1, state.nxt.q)

    A_tilde = tildeA(state.now.q, state.now.qdot)
    p_tilde = tildep(state.now.c, state.now.v)
    q0, q1, q2, q3 = Mdq(state.nxt.q, state.nxt.c, A_tilde, p_tilde)

    g[0 + i * 4] += q0
    g[1 + i * 4] += q1
    g[2 + i * 4] += q2
    g[3 + i * 4] += q3


@wp.func
def Mdq(A: mat33, p: vec3, A_tilde: mat33, p_tilde: vec3):
    q0 = p - p_tilde
    q1 = A[0] - A_tilde[0]
    q2 = A[1] - A_tilde[1]
    q3 = A[2] - A_tilde[2]
    return q0 * mass, q1 * I0, q2 * I0, q3 * I0

@wp.func
def tildeA(A0: mat33, Adot: mat33) -> mat33:
    return A0 + dt * Adot

@wp.func
def tildep(p0: vec3, pdot: vec3) -> vec3:
    return p0 + dt * pdot + dt * dt * vec3(scalar(0.0), gravity, scalar(0.0))
