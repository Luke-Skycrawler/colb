from scalar_types import *
from BDF1 import BDFAffine, AffineState
from warp.optim.linear import cg, bicgstab
from warp.sparse import bsr_set_from_triplets, bsr_zeros

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
gravity = scalar(0.0)
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




@wp.struct
class BSR:
    stride: wp.array(dtype = int)
    offset: wp.array(dtype = int)
    blocks: wp.array(dtype = mat33)

    
def bsr_empty(n = 1, nnz = 1):
    bsr = BSR()
    bsr.stride = wp.zeros(n, dtype = int)
    bsr.offset = wp.zeros(n, dtype = int)
    bsr.blocks = wp.zeros(nnz  * 16, dtype = mat33)
    return bsr


@wp.struct
class AffineBodyStates:
    states: wp.array(dtype = BDFAffine)

def affine_body_states_empty(n_bodies):
    
    states = AffineBodyStates()

    states.states = wp.zeros(n_bodies, dtype = BDFAffine)

    return states

@wp.kernel
def _init_spin(history: AffineBodyStates):
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    history.states[i].now.c = vec3(z, z, z)
    history.states[i].now.q = wp.diag(vec3(o))
    history.states[i].now.v = vec3(z, z, z)
    history.states[i].now.qdot = wp.skew(vec3(o, z, z))

    history.states[i].nxt = history.states[i].now

@wp.kernel
def _q_gets_q0(states: AffineBodyStates):
    i = wp.tid()
    states.states[i].nxt = states.states[i].now

@wp.kernel
def _update_q(states: AffineBodyStates, dq: wp.array(dtype = vec3)):
    i = wp.tid()

    states.states[i].nxt.c = states.states[i].nxt.c - dq[i * 4 + 0]
    # states.p[i] = states.p[i] - dq[i * 4 + 0]
    q1 = states.states[i].nxt.q[0] - dq[i * 4 + 1]
    q2 = states.states[i].nxt.q[1] - dq[i * 4 + 2]
    q3 = states.states[i].nxt.q[2] - dq[i * 4 + 3]
    states.states[i].nxt.q = wp.matrix_from_rows(q1, q2, q3)

@wp.kernel
def _update_q0qdot(states: AffineBodyStates):
    i = wp.tid()
    # states.pdot[i] = (states.p[i] - states.p0[i]) / dt
    # states.Adot[i] = (states.A[i] - states.A0[i]) / dt

    states.states[i].nxt.v = (states.states[i].nxt.c - states.states[i].now.c) / dt
    states.states[i].nxt.qdot = (states.states[i].nxt.q - states.states[i].now.q) / dt

    # states.p0[i] = states.p[i]
    # states.A0[i] = states.A[i]
    states.states[i].now = states.states[i].nxt

@wp.kernel
def _set_triplets(rows: wp.array(dtype = int), cols: wp.array(dtype = int)):
    for i in range(4):
        for j in range(4):
            rows[i + j * 4] = i
            cols[i + j * 4] = j


if __name__ == "__main__":
    wp.init()
    bsr = bsr_empty(1) 
    states = affine_body_states_empty(1)
    # A = wp.zeros(1, dtype = mat33)
    g = wp.zeros(4, dtype = vec3)
    dq = wp.zeros_like(g)
    wp.launch(_init_spin, 1, inputs = [states])
    inertia = InertialEnergy()
    import polyscope as ps 
    import igl
    V, _, _, F, _, _  = igl.read_obj("assets/cube.obj")
    hess = bsr_zeros(4, 4, mat33)
    rows = wp.zeros(16, dtype = int)
    cols = wp.zeros(16, dtype = int)
    values = wp.zeros(16, dtype = mat33)
    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    # for frame in range(10):
    while True:
        
        # wp.copy(states.A, states.A0)
        # wp.copy(states.p, states.p0)
        wp.launch(_q_gets_q0, 1, inputs = [states])
        it = 0 
        while True:
            inertia.gradient(g, states.states)
            inertia.hessian(bsr, states.states)


            values.assign(bsr.blocks.flatten())
            # print(bsr.blocks.numpy())

            wp.launch(_set_triplets, 1, inputs = [rows, cols])
            bsr_set_from_triplets(hess, rows, cols, values)
            bicgstab(hess, g, dq, 1e-4)

            # print(bsr.blocks.numpy())
            # print(hess.values.numpy())
            # print(g.numpy())
            # print(dq.numpy())

            wp.launch(_update_q, 1, inputs = [states, dq])
            
            it += 1
            if it > 1:
                inertia.gradient(g, states.states)
                # print("residue gradient: ", g.numpy())
                break
        
        Anp = states.states.numpy()["now"]["q"]
        pnp = states.states.numpy()["now"]["c"]

        for i in range(1):
            A = Anp[i].T
            p = pnp[i]
            x_view = A @ V.T + p.reshape(3, 1)

        wp.launch(_update_q0qdot, 1, inputs = [states])
        mesh.update_vertex_positions(x_view.T)
        print("a dot = ", states.states.numpy()["nxt"]["qdot"])
        ps.frame_tick()
    ps.show()

    
    
    