import warp as wp 
from quat_util import RigidState, scalar, vec3, vec4, mat33, mat44, quat_mult
@wp.struct 
class BDFHistory: 
    '''
    BDFHistory stores the history of rigid body states needed for BDF1 time integration.
    '''
    now: RigidState
    nxt: RigidState

@wp.func 
def cdot(history: BDFHistory, dt: scalar) -> vec3:
    '''
    inputs: 
    - now: current rigid body state
    - nxt: next rigid body state
    - dt: time step
    '''
    return (history.nxt.c - history.now.c) / dt

@wp.func 
def qdot(history: BDFHistory, dt: scalar) -> vec4: 
    '''
    inputs: 
    - now: current rigid body state
    - nxt: next rigid body state
    - dt: time step
    '''
    return (history.nxt.q - history.now.q) / dt

@wp.func 
def vdot(history: BDFHistory, dt: scalar) -> vec3:
    '''
    inputs: 
    - now: current rigid body state
    - nxt: next rigid body state
    - dt: time step
    '''
    return (history.nxt.v - history.now.v) / dt

@wp.func 
def wdot(history: BDFHistory, dt: scalar) -> vec4:
    '''
    inputs: 
    - now: current rigid body state
    - nxt: next rigid body state
    - dt: time step
    '''
    return (history.nxt.w - history.now.w) / dt

@wp.func 
def dcdot_dc(history: BDFHistory, dt: scalar) -> mat33:
    return wp.identity(3, scalar) / dt

@wp.func
def dqdot_dq(history: BDFHistory, dt: scalar) -> mat44:
    return wp.identity(4, scalar) / dt

@wp.func
def dwdot_dw(history: BDFHistory, dt: scalar) -> mat44:
    return wp.identity(4, scalar) / dt

@wp.func
def dvdot_dv(history: BDFHistory, dt: scalar) -> mat33:
    return wp.identity(3, scalar) / dt

    
@wp.kernel
def forward_states(history: wp.array(dtype = BDFHistory), dt: scalar):
    i = wp.tid()
    history[i].nxt.v = (history[i].nxt.c - history[i].now.c) / dt

    q_prev = history[i].now.q
    q_prev_inv = vec4(-q_prev.x, -q_prev.y, -q_prev.z, q_prev.w)
    dq = quat_mult(history[i].nxt.q, q_prev_inv)
    
    omega = scalar(2.0) * vec3(dq.x, dq.y, dq.z) / dt
    if dq.w < scalar(0.0):
        omega = -omega

    history[i].nxt.omega = omega
    history[i].now = history[i].nxt

@wp.kernel
def init(history: wp.array(dtype = BDFHistory)):
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    omega = scalar(10.0)
    d = scalar(0.01)
    history[i].now.q = vec4(z, z, z, o)
    history[i].now.w = vec4(z, omega, d, z)
    history[i].now.c = vec3(z, z, z)
    history[i].now.v = vec3(z, z, z)


    history[i].nxt.q = vec4(z, z, z, o)
    history[i].nxt.w = vec4(z, z, z, z)
    history[i].nxt.c = vec3(z, z, z)
    history[i].nxt.v = vec3(z, z, z)

