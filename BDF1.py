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

    