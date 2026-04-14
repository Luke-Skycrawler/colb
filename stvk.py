import warp as wp 
from scalar_types import *
from BDF1 import BDFAffine, AffineState
from rbd_simple import Inertia
kappa = scalar(1e5)
@wp.func
def energy_ortho(state: BDFAffine) -> scalar:
    q = state.nxt.q
    e = scalar(0.0)
    for i in range(3):
        for j in range(3):
            term = wp.dot(q[i], q[j]) - scalar(wp.where(i == j, 1.0, 0.0))
            e += term * term
    return e * kappa
            
            

@wp.func
def grad_ortho(i: int, state: BDFAffine) -> vec3:
    q = state.nxt.q
    grad = -q[i]
    for j in range(3):
        grad += wp.dot(q[i], q[j]) * q[j]

    return grad * scalar(4.0) * kappa

@wp.func
def hessian_ortho(i: int, j: int, state: BDFAffine) -> mat33:
    q = state.nxt.q
    hess = mat33(scalar(0.0))
    if i == j:
        qiqiT = wp.outer(q[i], q[i]) 
        qiTqi = wp.dot(q[i], q[i]) - scalar(1.0)
        term2 = wp.diag(vec3(qiTqi))

        for k in range(3):
            hess += wp.outer(q[k], q[k])
        hess += qiqiT + term2
    else:
        hess = wp.outer(q[j], q[i]) + wp.diag(vec3(wp.dot(q[j], q[i])))
    return hess * scalar(4.0) * kappa
