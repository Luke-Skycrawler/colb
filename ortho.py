from scalar_types import *
wp.config.max_unroll = 1
wp.config.enable_backward = False

stiffness = scalar(1e9)



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

