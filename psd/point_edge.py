import numpy as np 
import ipctk 
import warp as wp
from scalar_types import *

wp.config.max_unroll = 1
wp.config.enable_backward = False

@wp.func 
def point_edge_distance(p: vec3,  edge0: vec3, edge1: vec3): 

    e0 = edge1 - edge0
    e2 = p - edge0


    alpha = wp.dot(e2, e0) / wp.dot(e0, e0)

    q = edge0 + alpha * e0
    d = wp.length(p - q)    
    return alpha, d 

@wp.func 
def point_edge_distance_gradient_hessian(p: vec3, edge0: vec3, edge1: vec3):
    z = scalar(0.0)
    o = scalar(1.0)
    alpha, d = point_edge_distance(p, edge0, edge1)
    e0 = edge1 - edge0
    e2 = p - edge0
    
    e2p = e2 - alpha * e0 
    e1p = wp.normalize(wp.cross(e0, e2p))
    
    lambdas = vec3(o) * scalar(2.0)
    
    e2p_unit = wp.normalize(e2p)
    e0_unit = wp.normalize(e0)
    
    # last 3 rows of q. The first 6 rows are zero
    qs3 = wp.matrix_from_cols(e2p_unit, e1p, e0_unit)
    
    
    # ker_pcpx_simple = mat33(
    #     z, -o, o,
    #     z, z, z,
    #     o, alpha - o, -alpha
    # )


    dadx = dalphadx(e0, e2)

    k2 = vec3(o, alpha - o, -alpha)
    grad = wp.outer(k2, e2p_unit) * scalar(2.0) * d
    # grad = wp.matrix_from_rows(e2p_unit, (alpha - o) * e2p_unit, -alpha * e2p_unit) * 2.0 * d
    # flatten it by rows to get the 9x1 gradient 

    k2k2T = wp.outer(k2, k2)
    Hu33 = qs3 @ wp.diag(lambdas) @ wp.transpose(qs3)
    
    e0THue0 = wp.dot(Hu33 @ e0, e0)
    # hess_delta = outer(dadx, dadx) * e0^T H e0
    # out9x9 = wp.outer(dadx, dadx) * e0THue0    
    out9x9 = wp.matrix(
        z, 
        shape = (9, 9),
        dtype = scalar
    )

    # hess_s = kron(k2k2T, Hu33)
    for ii in range(3):
        for jj in range(3):
            block = -wp.outer(dadx[ii], dadx[jj]) * e0THue0 + k2k2T[ii, jj] * Hu33 

            for kk in range(3):
                for ll in range(3):
                    out9x9[ii * 3 + kk, jj * 3 + ll] = block[kk, ll]
    
    return grad, out9x9
    
@wp.func 
def dalphadx(e0: vec3, e2: vec3):
    term = scalar(1.0) / wp.dot(e0, e0)
    term2 = scalar(2.0) * e0 * wp.dot(e0, e2) / wp.dot(e0, e0)
    
    return wp.matrix_from_rows(term * e0, term * (term2 - e2 - e0), term * (e2 - term2))
    
@wp.kernel
def point_edge_distance_gradient_hessian_kernel(x: wp.array2d(dtype = vec3), out_grad: wp.array(dtype = mat33), out_hess: wp.array(dtype = mat99)):
    i = wp.tid()
    p = x[i, 0]
    edge0 = x[i, 1]
    edge1 = x[i, 2]

    grad, hess = point_edge_distance_gradient_hessian(p, edge0, edge1)
    out_grad[i] = grad
    out_hess[i] = hess

def test_warp(grad, hess, p, edge0, edge1):
    ipc_ref = ipctk.point_line_distance_hessian(p, edge0, edge1)

    diff = ipc_ref - hess 
    print(f"ipc ref norm = {np.linalg.norm(ipc_ref)}, hess norm = {np.linalg.norm(hess)}, diff norm = {np.linalg.norm(diff)}")

    ipc_grad = ipctk.point_line_distance_gradient(p, edge0, edge1)
    grad_diff = ipc_grad - grad
    print(f"ipc grad norm = {np.linalg.norm(ipc_grad)}, grad norm = {np.linalg.norm(grad)}, diff norm = {np.linalg.norm(grad_diff)}")

def test(p, edge0, edge1):
    
    e0 = edge1 - edge0
    e2 = p - edge0


    alpha = np.dot(e2, e0) / np.dot(e0, e0)
    t = np.clip(alpha, 0.0, 1.0)

    q = edge0 + t * e0
    d = np.linalg.norm(p - q)



    e2p = e2 - alpha * e0
    e1p = np.cross(e0, e2p)

    e1p /= np.linalg.norm(e1p)
    
    lambdas = np.zeros((3,))
    lambdas[0] = 2
    lambdas[1] = 2
    lambdas[2] = 2

    e2p_unit = e2p / np.linalg.norm(e2p)
    e0_unit = e0 / np.linalg.norm(e0)
    
    
    qs = np.zeros((9, 3))
    z6 = np.zeros((6,))

    qs[:, 0] = np.concatenate([z6, e2p_unit])
    qs[:, 1] = np.concatenate([z6, e1p])
    qs[:, 2] = np.concatenate([z6, e0])

    for i in range(3):
        qs[:, i] /= np.linalg.norm(qs[:, i])

    ker_pcpx_simple = np.array([
        [0, 0, 1],
        [-1, 0, alpha - 1],
        [1, 0, -alpha]
    ]).T
    pcpx_simple = np.kron(ker_pcpx_simple, np.eye(3))

    gu = qs[:, 0]
    grad = pcpx_simple.T @ gu * 2.0 * d
    ref_grad = ipctk.point_line_distance_gradient(p, edge0, edge1)
    # print(f"grad = {grad}, ref_grad = {ref_grad}")
    print(f"grad diff norm = {np.linalg.norm(grad - ref_grad)}")
    
    pcpx_delta = np.zeros((9, 9))

    term = 1 / np.dot(e0, e0)
    e0de2 = np.dot(e0_unit, e2)
    term2 = 2 * e0_unit * e0de2
    dalphadx = np.concatenate([
        e0,
        term2 - e2 - e0,
        e2 - term2
    ]) * term

    Hu = qs @ np.diag(lambdas) @ qs.T
    Hu33 = Hu[6:9, 6:9]
    pcpx_delta[6:9, :] = np.outer(e0, dalphadx)
    left = np.outer(ker_pcpx_simple[2], ker_pcpx_simple[2])
    simple_hess = np.kron(left, Hu33)

    e0THue0 = e0 @ Hu33 @ e0
    delta_hess = np.outer(dalphadx, dalphadx) * e0THue0
    # hess = pcpx_simple.T @ Hu @ pcpx_simple - pcpx_delta.T @ Hu @ pcpx_delta
    hess = simple_hess - delta_hess

    A2 = delta_hess
    print(f"A2 norm = {np.linalg.norm(A2)}")
    
    ipc_ref = ipctk.point_line_distance_hessian(p, edge0, edge1)

    diff = ipc_ref - hess 
    print(f"ipc ref norm = {np.linalg.norm(ipc_ref)}, hess norm = {np.linalg.norm(hess)}, diff norm = {np.linalg.norm(diff)}")

    # print(f"ipc ref = \n{ipc_ref}\n\nhess = \n{hess}\ndiff = \n{diff}")

if __name__ == "__main__":
    # x = np.load("pe.npz")

    # p = np.array([0, 1, 0], dtype=float)
    # edge0 = np.array([-1, 0, 0], dtype=float)
    # edge1 = np.array([1, 0, 0], dtype=float)

    data = np.load("pe.npz")
    n_tests = 10

    xnp = data["x"][:n_tests]
    
    x = wp.zeros((n_tests, 3), dtype = vec3)
    x.assign(xnp)
    grad = wp.zeros((n_tests, ), dtype = mat33)
    hess = wp.zeros((n_tests, ), dtype = mat99)

    wp.launch(point_edge_distance_gradient_hessian_kernel, dim = (n_tests, ), inputs = [x, grad, hess])    
    

    gradnp = grad.numpy()
    hessnp = hess.numpy()
    for i in range(n_tests):
        p = xnp[i, 0]
        edge0 = xnp[i, 1]
        edge1 = xnp[i, 2]

        # test(p, edge0, edge1)
        test_warp(gradnp[i].reshape(-1), hessnp[i], p, edge0, edge1)
