import numpy as np 
import ipctk 

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
    
    pcpx_simple = np.kron(np.array([
        [0, 0, 1],
        [-1, 0, alpha - 1],
        [1, 0, -alpha]
    ]), np.eye(3)).T

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

    pcpx_delta[6:9, :] = np.outer(e0, dalphadx)
    
    Hu = qs @ np.diag(lambdas) @ qs.T
    hess = pcpx_simple.T @ Hu @ pcpx_simple - pcpx_delta.T @ Hu @ pcpx_delta

    A2 = pcpx_delta.T @ Hu @ pcpx_delta
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
    for i in range(n_tests):
        p = data["x"][i, 0]
        edge0 = data["x"][i, 1]
        edge1 = data["x"][i, 2]

        test(p, edge0, edge1)
    