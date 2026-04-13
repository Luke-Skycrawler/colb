import numpy as np 
import ipctk
def molli(a, b):
    c = np.cross(a, b)
    p = np.dot(c, c)
    return p
def test(a, b):
    
    p = molli(a, b)

    dpda = -2.0 * np.cross(b, np.cross(b, a))
    dpdb = -2.0 * np.cross(a, np.cross(a, b))

    h = 1e-3

    da = np.random.rand(3) * h
    db = np.random.rand(3) * h

    dp_fd = (molli(a + da, b + db) - molli(a - da, b - db)) / 2

    dp_pred = np.dot(dpda, da) + np.dot(dpdb, db)

    diff = np.abs(dp_fd - dp_pred)

    print(f"dp_fd = {dp_fd}, dp_pred = {dp_pred}")
    print(f"diff = {diff}")

    d2pda2 = -2.0 * (np.outer(b, b) - np.dot(b, b) * np.eye(3))
    d2pdb2 = -2.0 * (np.outer(a, a) - np.dot(a, a) * np.eye(3))
    d2pdadb = -2.0 * (np.outer(b, a) - 2 * np.outer(a, b) + np.eye(3) * np.dot(a, b))

    dp_disturb = molli(a + da, b + db) + molli(a - da, b - db) - 2 * p

    dx = np.concatenate([da, db])
    d2p = np.block([[d2pda2, d2pdadb],
                    [d2pdadb.T, d2pdb2]])

    dpdx = np.concatenate([dpda, dpdb])

    
    # dp_pred = p + np.dot(dpdx, dx) + 0.5 * np.dot(dx, np.dot(d2p, dx))
    dp_pred = np.dot(np.dot(d2p, dx), dx)

    print(f"dp_disturb = {dp_disturb}, dp_pred = {dp_pred}")

    z3 = np.zeros((3,))
    ref = ipctk.edge_edge_cross_squarednorm_hessian(z3, a, z3, b)
    
    ref = np.block([[
        ref[3: 6, 3:6], ref[3:6, 9:12]],
        [ref[9:12, 3:6], ref[9:12, 9:12]
    ]])
    print(f"ref = {ref}\nd2p = {d2p}\ndiff = {np.linalg.norm(ref - d2p)}")
    return d2p


def mollifier_hessian(a, b):
    d2pda2 = -2.0 * (np.outer(b, b) - np.dot(b, b) * np.eye(3))
    d2pdb2 = -2.0 * (np.outer(a, a) - np.dot(a, a) * np.eye(3))
    d2pdadb = -2.0 * (np.outer(b, a) - 2 * np.outer(a, b) + np.eye(3) * np.dot(a, b))


    d2p = np.block([[d2pda2, d2pdadb],
                    [d2pdadb.T, d2pdb2]])
    return d2p

def eigs_decompose():
    a = np.array([3, 0, 0], dtype = float)
    b = np.array([0, 2, 0], dtype = float)
    d2p = mollifier_hessian(a, b)
    eigvals, eigvecs = np.linalg.eig(d2p)

    '''
    lam 0 = a^2
    lam 1 = b^2
    lam 2, 3 = +- ab
    lam 4, 5 = 0.5 * ((a^2 + b^2) +- sqrt((a^2 + b^2)^2 + 12 * a^2 * b^2))
    '''
    
    print(eigvals, eigvecs)

def eigs_analytical(e0, e1): 
    a = e0 
    alpha = np.dot(e0, e1) / np.dot(e0, e0)
    b = e1 - alpha * e0
    
    n = np.cross(a, b)
    n /= np.sqrt(np.dot(n, n))
    
    aa = np.sqrt(np.dot(a, a))
    bb = np.sqrt(np.dot(b, b))
    
    lambdas = np.zeros((6,))
    lambdas[0] = bb * bb
    lambdas[1] = aa * aa
    
    lambdas[2] = aa * bb
    lambdas[3] = -aa * bb

    term = aa* aa + bb * bb
    delta = term * term + 12 * aa * aa * bb * bb
    lambdas[4] = 0.5 * (term + np.sqrt(delta))
    lambdas[5] = 0.5 * (term - np.sqrt(delta))


    qs = np.zeros((6, 6))
    z3 = np.zeros((3,))

    qs[:, 0] = np.concatenate([n, z3])
    qs[:, 1] = np.concatenate([z3, n])
    qs[:, 2] = np.concatenate([-b / bb, a / aa])
    qs[:, 3] = np.concatenate([b / bb, a / aa])

    for i in range(4, 6):
        qs[:, i] = np.concatenate([2 * bb * a, (lambdas[i] / bb - bb) * b])
    
    for i in range(6): 
        qs[:, i] /= np.linalg.norm(qs[:, i])
    lambdas *= 2.0

    d2p = mollifier_hessian(a, b)
    eigvals, eigvecs = np.linalg.eig(d2p)

    diff_eigsys = qs @ np.diag(lambdas) @ qs.T - d2p
    diff_eigsys_norm = np.linalg.norm(diff_eigsys)

    eigvals_sort = np.sort(eigvals)
    lambdas_sort = np.sort(lambdas)

    diff_vals = np.linalg.norm(eigvals_sort - lambdas_sort)

    debug = False
    if debug: 

        # sort q columns according to the order of lambdas
        idx = np.argsort(lambdas)
        qs = qs[:, idx]
        
        # sort eigvecs according to the order of eigvals
        idx_eig = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx_eig]

        eigvals = eigvals_sort
        lambdas = lambdas_sort



        print(f"analytical lambdas = \n{lambdas}, \neigvals = \n\n{qs}\n\n\n\n")
        print(f"numerical eigen values = \n{eigvals}, \neigvecs = \n\n{eigvecs}")

    
    print(f"analytical lambdas = \n{np.sort(lambdas)}\neigvals = \n{np.sort(eigvals)}, \ndiff eig vals = {diff_vals}, \ndiff_eigsys = {diff_eigsys_norm}")


def test_full():
    a = np.array([3, 0, 0], dtype = float)
    b = np.array([1, 2, 0], dtype = float)

    d2p = mollifier_hessian(a, b)

    I3 = np.eye(3)
    z3 = np.zeros((3, 3))
    alpha = np.dot(a, b) / np.dot(a, a)
    dcdx = np.block([[I3, -alpha * I3],
                    [z3, I3]]).T
    
    d2p_perp = mollifier_hessian(a, b - alpha * a)
    d2p_proj = dcdx.T @ d2p_perp @ dcdx
    
    print(f"d2p = {np.linalg.norm(d2p)}\nd2p_proj = {np.linalg.norm(d2p_proj)}\ndiff = {np.linalg.norm(d2p - d2p_proj)}")

    # A1 = dcdx_delta^T H dcdx_delta is indeed zero 


if __name__ == "__main__":
    a = np.random.rand(3)
    b = np.random.rand(3)

    # a = np.array([2, 0, 0], dtype = float)
    # b = np.array([0, 1, 0], dtype = float)
    # test(a, b)

    # eigs_decompose()
    # test_full()
    eigs_analytical(a, b)
    # x = test(a, b)
    # y = test(a, b + a)
    # print(f"x y diff = {np.linalg.norm(x - y)}")

