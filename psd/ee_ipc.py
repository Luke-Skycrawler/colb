import warp as wp 
from scalar_types import *
import numpy as np
from psd.hl import signed_distance, eig_Hl_tid, gl

from psd.ee import C_ee, dceedx_s
from psd.barrier import *
import ipctk
ipctk_ref = True


@wp.kernel
def _Q_lambda_ee(x: wp.array2d(dtype = vec3), q: wp.array2d(dtype = vec3), lam: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]

    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    lam0, lam1, lam2, lam3 = eig_Hl_tid(e0p, e1p, e2p, q, i)
    l = signed_distance(e0p, e1p, e2p)

    # lam0 = wp.max(lam0, scalar(0.0))
    # lam1 = wp.max(lam1, scalar(0.0))
    # lam2 = wp.max(lam2, scalar(0.0))
    # lam3 = wp.max(lam3, scalar(0.0))

    lam[i, 0] = lam0
    lam[i, 1] = lam1
    lam[i, 2] = lam2
    lam[i, 3] = lam3
    lam[i, 4] = scalar(2.0)

    gl0, gl1, gl2 = gl(l, e2p)
    q[i, 4 * 3 + 0] = gl0 * scalar(2.0) * l
    q[i, 4 * 3 + 1] = gl1 * scalar(2.0) * l
    q[i, 4 * 3 + 2] = gl2 * scalar(2.0) * l


def ipc_term_ee():
    xnp = np.load("ee.npz")["x"]
    print(f"x shape = {xnp.shape}")
    nee = xnp.shape[0]


    Q = wp.zeros((nee, 5 * 3), dtype = vec3)
    Lam = wp.zeros((nee, 5), dtype = scalar)
    dcdx = wp.zeros((nee, ), dtype = mat34)
    Jei = wp.zeros((nee, ), dtype = mat24)
    Jej = wp.zeros((nee, ), dtype = mat24)
    x = wp.zeros((nee, 4), dtype = vec3)
    x.assign(xnp)


    wp.launch(_Q_lambda_ee, dim = (nee, ), inputs = [x, Q, Lam])
    
    wp.launch(_dcdx, dim = (nee, ), inputs = [x, dcdx])

    wp.launch(extract_JeiJej, dim = (nee, ), inputs = [x, Jei, Jej])
    

    Qnp = Q.numpy()
    Jeinp = Jei.numpy()
    Jejnp = Jej.numpy()
    Lamnp = Lam.numpy()
    dcdxnp = dcdx.numpy()
    xnp = x.numpy()

    gnp = np.zeros((nee, 24))

    Hinp = np.zeros((nee, 12, 12))  
    Hjnp = np.zeros((nee, 12, 12))
    Hijnp = np.zeros((nee, 12, 12))
    
    err_grad = []
    err_hess = []
    ipc_hess_norm = []
    for i in range(10):
        with wp.ScopedTimer(f"ee contact {i}"):
            # B_ = barrier_derivative_np(d2np[i])
            # B__ = barrier_derivative2_np(d2np[i])
            ee_grad, g = extract_g(Qnp[i], dcdxnp[i], Jeinp[i], Jejnp[i])
            qq = extract_Q(Qnp[i])
            Hl = QLQinv(qq, Lamnp[i])
            Hee = dcTHldc(dcdxnp[i], Hl)

            if ipctk_ref:
                ei0 = xnp[i, 0]
                ei1 = xnp[i, 1]
                ej0 = xnp[i, 2]
                ej1 = xnp[i, 3]

                # gee_ipc = ipctk.line_line_distance_gradient(ei0, ei1, ej0, ej1)

                # Hee_ipc = ipctk.line_line_distance_hessian(ei0, ei1, ej0, ej1)
                # d2_ipc = ipctk.line_line_distance(ei0, ei1, ej0, ej1)

                gee_ipc = ipctk.edge_edge_distance_gradient(ei0, ei1, ej0, ej1)

                Hee_ipc = ipctk.edge_edge_distance_hessian(ei0, ei1, ej0, ej1)

                d2_ipc = ipctk.edge_edge_distance(ei0, ei1, ej0, ej1)
                
                # print(f"ee_grad = {ee_grad}\nref = {gee_ipc}\ndiff = {(ee_grad - gee_ipc)}")
                print(f"H = {np.linalg.norm(Hee)}\nref = {np.linalg.norm(Hee_ipc)}, diff = {np.linalg.norm(Hee - Hee_ipc)}")


                B_ = barrier_derivative_np(d2_ipc)
                B__ = barrier_derivative2_np(d2_ipc)

                err_grad.append(np.linalg.norm(ee_grad - gee_ipc))
                err_hess.append(np.linalg.norm(Hee - Hee_ipc))
                ipc_hess_norm.append(np.linalg.norm(Hee_ipc))

                Hee = Hee_ipc
                ee_grad = gee_ipc
                        
                        
                        
            # Hipc = Hee * B_ + np.outer(ee_grad, ee_grad) * B__ 

            # g = JTg(Jeinp[i], Jejnp[i], ee_grad)
            # g *= B_
            # # H12 tested
            # Hi, Hj, Hij = JTH12J(Hipc, Jeinp[i], Jejnp[i])

            # gnp[i] = g
            # Hinp[i] = Hi
            # Hjnp[i] = Hj
            # Hijnp[i] = Hij
    err_grad = np.array(err_grad)
    err_hess = np.array(err_hess)
    ipc_hess_norm = np.array(ipc_hess_norm)
    print(f"max grad err = {err_grad.max():.6e}, mean grad err = {err_grad.mean():.6e}")
    print(f"max hess err = {err_hess.max():.6e}, mean hess err = {err_hess.mean():.6e}")
    print(f"max ipc hess norm = {ipc_hess_norm.max():.6e}, mean ipc hess norm = {ipc_hess_norm.mean():.6e}")
  

def QLQinv(Q, lam):
    QTQ = Q.T @ Q
    diag_inv = np.array([(scalar(1.0) / QTQ[i, i]) for i in range(5)])
    diag_inv = np.diag(diag_inv)
    Q_inv = diag_inv @ Q.T
    Lam = np.diag(lam)
    return Q @ Lam @ Q_inv

def dcTHldc(_dcdx, Hl_pos):
    dcdx = np.kron(_dcdx, np.eye(3))
    return dcdx.T @ Hl_pos @ dcdx

def extract_Q(q):
    Q = q.reshape(5, 9).T
    return Q
 
def extract_g(Q, dcdx, Jei, Jej):
    gl = np.zeros(9)
    for i in range(3):
        gl[3 * i: 3 * i + 3] = Q[4 * 3 + i]
    g12 = np.kron(dcdx.T, np.eye(3)) @ gl
    g = JTg(Jei, Jej, g12)
    return g12, g

def JTg(Jei, Jej, g12):
    gei = np.kron(Jei.T, np.eye(3)) @ g12[:6]
    gej = np.kron(Jej.T, np.eye(3)) @ g12[6:]
    g = np.zeros(24)
    g[:12] = gei
    g[12:] = gej
    return g

def JTH12J(H12, Jei, Jej):
    Jei = np.kron(Jei, np.eye(3))
    Jej = np.kron(Jej, np.eye(3))
    Hpp = H12[:6, :6]
    Hpt = H12[:6, 6:]
    Htt = H12[6:, 6:]
    Hi = Jei.T @ Hpp @ Jei
    Hj = Jej.T @ Htt @ Jej
    Hij = Jei.T @ Hpt @ Jej
    return Hi, Hj, Hij



@wp.kernel
def extract_JeiJej(x: wp.array2d(dtype = vec3), Jei: wp.array(dtype = mat24), Jej: wp.array(dtype = mat24)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]
    Jei[i] = mat24(
        scalar(1.0), x0[0], x0[1], x0[2],
        scalar(1.0), x1[0], x1[1], x1[2]
    )
    Jej[i] = mat24(
        scalar(1.0), x2[0], x2[1], x2[2], 
        scalar(1.0), x3[0], x3[1], x3[2]
    )

@wp.kernel
def _dcdx(x: wp.array2d(dtype = vec3), ret: wp.array(dtype = mat34)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]
    dcdx_simple = dceedx_s(x0, x1, x2, x3)
    ret[i] = dcdx_simple

if __name__ == "__main__":
    ipc_term_ee()   