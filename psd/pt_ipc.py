import warp as wp 
from scalar_types import *
import numpy as np
from psd.hl import signed_distance, eig_Hl_tid, gl

from psd.pt import C_vf, dcvfdx_s
from psd.barrier import *
from psd.assemble import assemble
import ipctk
from psd.ee_ipc import extract_Q, QLQinv, dcTHldc
wp.config.max_unroll = 1
wp.config.enable_backward = False
ipctk_ref = True
def ipc_term_pt():
    npt = 10
    xnp = np.load("pt.npz")["x"][:npt]
    # npt = xnp.shape[0]

    Q = wp.zeros((npt, 5 * 3), dtype = vec3)
    Lam = wp.zeros((npt, 5), dtype = scalar)
    dcdx = wp.zeros((npt, ), dtype = mat34)
    Jt = wp.zeros((npt, ), dtype = mat34)
    Jp = wp.zeros((npt, ), dtype = vec4) 
    x = wp.zeros((npt, 4), dtype = vec3)
    x.assign(xnp)
    H = wp.zeros((npt, ), dtype = mat12)

    wp.launch(_Q_lambda_pt, dim = (npt, ), inputs = [x, Q, Lam])
    wp.launch(_dcdx, dim = (npt, ), inputs = [x, dcdx])
    wp.launch(extract_JpJt, dim = (npt,), inputs=[x, Jp, Jt])
    wp.launch(assemble, dim = (npt,), inputs=[dcdx, Q, Lam, H])

    Qnp = Q.numpy()
    Jpnp = Jp.numpy()
    Jtnp = Jt.numpy()
    Lamnp = Lam.numpy()
    dcdxnp = dcdx.numpy()
    xnp = x.numpy()
    Hnp = H.numpy()
    
            
    for i in range(npt):
        with wp.ScopedTimer(f"pt contact {i}"):

            pt_grad, g = extract_g(Qnp[i], dcdxnp[i], Jpnp[i], Jtnp[i])
            
            qq = extract_Q(Qnp[i])
            Hl_pos = QLQinv(qq, Lamnp[i])
            Hpt = dcTHldc(dcdxnp[i], Hl_pos)

            Hpt = Hnp[i]

            if ipctk_ref:
                p = xnp[i, 0]
                t0 = xnp[i, 1]
                t1 = xnp[i, 2]
                t2 = xnp[i, 3]

                gpt_ipc = ipctk.point_plane_distance_gradient(p, t0, t1, t2)
                Hpt_ipc = ipctk.point_plane_distance_hessian(p, t0, t1, t2)
            
                # print(f"pt_grad = {pt_grad}\nref = {gpt_ipc}\ndiff = {np.linalg.norm(pt_grad - gpt_ipc)}")

                print(f"H = {np.linalg.norm(Hpt)}\nref = {np.linalg.norm(Hpt_ipc)}, diff = {np.linalg.norm(Hpt - Hpt_ipc)}")




def extract_g(Q, dcdx, Jp, Jt):
    gl = np.zeros(9)
    for i in range(3):
        gl[3 * i: 3 * i + 3] = Q[4 * 3 + i]
    g12 = np.kron(dcdx.T, np.eye(3)) @ gl
    g = JTg(Jp, Jt, g12)
    return g12, g

def JTg(Jp, Jt, g12):
    Jp = Jp.reshape(1, 4)
    gp = np.kron(Jp.T, np.eye(3)) @ g12[:3]
    gt = np.kron(Jt.T, np.eye(3)) @ g12[3:]
    g = np.zeros(24)
    g[:12] = gp
    g[12:] = gt
    return g


    
@wp.kernel
def extract_JpJt(x: wp.array2d(dtype = vec3), Jp: wp.array(dtype = vec4), Jt: wp.array(dtype = mat34)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]
    Jp[i] = vec4(scalar(1.0), x0[0], x0[1], x0[2])
    Jt[i] = mat34(scalar(1.0), x1[0], x1[1], x1[2], 
               scalar(1.0), x2[0], x2[1], x2[2], 
               scalar(1.0), x3[0], x3[1], x3[2])
    

@wp.kernel
def _Q_lambda_pt(x: wp.array2d(dtype = vec3), q: wp.array2d(dtype = vec3), lam: wp.array2d(dtype = scalar)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]
    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
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

    for ii in range(5):         
        sum = scalar(0.0)
        for jj in range(3):
            sum += wp.length_sq(q[i, ii * 3 + jj])
        sum = wp.sqrt(sum)
        for jj in range(3):
            q[i, ii * 3 + jj] /= sum        

@wp.kernel
def _dcdx(x: wp.array2d(dtype = vec3), ret: wp.array(dtype = mat34)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]
    dcdx_simple = dcvfdx_s(x0, x1, x2, x3)
    ret[i] = dcdx_simple

if __name__ == "__main__":
    wp.init()
    ipc_term_pt()   