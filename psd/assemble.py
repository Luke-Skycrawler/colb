import warp as wp 
from scalar_types import *

wp.config.max_unroll = 1
wp.config.enable_backward = False

@wp.kernel
def assemble(dcdx: wp.array(dtype = mat34), q: wp.array2d(dtype = vec3), lam: wp.array2d(dtype = scalar), out: wp.array(dtype = mat12)):
    i = wp.tid()
    o = scalar(1.0)
    dcdxi = dcdx[i]
    # u = qT @ dcdx
    u = wp.zeros((5, 4), dtype = vec3)

    for ii in range(5): 
        for kk in range(3): 
            for jj in range(4):
                u[ii, jj] += q[i, ii * 3 + kk] * dcdxi[kk, jj]
    
    # u^T @ L @ u
    tmp = wp.zeros((4, 4), dtype = mat33)
    for ii in range(5): 
        for jj in range(4):
            for kk in range(4):
                tmp[jj, kk] += lam[i, ii] * wp.outer(u[ii,   jj], u[ii, kk])

    for ii in range(4):
        for jj in range(4):
            for mm in range(3):
                for nn in range(3):
                    out[i][ii * 3 + mm, jj * 3 + nn] = tmp[ii, jj][mm][nn]


    