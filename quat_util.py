import numpy as np 
import warp as wp 

mat33 = wp.mat33d
mat44 = wp.mat44d
vec4 = wp.vec4d
vec3 = wp.vec3d
scalar = wp.float64

@wp.func 
def Gq(q: vec4):
    x, y, z, w = q.x, q.y, q.z, q.w
    return wp.matrix(
        w, -z, y, -x,
        z, w, -x, -y,
        -y, x, w, -z,
        shape = (3, 4),
        dtype = scalar
    )

@wp.func 
def Hq(q: vec4):
    x, y, z, w = q.x, q.y, q.z, q.w
    return wp.matrix(
        w, z, -y, -x,
        -z, w, x, -y,
        y, -x, w, -z,
        shape = (3, 4),
        dtype = scalar 
    )

@wp.func
def Rq(q: vec4) -> mat33: 
    return Gq(q) @ wp.transpose(Hq(q))
    
@wp.struct
class RigidState: 
    c: vec3
    q: vec4    
    v: vec3
    w: vec4
    omega: vec3


@wp.func
def dGdqx(): 
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, z, z, -o, 
        z, z, -o, z,
        z, o, z, z,
        shape = (3, 4),
        dtype = scalar
    )

@wp.func 
def dGdqy():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, z, o, z,
        z, z, z, -o,
        -o, z, z, z,
        shape = (3, 4),
        dtype = scalar
    )

@wp.func 
def dGdqz():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, -o, z, z, 
        o, z, z, z,
        z, z, z, -o,
        shape = (3, 4),
        dtype = scalar
    )

@wp.func 
def dGdqw():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        o, z, z, z, 
        z, o, z, z,
        z, z, o, z,
        shape = (3, 4),
        dtype = scalar
    )


@wp.kernel
def _test_kernel(q: wp.array(dtype = vec4), diff: wp.array(dtype = scalar)):
    i = wp.tid()
    qi = q[i]
    G = Gq(qi)

    gx = dGdqx() * qi.x
    gy = dGdqy() * qi.y
    gz = dGdqz() * qi.z
    gw = dGdqw() * qi.w

    dg = gx + gy + gz + gw - G
    
    diff[i] = wp.trace(dg @ wp.transpose(dg))
    

@wp.func 
def dHdqx():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, z, z, -o, 
        z, z, o, z,
        z, -o, z, z,
        shape = (3, 4), dtype = scalar
    )

@wp.func
def dHdqy():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, z, -o, z,
        z, z, z, -o,
        o, z, z, z,
        shape = (3, 4), dtype = scalar
    )

@wp.func 
def dHdqz():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        z, o, z, z, 
        -o, z, z, z,
        z, z, z, -o,
        shape = (3, 4), dtype = scalar
    )

@wp.func
def dHdqw():
    z = scalar(0.)
    o = scalar(1.)
    return wp.matrix(
        o, z, z, z, 
        z, o, z, z,
        z, z, o, z,
        shape = (3, 4), dtype = scalar
    )

@wp.kernel
def _test_H_kernel(q: wp.array(dtype = vec4), diff: wp.array(dtype = scalar)):
    i = wp.tid()
    qi = q[i]
    H = Hq(qi)

    hx = dHdqx() * qi.x
    hy = dHdqy() * qi.y
    hz = dHdqz() * qi.z
    hw = dHdqw() * qi.w

    dh = hx + hy + hz + hw - H

    diff[i] = wp.trace(dh @ wp.transpose(dh))

@wp.func 
def quat_mult(q1: vec4, q2: vec4) -> vec4:
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w
    return vec4(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )

if __name__ == "__main__":
    wp.init()
    n_samples = 100
    q = wp.zeros(n_samples, dtype = vec4)
    q.assign(np.random.rand(n_samples, 4))
    diff = wp.zeros(n_samples, dtype = scalar)
    diffh = wp.zeros(n_samples, dtype = scalar)
    wp.launch(_test_kernel, n_samples, inputs = [q, diff])
    dnp = diff.numpy()
    print("max diff: ", np.max(np.abs(dnp)))
    print("q = ", q.numpy())

    wp.launch(_test_H_kernel, n_samples, inputs = [q, diffh])
    dhnp = diffh.numpy()
    print("H max diff: ", np.max(np.abs(dhnp)))
