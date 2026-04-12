import argparse
import numpy as np
import warp as wp

from scalar_types import *


eps = scalar(1e-6)

@wp.func
def tangent_space(n: vec3):
    '''
    computes two orthogonal tangent vectors given a normal vector n
    '''
    z = scalar(0.0)
    t1 = vec3(z)
    if wp.abs(n[0]) < wp.abs(n[1]) and wp.abs(n[0]) < wp.abs(n[2]):
        t1 = vec3(z, -n[2], n[1])
    elif wp.abs(n[1]) < wp.abs(n[2]):
        t1 = vec3(-n[2], z, n[0])
    else:
        t1 = vec3(-n[1], n[0], z)
    t1 = wp.normalize(t1)
    t2 = wp.cross(n, t1)
    return t1, t2

@wp.kernel
def gen_ee_test(ds: wp.array(dtype = vec3), e0s: wp.array(dtype = vec3), x: wp.array(dtype = vec3), abc: wp.array(dtype = vec3)):
    """
    inputs: d, e0 random vec3, ab random vec2 (alpha, beta in [0, 1])
    outputs: 4 vertices for each edge-edge test case, packed in x as:
        x[4*i + 0], x[4*i + 1], x[4*i + 2], x[4*i + 3]

    construction:
        p0 = -alpha * e0
        p1 = (1 - alpha) * e0
        p2 = d - beta * e1
        p3 = d + (1 - beta) * e1

    with |e0| = |e1| = |d| = 1 and e1 = normalize(cross(e0, d)).
    This makes d the shortest segment between the two lines and gives a known
    closest distance of 1.0 at (alpha, beta).
    """
    i = wp.tid()
    z = scalar(0.0)
    o = scalar(1.0)
    di = ds[i]
    if wp.length(di) < eps:
        di = vec3(z, z, o)
    di = wp.normalize(di)

    e0 = e0s[i]
    if wp.length(e0) < eps:
        e0 = vec3(o, z, z)

    # Make e0 not parallel to d (fallback picks an axis least aligned with d).
    e0 = e0 - wp.dot(e0, di) * di
    if wp.length(e0) < eps:
        axis = vec3(o, z, z)
        if wp.abs(di[0]) > scalar(0.9):
            axis = vec3(z, o, z)
        e0 = wp.cross(axis, di)
    e0 = wp.normalize(e0)

    e1x, e1y = tangent_space(di)
    theta = scalar(wp.pi * 2.0) * abc[i][1]
    e1 = wp.cos(theta) * e1x + wp.sin(theta) * e1y

    e1 = wp.normalize(e1)

    alpha = wp.clamp(abc[i][0], z, o)
    beta = wp.clamp(abc[i][1], z, o)

    base = i * 4
    x[base + 0] = -alpha * e0
    x[base + 1] = (o - alpha) * e0
    x[base + 2] = di - beta * e1
    x[base + 3] = di + (o - beta) * e1


@wp.kernel
def eval_ee_distance(x: wp.array(dtype = vec3), out_abd: wp.array(dtype = vec3)):
    i = wp.tid()
    base = i * 4

    p0 = x[base + 0]
    p1 = x[base + 1]
    p2 = x[base + 2]
    p3 = x[base + 3]

    abd = wp.closest_point_edge_edge(wp.vec3(p0), wp.vec3(p1), wp.vec3(p2), wp.vec3(p3), 1e-6)
    out_abd[i] = vec3(abd)


def generate_cases(n_cases: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    ds_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    e0s_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    ab_np = rng.random(size=(n_cases, 3)).astype(np.float64)


    ds = wp.array(ds_np, dtype=vec3)
    e0s = wp.array(e0s_np, dtype=vec3)
    ab = wp.array(ab_np, dtype=vec3)

    x = wp.zeros((n_cases * 4,), dtype=vec3)
    wp.launch(gen_ee_test, dim=n_cases, inputs=[ds, e0s, x, ab])

    out_abd = wp.zeros((n_cases,), dtype=vec3)
    wp.launch(eval_ee_distance, dim=n_cases, inputs=[x, out_abd])

    x_np = x.numpy().reshape(n_cases, 4, 3)
    abd_np = out_abd.numpy()

    return {
        "x": x_np,
        "ab_query": abd_np,
        "ab_input": ab_np,
        "distance_expected": np.ones((n_cases,), dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate edge-edge distance test cases.")
    parser.add_argument("--n", type=int, default=10000, help="Number of test cases.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out", type=str, default="", help="Optional output .npz path.")
    args = parser.parse_args()

    wp.init()
    data = generate_cases(args.n, seed=args.seed)

    abd = data["ab_query"]
    ab_in = data["ab_input"]

    alpha_err = np.abs(abd[:, 0] - ab_in[:, 0])
    beta_err = np.abs(abd[:, 1] - ab_in[:, 1])
    dist_err = np.abs(abd[:, 2] - data["distance_expected"])

    print(f"n={args.n}")
    print(
        "max errors: "
        f"alpha={alpha_err.max():.6e}, "
        f"beta={beta_err.max():.6e}, "
        f"distance={dist_err.max():.6e}"
    )

    if args.out:
        np.savez(
            args.out,
            x=data["x"],
            ab_query=data["ab_query"],
            ab_input=data["ab_input"],
            distance_expected=data["distance_expected"],
        )
        print(f"saved: {args.out}")

    
if __name__ == "__main__":
    main()
