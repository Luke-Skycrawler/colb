import argparse
import numpy as np
import warp as wp

from scalar_types import *


eps = scalar(1e-6)


@wp.kernel
def gen_pt_test(
    e0s: wp.array(dtype=vec3),
    e1s: wp.array(dtype=vec3),
    abd: wp.array(dtype=vec3),
    x: wp.array(dtype=vec3),
):
    """
    inputs: random e0, e1 (vec3) and alpha, beta, dist in [0, 1]
    outputs: 4 vertices for each point-triangle test case, packed in x as:
        x[4*i + 0], x[4*i + 1], x[4*i + 2], x[4*i + 3]

    construction:
        t2 = 0
        t0 = e0
        t1 = e1
        q  = alpha * e0 + beta * e1
        p  = q + dist * n

    where e0 and e1 are orthonormalized and n = normalize(cross(e0, e1)).
    alpha and beta are remapped to guarantee alpha >= 0, beta >= 0,
    alpha + beta <= 1 (interior triangle projection).
    """
    i = wp.tid()

    z = scalar(0.0)
    o = scalar(1.0)

    e0 = e0s[i]
    if wp.length(e0) < eps:
        e0 = vec3(o, z, z)
    e0 = wp.normalize(e0)

    e1 = e1s[i]
    if wp.length(e1) < eps:
        e1 = vec3(z, o, z)

    # Remove component along e0 so e1 is not parallel to e0.
    e1 = e1 - wp.dot(e1, e0) * e0
    if wp.length(e1) < eps:
        axis = vec3(z, z, o)
        if wp.abs(e0[2]) > scalar(0.9):
            axis = vec3(z, o, z)
        e1 = wp.cross(axis, e0)
    e1 = wp.normalize(e1)

    n = wp.cross(e0, e1)
    if wp.length(n) < eps:
        n = vec3(z, z, o)
    n = wp.normalize(n)

    alpha = wp.clamp(abd[i][0], z, o)
    beta = wp.clamp(abd[i][1], z, o)

    # Mirror to keep the projected point inside the triangle.
    if alpha + beta > o:
        alpha = o - alpha
        beta = o - beta

    dist = wp.max(abd[i][2], scalar(1e-5))

    q = alpha * e0 + beta * e1
    p = q + dist * n

    base = i * 4
    x[base + 0] = p
    x[base + 1] = e0
    x[base + 2] = vec3(z)
    x[base + 3] = e1


@wp.kernel
def eval_pt_distance(x: wp.array(dtype=vec3), out_abd: wp.array(dtype=vec3)):
    i = wp.tid()
    base = i * 4

    p = x[base + 0]
    t0 = x[base + 1]
    t2 = x[base + 2]
    t1 = x[base + 3]

    e0 = t0 - t2
    e1 = t1 - t2
    e2 = p - t2

    e0e0 = wp.dot(e0, e0)
    e0e1 = wp.dot(e0, e1)
    e1e1 = wp.dot(e1, e1)

    A = mat22(e0e0, e0e1, e0e1, e1e1)
    b = vec2(wp.dot(e0, e2), wp.dot(e1, e2))
    ab = wp.inverse(A) @ b

    proj = t2 + ab[0] * e0 + ab[1] * e1
    d = wp.length(p - proj)

    out_abd[i] = vec3(ab[0], ab[1], d)


def generate_cases(n_cases: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    e0s_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    e1s_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    abd_np = rng.random(size=(n_cases, 3)).astype(np.float64)

    # Keep inputs synchronized with the in-kernel interior remap.
    a = abd_np[:, 0]
    b = abd_np[:, 1]
    mask = (a + b) > 1.0
    abd_np[mask, 0] = 1.0 - a[mask]
    abd_np[mask, 1] = 1.0 - b[mask]
    abd_np[:, 2] = np.maximum(abd_np[:, 2], 1e-5)

    e0s = wp.array(e0s_np, dtype=vec3)
    e1s = wp.array(e1s_np, dtype=vec3)
    abd = wp.array(abd_np, dtype=vec3)

    x = wp.zeros((n_cases * 4,), dtype=vec3)
    wp.launch(gen_pt_test, dim=n_cases, inputs=[e0s, e1s, abd, x])

    out_abd = wp.zeros((n_cases,), dtype=vec3)
    wp.launch(eval_pt_distance, dim=n_cases, inputs=[x, out_abd])

    x_np = x.numpy().reshape(n_cases, 4, 3)
    abd_query_np = out_abd.numpy()

    return {
        "x": x_np,
        "abd_query": abd_query_np,
        "abd_input": abd_np,
        "distance_expected": abd_np[:, 2].copy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate point-triangle distance test cases.")
    parser.add_argument("--n", type=int, default=10000, help="Number of test cases.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out", type=str, default="", help="Optional output .npz path.")
    args = parser.parse_args()

    wp.init()
    data = generate_cases(args.n, seed=args.seed)

    abd = data["abd_query"]
    abd_in = data["abd_input"]

    alpha_err = np.abs(abd[:, 0] - abd_in[:, 0])
    beta_err = np.abs(abd[:, 1] - abd_in[:, 1])
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
            abd_query=data["abd_query"],
            abd_input=data["abd_input"],
            distance_expected=data["distance_expected"],
        )
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()