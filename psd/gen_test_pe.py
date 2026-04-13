import argparse
import numpy as np
import warp as wp

from scalar_types import *


eps = scalar(1e-6)


@wp.kernel
def gen_pe_test(
    ds: wp.array(dtype=vec3),
    e0s: wp.array(dtype=vec3),
    ab: wp.array(dtype=scalar),
    x: wp.array(dtype=vec3),
):
    """
    inputs: d, e0 random vec3 and alpha in [0, 1]
    outputs: 3 vertices for each point-edge test case, packed in x as:
        x[3*i + 0], x[3*i + 1], x[3*i + 2]

    construction:
        p = d
        e0 = -alpha * e
        e1 = (1 - alpha) * e

    with |e| = |d| = 1 and dot(d, e) = 0.
    This gives known closest-point parameter alpha and distance 1.0.
    """
    i = wp.tid()

    z = scalar(0.0)
    o = scalar(1.0)

    di = ds[i]
    if wp.length(di) < eps:
        di = vec3(z, z, o)
    di = wp.normalize(di)

    ei = e0s[i]
    if wp.length(ei) < eps:
        ei = vec3(o, z, z)

    # Remove the component parallel to d so that d is perpendicular to edge direction.
    ei = ei - wp.dot(ei, di) * di
    if wp.length(ei) < eps:
        axis = vec3(o, z, z)
        if wp.abs(di[0]) > scalar(0.9):
            axis = vec3(z, o, z)
        ei = wp.cross(axis, di)
    ei = wp.normalize(ei)

    alpha = wp.clamp(ab[i], z, o)

    base = i * 3
    x[base + 0] = di
    x[base + 1] = -alpha * ei
    x[base + 2] = (o - alpha) * ei


@wp.kernel
def eval_pe_distance(x: wp.array(dtype=vec3), out_ad: wp.array(dtype=vec2)):
    i = wp.tid()
    base = i * 3

    p = x[base + 0]
    e0 = x[base + 1]
    e1 = x[base + 2]

    edge = e1 - e0
    t = wp.dot(p - e0, edge) / wp.dot(edge, edge)
    t = wp.clamp(t, scalar(0.0), scalar(1.0))

    q = e0 + t * edge
    d = wp.length(p - q)

    out_ad[i] = vec2(t, d)


def generate_cases(n_cases: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    ds_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    e0s_np = rng.normal(size=(n_cases, 3)).astype(np.float64)
    ab_np = rng.random(size=(n_cases, )).astype(np.float64)

    ds = wp.array(ds_np, dtype=vec3)
    e0s = wp.array(e0s_np, dtype=vec3)
    ab = wp.array(ab_np, dtype=scalar)

    x = wp.zeros((n_cases * 3,), dtype=vec3)
    wp.launch(gen_pe_test, dim=n_cases, inputs=[ds, e0s, ab, x])

    out_ad = wp.zeros((n_cases,), dtype=vec2)
    wp.launch(eval_pe_distance, dim=n_cases, inputs=[x, out_ad])

    x_np = x.numpy().reshape(n_cases, 3, 3)
    ad_np = out_ad.numpy()

    return {
        "x": x_np,
        "ad_query": ad_np,
        "a_input": ab_np,
        "distance_expected": np.ones((n_cases,), dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate point-edge distance test cases.")
    parser.add_argument("--n", type=int, default=10000, help="Number of test cases.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out", type=str, default="", help="Optional output .npz path.")
    args = parser.parse_args()

    wp.init()
    data = generate_cases(args.n, seed=args.seed)

    ad = data["ad_query"]
    a_in = data["a_input"]

    alpha_err = np.abs(ad[:, 0] - a_in)
    dist_err = np.abs(ad[:, 1] - data["distance_expected"])

    print(f"n={args.n}")
    print(
        "max errors: "
        f"alpha={alpha_err.max():.6e}, "
        f"distance={dist_err.max():.6e}"
    )

    if args.out:
        np.savez(
            args.out,
            x=data["x"],
            ad_query=data["ad_query"],
            a_input=data["a_input"],
            distance_expected=data["distance_expected"],
        )
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()