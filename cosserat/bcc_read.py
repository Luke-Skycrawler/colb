import struct
import numpy as np
import polyscope as ps 
import os
def read_bcc_simple(filename):
    with open(filename, "rb") as f:

        header = struct.unpack("<3sB2sBBQQ40s", f.read(64))
        curve_count = header[5]
        total_cp = header[6]

        curves = []
        yarn_start = [0]
        cnt = 0
        points = np.zeros((0, 3), dtype=np.float32)
        for _ in range(curve_count):
            n = struct.unpack("<i", f.read(4))[0]
            is_loop = n < 0
            n = abs(n)

            pts = np.fromfile(f, dtype=np.float32, count=n*3).reshape(-1,3)
            curves.append((pts, is_loop))
            cnt += n
            yarn_start.append(cnt)
            points = np.vstack([points, pts])
        
    yarn_start = np.array(yarn_start, dtype=np.int32)
    return curves, points, yarn_start


if __name__ == "__main__":
    patterns = [
        "flame_ribbing_pattern", 
        "cable_work_pattern",
        "openwork_trellis_pattern"
    ]
    pattern = patterns[2]
    curves, points, yarn_start = read_bcc_simple(f"assets/yarn/{pattern}.bcc")
    for pts, is_loop in curves:
        print(f"Curve with {len(pts)} points, loop: {is_loop}")

    print(f"points min = {points.min(axis=0)}, max = {points.max(axis=0)}")
    ps.init()
    if not os.path.exists(f"assets/yarn/{pattern}/spline_points.npy"):
        os.makedirs(f"assets/yarn/{pattern}", exist_ok = True)
    np.save(f"assets/yarn/{pattern}/spline_points.npy", points / 240.0)
    np.save(f"assets/yarn/{pattern}/yarn_start.npy", yarn_start)
    e0 = np.arange(len(points)-1)
    e1 = np.arange(1, len(points))
    edges = np.column_stack([e0, e1])
    scale = 5 / 6
    yarns = ps.register_curve_network("Cable Work Pattern", points * scale, edges)
    yarns.set_radius(5e-2, relative= False)
    ps.show()
    