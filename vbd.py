import warp as wp 
import numpy as np 
from primal import PrimalRbd, XConstraint, stiffness, eps, gravity, compute_u_minus_utilde
from quat_util import scalar, vec3, vec4, mat33, mat44, Rq, Gq, Hq, RigidState, quat_mult
from xpbd_contact import fetch_b0b1,  fetch_dist_n_r0r1
from BDF1 import BDFHistory
from rbd_simple import Inertia
from geometry import Soup
from warp.sparse import bsr_from_triplets
@wp.kernel
def add_du_kernel(du: wp.array(dtype = vec3), history: wp.array(dtype = BDFHistory), alpha: scalar, dt: scalar, color: int, colors: wp.array(dtype = int)):
    i = wp.tid()
    c = colors[i]
    # n_bodies = history.shape[0]
    if c == color:
        history[i].nxt.v -= alpha * du[i * 2] 
        history[i].nxt.c = history[i].now.c + dt * history[i].nxt.v
        
        history[i].nxt.omega -= alpha * du[i * 2 + 1]
        history[i].nxt.q = history[i].now.q + scalar(0.5) * wp.transpose(Gq(history[i].nxt.q)) @ history[i].nxt.omega * dt
        history[i].nxt.q = wp.normalize(history[i].nxt.q)

# @wp.func
# def id(b0: int, b1: int, n_bodies: int) -> int: 
#     bl = wp.min(b0, b1)
#     bu = wp.max(b0, b1)
#     # +1 to prevent ambiguity with 0-init arrays 
#     return bl * n_bodies + bu + 1

# @wp.func 
# def hash(id: int) -> int:
#     return id % 8191

# @wp.kernel
# def register_constraints(constraints: wp.array(dtype = XConstraint), soup: Soup, htable: wp.array(dtype = int), n_bodies: int):
#     i = wp.tid()
#     b0, b1 = fetch_b0b1(constraints[i], soup)
#     id = id(b0, b1, n_bodies)
#     h = hash(id)

#     idx_test = int(h)
#     value = -1
#     while True:
#         value = wp.atomic_cas(htable, idx_test, id, 0)
#         if value == 0 or value == id:
#             break
#         else:
#             idx_test += 1

@wp.struct 
class CSRTriplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    vals: wp.array(dtype = int)

@wp.kernel 
def to_adjacency_csr(constraints: wp.array(dtype = XConstraint), triplets: CSRTriplets, soup: Soup):
    i = wp.tid()
    c = constraints[i]
    b0, b1 = fetch_b0b1(c, soup)
    
    if b0 != b1: 
        triplets.rows[i * 2 + 0] = b0
        triplets.cols[i * 2 + 0] = b1

        triplets.rows[i * 2 + 1] = b1
        triplets.cols[i * 2 + 1] = b0

@wp.kernel
def verify_coloring(offsets: wp.array(dtype = int), indices: wp.array(dtype = int), colors: wp.array(dtype = int), cnt_invalid: wp.array(dtype = int)):
    i = wp.tid() 
    # if colors[i] == 0:
    #     cnt_invalid[0] = 1
    
    for ii in range(offsets[i], offsets[i + 1]):
        j = indices[ii]
        if i != j and colors[i] == colors[j]:
            # cnt_invalid[0] = 1
            wp.atomic_add(cnt_invalid, 0, 1)

@wp.kernel
def gunrocks(node_values: wp.array(dtype = int), offsets: wp.array(dtype = int), indices: wp.array(dtype = int), colors_in: wp.array(dtype = int), colors_out: wp.array(dtype = int), sweep: int, cnt_uncolored: wp.array(dtype = int)):
    i = wp.tid()
    if colors_in[i] != 0:
        colors_out[i] = colors_in[i]
    else:
        local_minima = bool(True)
        local_maxima = bool(True)
        vi = node_values[i]
        for ii in range(offsets[i], offsets[i + 1]):
            j = indices[ii]

            if j != i and colors_in[j] == 0:
                vj = node_values[j]
                # Break equal-priority ties by index so every uncolored node has a strict order.
                if (vj < vi) or (vj == vi and j < i):
                    local_minima = False
                if (vj > vi) or (vj == vi and j > i):
                    local_maxima = False

        if local_minima:
            colors_out[i] = sweep * 2 + 1
        elif local_maxima:
            colors_out[i] = sweep * 2 + 2
        else:
            colors_out[i] = 0
            wp.atomic_add(cnt_uncolored, 0, 1)
    
class VBDRbd(PrimalRbd):
    def __init__(self, h, config_file):
        super().__init__(h, config_file)
        self.define_color()
        self.alpha = 1.0

        # Use a permutation to avoid duplicate priorities in coloring.
        vnp = np.random.permutation(self.n_bodies)
        self.node_values = wp.array(vnp, dtype = int)
    
    def bodywise_connectivity(self):
        triplets = CSRTriplets()
        triplets.rows = wp.zeros((self.n_contacts * 2,), dtype = int)
        triplets.cols = wp.zeros((self.n_contacts * 2,), dtype = int)
        triplets.vals = wp.ones((self.n_contacts * 2,), dtype = int)

        wp.launch(to_adjacency_csr, dim = (self.n_contacts,), inputs = [self.contacts_new.list, triplets, self.soup])
        adjacency = bsr_from_triplets(self.n_bodies, self.n_bodies, triplets.rows, triplets.cols, triplets.vals)
        
        return adjacency

    def colorization(self):
        adj = self.bodywise_connectivity()
        sweep = 0
        cnt_uncolored = wp.zeros((1,), dtype = int)
        colors_in = wp.zeros((self.n_bodies,), dtype = int)
        colors_out = wp.zeros((self.n_bodies,), dtype = int)
        while True: 
            cnt_uncolored.zero_()
            wp.launch(gunrocks, dim = (self.n_bodies,), inputs = [self.node_values, adj.offsets, adj.columns, colors_in, colors_out, sweep, cnt_uncolored])
            sweep += 1
            if cnt_uncolored.numpy()[0] == 0:
                wp.copy(self.colors, colors_out)
                break 
            colors_in, colors_out = colors_out, colors_in
        print(f"colorization done in {sweep} sweeps")

        cnt_invalid = wp.zeros((1,), dtype = int)
        wp.launch(verify_coloring, dim = (self.n_bodies,), inputs = [adj.offsets, adj.columns, self.colors, cnt_invalid])
        invalid = cnt_invalid.numpy()[0]
        if invalid: 

            # print trace 
            offsets = adj.offsets.numpy()
            indices = adj.columns.numpy()
            
            colors = self.colors.numpy()

            print(f"offsets = {offsets}")
            print(f"indices = {indices}")
            print(f"colors = {colors}")
            print(f"coloring invalid!")
            quit()
        return sweep * 2

    def define_color(self):
        angles = np.array([
            [90., 0., 90.],
            [0., 0., 90.],
            [90., 90., 0.],
        ])
        colors = []
        for obj in self.kinetic_objects:
            err = np.abs(obj.euler - angles)
            color = np.argmin(np.sum(err, axis=1))
            colors.append(color)

        colors = np.array(colors)
        self.colors = wp.array(colors, dtype = int)

    def add_du(self, alpha, color): 
        wp.launch(add_du_kernel, dim = (self.n_bodies,), inputs = [self.du, self.history, alpha, self.dt, color, self.colors])


    def step(self):
        for ss in range(10):

            with wp.ScopedTimer("step"):
                newton = True
                iter = 0
                max_iter = 2
                self.detect_collision()
                with wp.ScopedTimer("colorization"):
                    n_colors = self.colorization()
                while newton: 

                    # shuffle_colors = np.random.permutation(n_colors) + 1
                    shuffle_colors = range(1, n_colors + 1) if iter % 2 == 0 else range(n_colors, 0, -1)
                    # for color in range(1, n_colors + 1):
                    # for color in range(3):
                    for color in shuffle_colors:
                        self.compute_preconditioner()
                        self.compute_rhs()
                        
                        du_norm = self.compute_du() 
                        self.add_du(self.alpha, color)
                    iter += 1
                    print(f"    iter: {iter}, du norm: {du_norm}")
                    newton = not (du_norm < 1e-5 or iter >= max_iter)
            
                self.forward_states()
