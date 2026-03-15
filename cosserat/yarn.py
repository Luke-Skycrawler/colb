import warp as wp 
from geometry import SimComplexBase
import numpy as np 
from .cosserat import StableCosserat, Node, Seg, from_to, e3
from scalar_types import *
from .spring import init_rest_frame
import os
from viewer import PSViewer
import polyscope as ps
from .rod_contact import PrimalRod

eps = scalar(1e-6)
z = scalar(0.0)
o = scalar(1.0)

class YarnGeometryComplex(SimComplexBase):
    def __init__(self, folder = "assets/yarn"): 
        '''
        reads from yarn_start.npy, is_closed.npy (if exists), spline_points.npy, and is_pinned.npy(if exists)
        ''' 

        self.read_yarns(folder)
        self.seg_nxt = np.zeros((0, ), dtype = int)
        SimComplexBase.__init__(self)
        
    def read_yarns(self, folder):
        print(f"reading folder {folder} for yarn geometry")
        self.spline_points = np.load(f"{folder}/spline_points.npy")
        self.yarn_start = np.load(f"{folder}/yarn_start.npy")
        upsample = len(self.spline_points) // self.yarn_start[-1]
        # self.yarn_start *= upsample
        self.spline_points = self.spline_points[np.arange(0, self.spline_points.shape[0], upsample)]
        assert len(self.spline_points) == self.yarn_start[-1] 
        if os.path.exists(f"{folder}/is_closed.npy"):
            self.is_closed = np.load(f"{folder}/is_closed.npy")
        else:
            self.is_closed = np.zeros((len(self.yarn_start),), dtype = bool)
        if os.path.exists(f"{folder}/is_pinned.npy"):
            self.is_pinned = np.load(f"{folder}/is_pinned.npy")
        else:
            self.is_pinned = None
        

    def get_next_object(self):
        # each yarn is treated as an seperate object
        ys = self.yarn_start
        for i in range(len(ys) - 1):
            end = ys[i + 1] - 3
            start = ys[i]
            v = self.spline_points[start: end]
            e0 = np.arange(start, end - 1) - start
            e1 = np.arange(start + 1, end) - start

            if self.is_closed[i]:
                e0 = np.append(e0, end - 1 - start)
                e1 = np.append(e1, 0)

            nxt = e1 + self.n_nodes
            if not self.is_closed[i]:
                nxt = np.append(nxt, -1)
            self.seg_nxt = np.concatenate([self.seg_nxt, nxt])

            e = np.hstack((e0.reshape(-1, 1), e1.reshape(-1, 1)))
            f = np.zeros((0, 3,), dtype = int)

            v *= 200.0
            yield v, e, f

@wp.kernel
def reset_nodes(x: wp.array(dtype = Node), mass: wp.array(dtype = scalar), position: wp.array(dtype = vec3)):
    i = wp.tid()
    z3 = vec3(z, z, z)
    x[i].x = position[i]
    x[i].x0 = x[i].x
    x[i].v = z3
    x[i].v0 = z3
    x[i].mass = mass[i]
    x[i].last = i - 1

@wp.kernel
def reset_segs(x: wp.array(dtype = Node), seg: wp.array(dtype = Seg), nxt: wp.array(dtype = int)):
    i = wp.tid()
    p0 = x[i].x
    n = nxt[i]
    seg[i].nxt = n
    if n >= 0:
        p1 = x[n].x
    
        seg[i].l = wp.length(p1 - p0)
        seg[i].w = wp.quaternion(scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar)
        seg[i].q = wp.normalize(from_to(e3, wp.normalize(p1 - p0)))
        seg[i].q0 = seg[i].q

        
class Yarn(PrimalRod, YarnGeometryComplex):
    def __init__(self, folder, dt):
        YarnGeometryComplex.__init__(self, folder)
        # n_nodes = self.V.shape[0]
        PrimalRod.__init__(self, self.n_nodes, dt)
        
    def reset(self):
        assert self.seg_nxt.shape[0] == self.n_nodes, f"seg_nxt has length {self.seg_nxt.shape[0]} but expected {self.n_nodes}"
        assert self.seg_nxt.max() < self.n_nodes, f"seg_nxt has value {self.seg_nxt.max()} but expected less than {self.n_nodes}"
        nxt = wp.array(self.seg_nxt, dtype = int)
        mass = wp.ones((self.n_nodes,), dtype = scalar)
        position = wp.array(self.V, dtype = vec3)
        wp.launch(reset_nodes, (self.n_nodes, ), inputs = [self.nodes, mass, position])
        wp.launch(reset_segs, (self.n_segs, ), inputs = [self.nodes, self.segs, nxt])
        wp.launch(init_rest_frame, (self.n_segs, ), inputs = [self.segs])
        wp.copy(self.soup.x_transformed, position)

        l = self.segs.numpy()['l']
        nxt = self.segs.numpy()['nxt']
        minl = l[nxt >= 0].min()
        print(f"min segment length = {minl}")
        assert minl > 1e-6, "all segments should have at least 1e-6 length"

    def define_contact_viewer_interface(self): 
        pass

def test_load():
    folder = "assets/yarn/sleeve"
    yarn = Yarn(folder, 1e-3)    
    ps.init()
    wp.config.max_unroll = 0
    wp.config.enable_backward = False
    wp.init()
    viewer = PSViewer(yarn)
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == '__main__':
    test_load()