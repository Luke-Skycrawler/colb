import warp as wp 
from geometry import SimComplexBase
import numpy as np 
from .cosserat import StableCosserat, Node, Seg
from scalar_types import *
from .spring import reset_segs, init_rest_frame
import os
from viewer import PSViewer
import polyscope as ps
from .rod_contact import PrimalRod


z = scalar(0.0)
o = scalar(1.0)

class YarnGeometryComplex(SimComplexBase):
    def __init__(self, folder = "assets/yarn"): 
        '''
        reads from yarn_start.npy, is_closed.npy (if exists), spline_points.npy, and is_pinned.npy(if exists)
        ''' 

        self.read_yarns(folder)
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

class Yarn(PrimalRod, YarnGeometryComplex):
    def __init__(self, folder, dt):
        YarnGeometryComplex.__init__(self, folder)
        n_nodes = self.spline_points.shape[0]
        PrimalRod.__init__(self, n_nodes, dt)
        
    def reset(self):
        mass = wp.ones((self.n_nodes,), dtype = scalar)
        position = wp.array(self.V, dtype = vec3)
        wp.launch(reset_nodes, (self.n_nodes, ), inputs = [self.nodes, mass, position])
        wp.launch(reset_segs, (self.n_segs, ), inputs = [self.nodes, self.segs])
        wp.launch(init_rest_frame, (self.n_segs, ), inputs = [self.segs])
        wp.copy(self.soup.x_transformed, position)

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