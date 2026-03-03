import warp as wp
import igl
import numpy as np
from quat_util import vec3, vec4, mat44, scalar, Rq
from medial import SlabMesh
@wp.kernel
def _compute_transforms(transforms: wp.array(dtype=mat44), q: wp.array(dtype=vec4), c: wp.array(dtype=vec3)):
    i = wp.tid(0)
    qi = q[i]
    ci = c[i]
    ri = Rq(qi)
    z = scalar(0.0)
    o = scalar(1.0)
    ti = mat44(
        ri[0, 0], ri[0, 1], ri[0, 2], ci[0],
        ri[1, 0], ri[1, 1], ri[1, 2], ci[1],
        ri[2, 0], ri[2, 1], ri[2, 2], ci[2],
        z,       z,        z,        o
    )
    transforms[i] = ti

@wp.struct 
class Soup: 
    xcs: wp.array(dtype = vec3)
    triangles: wp.array(dtype = int)
    edges: wp.array(dtype = int)
    body: wp.array(dtype = int)
    x_transformed: wp.array(dtype = vec3)


class SimComplexBase:
    def __init__(self): 
        self.n_nodes = 0
        V = np.zeros((0, 3), dtype = float)
        F_from_file = np.zeros((0, 3), dtype = int)
        F = np.zeros((0, 3), dtype = int)
        E = np.zeros((0, 2), dtype = int)
        B = np.zeros(0, dtype = int)
        body_idx = 0

        nxt = self.get_next_object()
        for nxt in self.get_next_object():
            v, e, ff = nxt
            b = np.ones((v.shape[0],), dtype = int) * body_idx
            V = np.vstack((V, v))
            E = np.vstack((E, e + self.n_nodes))
            B = np.hstack((B, b))
            if ff.shape[0]:
                F_from_file = np.vstack([F_from_file, ff + self.n_nodes])

            self.n_nodes += v.shape[0]
            body_idx += 1

        self.xcs = wp.zeros((self.n_nodes), dtype = vec3) 
        self.xcs.assign(V)

        F = np.vstack([F, F_from_file])
        self.indices = wp.array(F.reshape(-1), dtype = int)        
        self.edges = wp.array(E.reshape(-1), dtype = int)
        self.body = wp.array(B, dtype = int)
        

        geom = Soup()
        geom.xcs = self.xcs
        geom.triangles = self.indices
        geom.body = self.body
        geom.edges = self.edges
        geom.x_transformed = wp.zeros_like(self.xcs)
        self.soup = geom

        self.V = V
        self.F = F
        self.E = E
        self.n_bodies = body_idx

    def get_next_object(self):
        return []

class OBJComplex(SimComplexBase):
    def __init__(self):
        '''
        form a complex of all simulation meshes and exposes tet geometry interface 

        NOTE: need to have self.meshes_filename predefined before calling super().__init__()
        '''
        super().__init__()
        print(f"{self.meshes_filename} loaded, {self.n_nodes} nodes, {self.F.shape[0]} faces")

    def get_next_object(self):
        # transforms = self.compute_transforms() 
        meshes_filename = self.meshes_filename
        for f in meshes_filename:
            
            if f.endswith(".obj"):
                v, _, _, ff, _, _ = igl.read_obj(f)
                e = igl.edges(ff)

            elif f.endswith(".ma"): 
                mesh = SlabMesh(f)
                v, e, ff, r = mesh.V, mesh.E, mesh.F, mesh.R
            
            elif f.endswith(".tobj"):
                print("tobj not supported")
                quit()
            
            yield v, e, ff

    def compute_transforms(self):
        '''
        must have self.c_init, self.q_init defined
        '''
        qs = self.q_init
        cs = self.c_init
        q = wp.array(qs, dtype = vec4)
        c = wp.array(cs, dtype = vec3)

        n_meshes = qs.shape[0]
        transforms = wp.zeros((n_meshes, ), dtype = mat44)
        wp.launch(_compute_transforms, (n_meshes,), inputs = [transforms, q, c])
        return transforms.numpy()