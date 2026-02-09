import numpy as np
import igl
class SlabMesh:
    def __init__(self, filename = ""): 
        self.V = None
        self.E  = None
        self.F = None
        self.R = None

        if filename != "":
            self.load(filename)

        self.nv = self.V.shape[0]
        self.ne = self.E.shape[0]
        self.nf = self.F.shape[0]
            
    def load(self, filename): 
        vertices = []
        edges = []
        faces = []

        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()

                if not parts:
                    continue

                if parts[0] == "v": 
                    vertex = list(map(float, parts[1:]))
                    vertices.append(np.array(vertex))

                elif parts[0] == "e":
                    edge = [int(idx) for idx in parts[1:]]
                    edges.append(np.array(edge))
                
                elif parts[0] == "f":
                    faces.append(np.array([int(idx) for idx in parts[1:]]))

        vertices = np.array(vertices)
        self.V = vertices[:, : 3]
        self.R = vertices[:, 3]
        
        self.E0 = np.array(edges)
        # extracts edges from faces and append to the edge set
        self.F = np.array(faces).reshape(-1, 3)
        E = igl.edges(self.F) if self.F.shape[0] > 0 else np.zeros((0, 2), dtype = int)
        E = np.concatenate((self.E0, E))
        sorted_edges = np.sort(E, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)
        self.E = unique_edges
        

    def export_ma(self, filename):
        assert(filename.endswith(".ma"))

        to_strv = lambda x, r: f"v {x[0]} {x[1]} {x[2]} {r}\n"

        to_stre = lambda e: f"e {e[0]} {e[1]}\n"
        to_strf = lambda f: f"f {f[0]} {f[1]} {f[2]}\n"

        linesv = [to_strv(*vr) for vr in zip(self.V, self.R)] 
        linese = [to_stre(t) for t in self.E0]
        linesf = [to_strf(t) for t in self.F]
        with open(filename, 'w') as f:
            tot = f"{self.V.shape[0]} {self.E0.shape[0]} {self.F.shape[0]}\n"
            f.writelines([tot] + linesv + linese + linesf)

    def export_ply(self, filename):
        assert(filename.endswith(".ply"))
        headers = [
            'ply\n',
            'format ascii 1.0\n',
            f'element vertex {self.V.shape[0]}\n',
            'property float x\n',
            'property float y\n',
            'property float z\n',
            'property float r\n',
            f'element face {self.E.shape[0] + self.F.shape[0]}\n',
            'property list uchar int vertex_indices\n',
            'end_header\n'
        ]
        to_strv = lambda x, r: f"{x[0]} {x[1]} {x[2]} {r}\n"
        to_stre = lambda e: f"2 {e[0]} {e[1]}\n"
        to_strf = lambda f: f"3 {f[0]} {f[1]} {f[2]}\n"

        linesv = [to_strv(*vr) for vr in zip(self.V, self.R)]
        linese = [to_stre(t) for t in self.E]
        linesf = [to_strf(t) for t in self.F]
        with open(filename, 'w') as f:
            f.writelines(headers + linesv + linese + linesf)

def slabmesh_default() -> SlabMesh: 
    
    mesh = SlabMesh()
    mesh.V = np.array([
        [-0.4, 0.0, 0.0],
        [0.4, 0.0, 0.0]
    ])
    mesh.E = np.array([
        [0, 1]
    ], dtype = np.int32)

    mesh.F = np.zeros((0, 3), dtype = np.int32)

    mesh.R = np.array([0.1, 0.1])
    return mesh


if __name__ == "__main__":
    mesh = SlabMesh("data/test.ma")
    mesh.export_ply("output/test.ply")
    mesh.export_ma("output/test.ma")
    