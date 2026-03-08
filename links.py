import polyscope as ps 
import warp as wp
from viewer import PSViewer
from utils.scene import JSONComplex
from xpbd_contact import XPBDRbd
from primal import PrimalRbd
from vbd import VBDRbd
from gauss_newton import LineSearchGDRbd

def free_float():
    ps.init()
    wp.config.max_unroll = 0
    wp.init()
    dt = 4e-3
    # rbd = XPBDRbd(dt, ["assets/link/link.obj"])
    # rbd = XPBDRbd(dt, "assets/chains.json")
    # rbd = PrimalRbd(dt, "assets/chains.json")
    # rbd = VBDRbd(dt, "assets/chains.json")
    rbd = LineSearchGDRbd(dt, "assets/chains.json")

    viewer = PSViewer(rbd)
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()

def load_chains(): 
    scene = JSONComplex(scene_config_file="assets/chains.json")
    print(scene.V.shape)
    
if __name__ == "__main__":
    free_float()
    # load_chains()
    

