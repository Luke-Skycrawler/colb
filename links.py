import polyscope as ps 
from rbd_simple import XPBDRbd
import warp as wp
from viewer import PSViewer

from utils.scene import JSONComplex
def free_float():
    ps.init()
    wp.config.max_unroll = 0
    wp.init()
    dt = 1e-2
    # rbd = XPBDRbd(dt, ["assets/link/link.obj"])
    rbd = XPBDRbd(dt, "assets/chains.json")
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
    

