import polyscope as ps
import polyscope.imgui as gui
import numpy as np 
class PSViewer:
    def __init__(self, rod):
        self.nodes = rod.nodes.numpy()["x"]
        self.n_nodes = self.nodes.shape[0]
        self.l0 = rod.segs.numpy()["l"]
        select = self.l0 > 0.0
        e_start = np.arange(self.n_nodes - 1)[select]
        e_end = e_start + 1
        self.E = np.hstack((e_start.reshape(-1, 1), e_end.reshape(-1, 1)))


        self.ps_segs = ps.register_curve_network("rod", self.nodes, self.E)
        self.ps_segs.set_radius(5e-3, relative= False)
        self.rod = rod

        self.frame = 0
        self.ui_pause = True
        self.animate = False
        self.ui_reload_from = 0
        self.end_frame = 8000
        self.capture_interval = 1
     
    def save(self):
        return 
        # ps.screenshot(f"output/{self.frame:04d}.jpg")
        self.frame = self.rbd.frame
        # if self.frame % 4 == 0:
        if False:
            igl.write_obj(f"output/obj/{self.frame:04d}.obj", self.V, self.F)
        if hasattr(self.rbd, "save_states"):
            self.rbd.save_states()

    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.rod.reset()
            self.frame = self.rod.frame
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.nodes)
            print(f"output/x_{self.frame}.npy saved")


        if self.animate: 
            self.rod.step()
            self.nodes[:] = self.rod.nodes.numpy()["x"]
            self.ps_segs.update_node_positions(self.nodes)
            self.frame = self.rod.frame

            print("frame = ", self.frame)
            # print(f"position = {self.nodes}")
            # print(f"quats = {self.rod.segs.numpy()['q']}")

            if self.frame % self.capture_interval == 0:
                self.save()
        if self.frame >= self.end_frame:
            print(f"end frame = {self.frame} reached, exiting")
            quit()