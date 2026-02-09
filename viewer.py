import polyscope as ps
import polyscope.imgui as gui
import numpy as np 
class PSViewer:
    def __init__(self, rbd):
        self.V = rbd.xcs.numpy()
        self.F = rbd.F

        self.ps_mesh = ps.register_surface_mesh("rbd", self.V, self.F)
        self.frame = 0
        self.rbd = rbd
        self.ui_pause = False
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
            self.rbd.reset()
            self.frame = self.rbd.frame
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.V)
            print(f"output/x_{self.frame}.npy saved")


        if self.animate: 
            self.rbd.step()
            self.V = self.rbd.compute_V()
            self.ps_mesh.update_vertex_positions(self.V)
            self.frame = self.rbd.frame
            
            print("frame = ", self.frame)

            if self.frame % self.capture_interval == 0:
                self.save()
        if self.frame >= self.end_frame:
            print(f"end frame = {self.frame} reached, exiting")
            quit()