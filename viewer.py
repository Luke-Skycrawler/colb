import polyscope as ps
import polyscope.imgui as gui
import numpy as np 
class PSViewer:
    def __init__(self, rbd):
        self.V = rbd.soup.x_transformed.numpy()
        # self.F = rbd.F

        # self.ps_mesh = ps.register_surface_mesh("rbd", self.V, self.F)
        self.ps_medial = ps.register_curve_network("rbd", self.V, rbd.E)
        self.ps_medial.set_radius(0.0475, relative=False)
        self.frame = 0
        self.rbd = rbd
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
            contact_points, magnitudes = self.rbd.get_contact_points()
            contacts = ps.register_point_cloud("contact points", contact_points, color = (1.0, 0.0, 0.0))
            if len(contact_points) > 0:
                contacts.add_scalar_quantity("magnitude", magnitudes * 20.0)
                contacts.set_point_radius_quantity("magnitude", autoscale = False)
            # self.ps_mesh.update_vertex_positions(self.V)
            self.ps_medial.update_node_positions(self.V)
            self.frame = self.rbd.frame
            
            print("frame = ", self.frame)

            if self.frame % self.capture_interval == 0:
                self.save()
        if self.frame >= self.end_frame:
            print(f"end frame = {self.frame} reached, exiting")
            quit()