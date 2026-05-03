import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
class Profiller: 
    def __init__(self, max_iters, frames = 4000):

    
        self.convergence = np.zeros((frames, max_iters))
        self.alphas = np.zeros_like(self.convergence)
    
    def recorded_attributes(self):
        return {
            "convergence": float, 
            "alphas": float,
        }
    
    def load(self, filename): 
        data = np.load(filename)
        attrb = list(self.recorded_attributes().keys())
        for i in attrb:
            if i not in data:
                print(f"warning: {i} not found in {filename}")
            setattr(self, i, data[i])

    def plot(self, attributes):
        attrb = list(self.recorded_attributes().keys())
        assert attributes in attrb

        data = getattr(self, attributes)
        # fig = plt.figure()
        x = np.arange(data.shape[0])

        y = data[:, -1]
        plt.plot(x, y, label = "last iter")

        y_8 = data[:, 7]
        plt.plot(x, y_8, label = "iter 8")   
        
        y_4 = data[:, 3]
        plt.plot(x, y_4, label = "iter 4")


        y_0 = data[:, 0]
        plt.plot(x, y_0, label = "iter 0")

        plt.yscale("log")
        plt.ylim(1e-5, 1)
        plt.legend()


    def save(self, filename):

        attrb = list(self.recorded_attributes().keys())
        # print(attrb)
        np.savez(filename, **{i: getattr(self, i) for i in attrb})

if __name__ == "__main__":
    prof = Profiller(max_iters = 16, frames = 50)
    mpl.use("WebAgg")
    prof.load("output/profiler.npz")
    prof.plot("convergence")
    plt.show()