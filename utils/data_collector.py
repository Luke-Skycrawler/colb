import numpy as np
import matplotlib.pyplot as plt
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
        for i in self.recorded_attributes().keys():
            if i not in data:
                print(f"warning: {i} not found in {filename}")
            self[i] = data[i]

    def plot(self, attributes):
        assert attributes in self.recorded_attributes().keys()

        data = self[attributes]
        fig = plt.figure()
        x = np.arange(data.shape[0])
        y = data[:, -1]
        plt.plot(x, y)
        

    def save(self, filename):
        np.savez(filename, **{i: self[i] for i in self.recorded_attributes().keys()})

