import numpy as np

class MeanDensity:
    def __init__(self):
        # Shape (1, 3) for broadcasting
        self.means = (np.array([[ 0.47611974510772803, 0.4536891150122712, 0.4014851218547386 ]]) * 255).astype(np.uint8)

    def fill(self, image, segments):
        fill = np.ones_like(image)
        fill = fill * self.means 
        return fill

