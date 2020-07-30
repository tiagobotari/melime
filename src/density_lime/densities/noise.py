import numpy as np

class NoiseDensity:
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def fill(self, image, segments):
        img = image.copy() / 255
        noise = np.random.normal(0, self.sigma, img.shape)
        fill = np.clip(img + noise, 0, 1)
        fill = (fill * 255).astype(np.uint8)
        return fill
        
