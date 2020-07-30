import numpy as np
from PIL import Image, ImageFilter

class GaussianBlurDensity:
    def __init__(self, radius=10):
        self.radius = radius

    def fill(self, image, segments):
        fill = Image.fromarray(image.copy())
        fill = fill.filter(ImageFilter.GaussianBlur(radius=self.radius))
        fill = np.asarray(fill)

        return fill
