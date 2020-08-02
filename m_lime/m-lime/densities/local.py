import numpy as np

class LocalDensity:
    def fill(self, image, segments):
        fill = image.copy()
        for seg in np.unique(segments):
            sel = segments == seg
            fill[sel] = np.mean(image[sel], axis=0)
            
        return fill
