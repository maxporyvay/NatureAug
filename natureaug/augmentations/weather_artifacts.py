from .utils import plasma_fractal

import numpy as np

from random import random


class Fog:
    def __init__(self, p=0.5, severity=1):
        self.p = p
        self.severity = severity

    def __call__(self, sample):
        if random() < self.p:
            c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][self.severity - 1]
            sample = np.array(sample, dtype=np.float32) / 255.
            max_val = sample.max()
            sample += c[0] * plasma_fractal(wibbledecay=c[1])[:227, :227][..., np.newaxis]
            return np.clip(sample * max_val / (max_val + c[0]), 0, 1) * 255
        return sample
