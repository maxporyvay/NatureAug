from utils import plasma_fractal

import numpy as np

from random import random


def random_fog(x, p=0.5, severity=1):
    if random() < p:
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.
        max_val = x.max()
        x += c[0] * plasma_fractal(wibbledecay=c[1])[:227, :227][..., np.newaxis]
        return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    return x
