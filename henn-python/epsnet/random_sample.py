from epsnet.base_epsnet import BaseEPSNet
import numpy as np


class RandomSample(BaseEPSNet):
    def build_epsnet(self, points: np.ndarray, eps=None, size=None):
        if size is None or size >= len(points):
            return list(range(len(points)))
        indices = np.random.choice(len(points), size=size, replace=False)
        return indices
