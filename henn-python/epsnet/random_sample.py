from epsnet.base_epsnet import BaseEPSNet
import numpy as np


class RandomSample(BaseEPSNet):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def build_epsnet(self, points: np.ndarray, eps=None, size=None):
        n, d = points.shape
        if size is None:
            if eps is None:
                raise ValueError("Either eps or size must be provided")
            size = BaseEPSNet.get_size(n, eps, d)

        size = min(size, n)

        if size >= len(points):
            return list(range(len(points)))
        indices = np.random.choice(len(points), size=size, replace=False)
        return indices
