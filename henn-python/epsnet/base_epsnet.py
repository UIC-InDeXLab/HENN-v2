import numpy as np


class BaseEPSNet:
    def build_epsnet(self, points: np.ndarray, eps=None, size=None):
        """Return indices of selected points for EPSNet layer."""
        raise NotImplementedError("Subclasses must implement build_epsnet method")

    @classmethod
    def get_eps(n, size, d):
        raise NotImplementedError("get_eps method")
