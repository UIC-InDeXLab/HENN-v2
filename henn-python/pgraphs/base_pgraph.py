import numpy as np


class BaseProximityGraph:
    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ):
        """
        Build a proximity graph for the specified layer.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            params: Optional parameters for graph construction

        Returns:
            Dictionary mapping global indices to lists of connected global indices (adjacency list)
        """
        raise NotImplementedError("Subclasses must implement build_graph method")
