from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np


class Knn(BaseProximityGraph):
    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ):
        """
        Build a k-NN graph for the specified layer.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            params: Optional parameters for graph construction (expects 'k' key)

        Returns:
            Dictionary mapping global indices to lists of connected global indices (adjacency list)
        """
        if params is None or "k" not in params:
            raise ValueError("Parameter 'k' must be specified in params")

        k = params["k"]
        layer_points = henn_points[layer_indices]
        n = len(layer_indices)
        edges = {idx: [] for idx in layer_indices}

        for i in range(n):
            distances = np.linalg.norm(layer_points - layer_points[i], axis=1)
            nearest_indices = np.argsort(distances)[1 : k + 1]  # Skip self (index 0)
            edges[layer_indices[i]] = [layer_indices[j] for j in nearest_indices]

        return edges
