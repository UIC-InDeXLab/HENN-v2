from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
from tqdm import tqdm


class Knn(BaseProximityGraph):
    def __init__(
        self,
        distance="l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        super().__init__(distance, enable_logging, log_level)

    def build_graph(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        params: dict = None,
    ):
        """
        Build a k-NN graph for the specified layer using memory-efficient chunked processing.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            params: Optional parameters for graph construction (expects 'k' and 'chunk_size' keys)

        Returns:
            Dictionary mapping global indices to lists of connected global indices (adjacency list)
        """
        if params is None:
            params = {}

        k = params.get("k", 16)
        chunk_size = params.get("chunk_size", 1000)  # Process in chunks to save memory
        layer_points = henn_points[layer_indices]
        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Limit k to maximum possible neighbors
        k = min(k, n - 1)

        print(
            f"Building k-NN graph with memory-efficient chunked processing for {n} points..."
        )
        print(f"Using chunk size: {chunk_size}, k: {k}")

        # Pre-compute normalized points for cosine distance
        if self.distance == "cosine":
            norms = np.linalg.norm(layer_points, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_points = layer_points / norms
        else:
            # Pre-compute squared norms for L2 distance
            points_squared = np.sum(layer_points**2, axis=1)
            normalized_points = None

        edges = {}

        # Process points in chunks to avoid memory overflow
        for chunk_start in tqdm(range(0, n, chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk_indices = range(chunk_start, chunk_end)
            chunk_points = layer_points[chunk_indices]

            if self.distance == "cosine":
                chunk_normalized = normalized_points[chunk_indices]
                # Compute cosine similarities for this chunk against all points
                similarity_matrix = chunk_normalized @ normalized_points.T
                # Convert to distance matrix (cosine distance = 1 - cosine similarity)
                distance_matrix = 1 - similarity_matrix
            else:
                # Compute L2 distances for this chunk against all points
                # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
                chunk_squared = points_squared[chunk_indices]
                distance_matrix = (
                    chunk_squared[:, np.newaxis]
                    + points_squared[np.newaxis, :]
                    - 2 * chunk_points @ layer_points.T
                )
                # Take square root and ensure non-negative (numerical stability)
                distance_matrix = np.sqrt(np.maximum(distance_matrix, 0))

            # Set distances to self as infinity to exclude self-connections
            for i, global_i in enumerate(chunk_indices):
                distance_matrix[i, global_i] = np.inf

            # Find k nearest neighbors for each point in this chunk
            if k < n // 2:
                # Use argpartition for efficiency when k is small
                knn_indices = np.argpartition(distance_matrix, k, axis=1)[:, :k]

                # Sort only the k nearest neighbors for each point
                for i in range(len(chunk_indices)):
                    sorted_idx = np.argsort(distance_matrix[i, knn_indices[i]])
                    knn_indices[i] = knn_indices[i, sorted_idx]
            else:
                # Use argsort when k is large (close to n)
                knn_indices = np.argsort(distance_matrix, axis=1)[:, :k]

            # Build adjacency list for this chunk
            for i, local_idx in enumerate(chunk_indices):
                global_idx = layer_indices[local_idx]
                neighbor_local_indices = knn_indices[i]
                edges[global_idx] = [layer_indices[j] for j in neighbor_local_indices]

        return edges

    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ):
        """
        For k-NN graphs, use random selection as all nodes are equivalent.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            edges: Adjacency list (not used for k-NN)

        Returns:
            Global index of a random node
        """
        if not layer_indices:
            return None
        return np.random.choice(layer_indices)
