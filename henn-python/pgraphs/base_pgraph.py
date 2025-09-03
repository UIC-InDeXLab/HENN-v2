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
    
    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ):
        """
        Get the initial node for search in this layer.
        
        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            edges: Adjacency list (if needed for selection strategy)
            
        Returns:
            Global index of the initial search node
        """
        # Default implementation: random selection
        if not layer_indices:
            return None
        return np.random.choice(layer_indices)
