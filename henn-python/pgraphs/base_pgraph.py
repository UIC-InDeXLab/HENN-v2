import numpy as np
import logging


class BaseProximityGraph:
    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        self.distance = distance
        self.enable_logging = enable_logging
        self.log_level = log_level

        if self.enable_logging:
            self.logger = logging.getLogger(f"{id(self)}")
            self._setup_logging(enable_logging, log_level)

            # Create console handler if it doesn't exist
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            # Set log level
            level = getattr(logging, log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
            self.logger.disabled = False

    def build_graph(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        params: dict = None,
        distance: str = "l2",  # l2 or cosine
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
