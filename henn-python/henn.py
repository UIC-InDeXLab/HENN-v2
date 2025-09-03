import numpy as np
from epsnet.base_epsnet import BaseEPSNet
from epsnet.random_sample import RandomSample
from pgraphs.base_pgraph import BaseProximityGraph
from pgraphs.knn import Knn
from pgraphs.nsw import NSW
from pgraphs.nsg import NSG
from pgraphs.fanng import FANNG
from pgraphs.kgraph import KGraph
import math
import logging


class HENNConfig:
    def __init__(
        self,
        epsnet_algorithm: str = "random",  # random, discrepancy, budget-aware
        epsnet_params: dict = None,
        pgraph_algorithm: str = "NSW",  # nsw, knn, nsg, fanng, kgraph
        pgraph_params: dict = None,
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        # TODO: build an instance of base_epsnet
        self.epsnet_algorithm = RandomSample()
        self.epsnet_params = epsnet_params or {}
        
        # Build an instance of base_pgraph based on algorithm choice
        if pgraph_algorithm.lower() == "knn":
            self.pgraph_algorithm = Knn()
            # Provide default parameters for KNN if none specified
            if not pgraph_params:
                self.pgraph_params = {"k": 16}
            else:
                self.pgraph_params = pgraph_params
        elif pgraph_algorithm.lower() == "nsw":
            self.pgraph_algorithm = NSW()
            # Provide default parameters for NSW if none specified
            if not pgraph_params:
                self.pgraph_params = {"M": 16, "efConstruction": 200}
            else:
                self.pgraph_params = pgraph_params
        elif pgraph_algorithm.lower() == "nsg":
            self.pgraph_algorithm = NSG()
            # Provide default parameters for NSG if none specified
            if not pgraph_params:
                self.pgraph_params = {"R": 16, "L": 100, "C": 300}
            else:
                self.pgraph_params = pgraph_params
        elif pgraph_algorithm.lower() == "fanng":
            self.pgraph_algorithm = FANNG()
            # Provide default parameters for FANNG if none specified
            if not pgraph_params:
                self.pgraph_params = {"K": 16, "L": 32, "R": 16, "alpha": 1.2}
            else:
                self.pgraph_params = pgraph_params
        elif pgraph_algorithm.lower() == "kgraph":
            self.pgraph_algorithm = KGraph()
            # Provide default parameters for KGraph if none specified
            if not pgraph_params:
                self.pgraph_params = {"k": 16, "rho": 1.0, "delta": 0.001, "max_iterations": 30}
            else:
                self.pgraph_params = pgraph_params
        else:
            # Default to NSW
            self.pgraph_algorithm = NSW()
            self.pgraph_params = pgraph_params or {"M": 16, "efConstruction": 200}
            
        self.enable_logging = enable_logging
        self.log_level = log_level


class Layer:
    def __init__(self, henn_points: np.ndarray):
        self.henn_points = henn_points  # Reference to all HENN points
        self.indices = None  # Indices of points in this layer
        self.edges = None  # Edges between points in this layer
        self.n = 0
        self.pgraph = None  # Reference to the proximity graph used to build this layer

    def add_indices(self, indices: list):
        self.indices = indices
        self.n = len(indices)

    def build_graph(self, pgraph: BaseProximityGraph, henn_points: np.ndarray, params: dict = None):
        if self.indices is None or self.n == 0:
            self.edges = {}
            return

        # Store reference to the proximity graph for later use in search
        self.pgraph = pgraph
        
        # Build graph using global indices - pgraph has access to all henn_points
        # and returns edges with global indices
        self.edges = pgraph.build_graph(henn_points, self.indices, params)

    def search(self, query_point: np.ndarray, k: int = 1, entry_point=None):
        """Search for k nearest neighbors using HNSW-like greedy search."""
        if self.indices is None or self.edges is None or self.n == 0:
            return []

        # Use provided entry point if valid, otherwise use proximity graph's selection strategy
        if entry_point is not None and entry_point in self.indices:
            start_global_idx = entry_point
        else:
            # Use the proximity graph's initial node selection strategy
            if self.pgraph is not None:
                start_global_idx = self.pgraph.get_initial_search_node(
                    self.henn_points, self.indices, self.edges
                )
            else:
                # Fallback to random selection if no pgraph reference
                random_local_idx = np.random.randint(0, self.n)
                start_global_idx = self.indices[random_local_idx]

        visited = set()
        candidates = [
            (
                np.linalg.norm(self.henn_points[start_global_idx] - query_point),
                start_global_idx,
            )
        ]
        w = []  # Dynamic list of closest points found so far

        while candidates:
            # Get closest unvisited candidate
            candidates.sort()
            current_dist, current_global_idx = candidates.pop(0)

            if current_global_idx in visited:
                continue

            visited.add(current_global_idx)

            # Add to result set if better than worst in w or w not full
            if len(w) < k:
                w.append((current_dist, current_global_idx))
                w.sort()
            elif current_dist < w[-1][0]:
                w.append((current_dist, current_global_idx))
                w.sort()
                w = w[:k]  # Keep only k best
            else:
                # Current node is farther than all in w, stop search
                break

            # Explore neighbors using global indices from edges
            if current_global_idx in self.edges:
                for neighbor_global_idx in self.edges[current_global_idx]:
                    # Only consider neighbors that are in this layer
                    if (
                        neighbor_global_idx in self.indices
                        and neighbor_global_idx not in visited
                    ):
                        neighbor_dist = np.linalg.norm(
                            self.henn_points[neighbor_global_idx] - query_point
                        )
                        candidates.append((neighbor_dist, neighbor_global_idx))

        # Return global indices of k nearest neighbors
        return [global_idx for _, global_idx in w]


class HENN:
    def __init__(
        self,
        points: np.ndarray,
        config: HENNConfig = None,
        exp_decay: int = 2,
    ):
        self.points = points
        self.d = points.shape[1]
        self.n = points.shape[0]
        self.exp_decay = exp_decay

        # Use default config if none provided
        if config is None:
            config = HENNConfig()
        
        self.config = config
        self.epsnet = config.epsnet_algorithm
        self.pgraph = config.pgraph_algorithm

        # Setup logging
        self.logger = logging.getLogger(f"HENN_{id(self)}")
        self._setup_logging(config.enable_logging, config.log_level)

        # To build
        self.layers = []
        self.L = math.floor(math.log(self.n, self.exp_decay))

        self.logger.info(
            f"Initialized HENN with {self.n} points in {self.d} dimensions"
        )

    def _setup_logging(self, enable_logging: bool, log_level: str):
        """Setup logging configuration for this HENN instance."""
        if enable_logging:
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
        else:
            self.logger.disabled = True

    def enable_logging(self, log_level: str = "INFO"):
        """Enable logging with specified level."""
        self._setup_logging(True, log_level)
        self.logger.info("Logging enabled")

    def disable_logging(self):
        """Disable logging."""
        self.logger.info("Disabling logging")
        self.logger.disabled = True

    def build(self):
        self.logger.info(f"Starting to build HENN with {self.L} layers...")
        current_indices = list(range(self.n))  # Start with all indices

        for l in range(self.L):
            self.logger.debug(f"Building layer {l+1}/{self.L}")
            layer = Layer(self.points)  # Pass reference to all points
            expected_size = self.n // (self.exp_decay**l)
            self.logger.debug(f"Expected epsnet size for layer {l+1}: {expected_size}")

            # Build epsnet from current indices - returns local indices within current_indices
            current_points = self.points[current_indices]
            selected_local_indices = self.epsnet.build_epsnet(
                points=current_points, size=expected_size
            )

            # Convert local indices back to global indices
            selected_global_indices = [
                current_indices[i] for i in selected_local_indices
            ]

            layer.add_indices(selected_global_indices)
            layer.build_graph(self.pgraph, self.points, self.config.pgraph_params)
            self.layers.append(layer)

            # Update for next iteration
            current_indices = selected_global_indices

        self.logger.info("HENN build completed successfully")

    def query(self, query_point: np.ndarray, k: int = 1, ef: int = 10):
        self.logger.debug(f"Querying for {k} nearest neighbors with ef={ef}")

        if not self.layers:
            self.logger.warning("No layers built. Call build() first.")
            return []

        entry_points = []

        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            self.logger.debug(
                f"Searching in layer {layer_idx + 1}/{len(self.layers)} with {layer.n} points"
            )

            if layer.n == 0:
                self.logger.debug(f"Layer {layer_idx + 1} is empty, skipping")
                continue

            # For the base layer (layer 0), search for max(ef, k) candidates
            # For upper layers, search for 1 candidate (greedy search)
            if layer_idx == 0:  # Base layer
                search_k = max(ef, k)
            else:  # Upper layers
                search_k = 1

            # Convert global entry points to local indices for this layer
            if entry_points:
                # Use the first valid global entry point directly
                entry_point = entry_points[0]
                self.logger.debug(f"Using global entry point {entry_point}")
                global_candidates = layer.search(query_point, search_k, entry_point)
            else:
                # No entry point available, let layer.search use random
                self.logger.debug("No entry points, using random entry point")
                global_candidates = layer.search(query_point, search_k)

            # Update entry points for the next (lower) layer
            entry_points = global_candidates

            if layer_idx == 0:  # Base layer
                # Return top-k from the final candidates
                if len(global_candidates) == 0:
                    self.logger.warning("No candidates found in base layer")
                    return []

                # Calculate actual distances and sort to ensure top-k
                candidate_distances = []
                for global_idx in global_candidates:
                    dist = np.linalg.norm(self.points[global_idx] - query_point)
                    candidate_distances.append((dist, global_idx))

                candidate_distances.sort(key=lambda x: x[0])
                result = [idx for _, idx in candidate_distances[:k]]

                self.logger.debug(
                    f"Returning {len(result)} nearest neighbors: {result}"
                )
                return result

        return []
