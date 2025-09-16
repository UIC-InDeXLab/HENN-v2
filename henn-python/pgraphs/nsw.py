from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
import heapq
from tqdm import tqdm


class NSW(BaseProximityGraph):
    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        super().__init__(distance, enable_logging, log_level)
        """Initialize NSW graph."""
        self.init_node = None

    def build_graph(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        params: dict = None,
    ):
        """
        Build a NSW (Navigable Small World) graph for the specified layer.
        This implementation follows HNSW principles for a single layer.
        Points are inserted incrementally, each connected to M nearest neighbors
        found through greedy search with efConstruction candidates.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            params: Optional parameters for graph construction
                   Expected keys: 'M' (max connections), 'efConstruction' (search width)

        Returns:
            Dictionary mapping global indices to lists of connected global indices (adjacency list)
        """
        if params is None:
            params = {}

        # Use HNSW standard parameters
        M = params.get("M", params.get("k", 16))  # Maximum connections per node
        efConstruction = params.get(
            "efConstruction", max(M, 200)
        )  # Search width during construction

        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Initialize adjacency list
        edges = {idx: [] for idx in layer_indices}

        # Start with first point
        inserted_indices = [layer_indices[0]]

        # Insert remaining points incrementally
        for i in tqdm(range(1, n), desc="Building NSW layer"):
            current_idx = layer_indices[i]
            current_point = henn_points[current_idx]

            # Find M best candidates using search_layer with efConstruction
            candidates = self._search_layer(
                henn_points, current_point, inserted_indices, efConstruction, edges
            )

            # Select M best neighbors for insertion
            M_neighbors = self._select_neighbors_simple(candidates, M)

            # Connect current point to selected neighbors (bidirectional)
            for neighbor_idx in M_neighbors:
                # Add connection from current to neighbor
                edges[current_idx].append(neighbor_idx)
                # Add connection from neighbor to current
                edges[neighbor_idx].append(current_idx)

                # Prune connections of neighbor if it exceeds max connections
                # if len(edges[neighbor_idx]) > M:
                # self._prune_connections(henn_points, neighbor_idx, M, edges)

            # Add current point to inserted set
            inserted_indices.append(current_idx)

        # Find the highest degree node for use as initial search node
        self._find_highest_degree(edges, layer_indices)

        return edges

    def _search_layer(
        self,
        henn_points: np.ndarray,
        query_point: np.ndarray,
        layer_points: list,
        ef: int,
        edges: dict,
    ):
        """
        Search layer for ef closest points to query_point using HNSW search algorithm.

        Args:
            henn_points: All points in the HENN structure
            query_point: Point to search for
            layer_points: List of points available in this layer
            ef: Number of closest points to find
            edges: Current adjacency list

        Returns:
            List of (distance, point_idx) tuples for ef closest points
        """
        if not layer_points:
            return []

        # Initialize with random entry point
        entry_point = random.choice(layer_points)

        visited = set()
        candidates = []  # Min heap for candidates to explore
        w = []  # Max heap for dynamic candidates (best ef found so far)

        # Calculate distance to entry point
        if self.distance == "cosine":
            norm_query_point = query_point / np.linalg.norm(query_point)
            entry_dist = 1 - np.dot(henn_points[entry_point], norm_query_point)
        else:  # Default to L2 distance
            entry_dist = np.linalg.norm(henn_points[entry_point] - query_point)

        # Add entry point to candidates and w
        heapq.heappush(candidates, (entry_dist, entry_point))
        heapq.heappush(w, (-entry_dist, entry_point))  # Negative for max heap
        visited.add(entry_point)

        while candidates:
            current_dist, current_idx = heapq.heappop(candidates)

            # If current candidate is farther than the farthest in w, stop
            if len(w) >= ef and current_dist > -w[0][0]:
                break

            # Explore neighbors of current point
            for neighbor_idx in edges.get(current_idx, []):
                if neighbor_idx not in visited and neighbor_idx in layer_points:
                    visited.add(neighbor_idx)

                    if self.distance == "cosine":
                        neighbor_dist = 1 - np.dot(
                            henn_points[neighbor_idx], query_point
                        )
                    else:  # Default to L2 distance
                        neighbor_dist = np.linalg.norm(
                            henn_points[neighbor_idx] - query_point
                        )

                    # Add to candidates for exploration
                    heapq.heappush(candidates, (neighbor_dist, neighbor_idx))

                    # Update w (keep best ef candidates)
                    if len(w) < ef:
                        heapq.heappush(w, (-neighbor_dist, neighbor_idx))
                    elif neighbor_dist < -w[0][0]:  # Better than worst in w
                        heapq.heapreplace(w, (-neighbor_dist, neighbor_idx))

        # Convert w to list of (distance, idx) with positive distances
        result = [(-dist, idx) for dist, idx in w]
        result.sort()  # Sort by distance (ascending)
        return result

    def _select_neighbors_simple(self, candidates, M):
        """
        Select M best neighbors from candidates using simple distance-based selection.

        Args:
            candidates: List of (distance, point_idx) tuples
            M: Maximum number of neighbors to select

        Returns:
            List of selected neighbor indices
        """
        # Sort by distance and take M closest
        candidates.sort(key=lambda x: x[0])
        return [idx for _, idx in candidates[:M]]

    # def _prune_connections(
    #     self, henn_points: np.ndarray, node_idx: int, M: int, edges: dict
    # ):
    #     """
    #     Prune connections of a node to maintain degree constraint.
    #     Uses simple distance-based pruning to keep M closest neighbors.

    #     Args:
    #         henn_points: All points in the HENN structure
    #         node_idx: Index of the node to prune
    #         M: Maximum allowed degree
    #         edges: Current adjacency list (modified in-place)
    #     """
    #     if len(edges[node_idx]) <= M:
    #         return

    #     node_point = henn_points[node_idx]

    #     # Calculate distances to all current neighbors
    #     neighbor_distances = []
    #     for neighbor_idx in edges[node_idx]:
    #         dist = np.linalg.norm(henn_points[neighbor_idx] - node_point)
    #         neighbor_distances.append((dist, neighbor_idx))

    #     # Sort by distance and keep only M closest
    #     neighbor_distances.sort()
    #     closest_neighbors = [idx for _, idx in neighbor_distances[:M]]

    #     # Remove connections to pruned neighbors (bidirectional)
    #     for neighbor_idx in edges[node_idx]:
    #         if neighbor_idx not in closest_neighbors:
    #             # Remove bidirectional connection
    #             if node_idx in edges[neighbor_idx]:
    #                 edges[neighbor_idx].remove(node_idx)

    #     # Update the node's adjacency list
    #     edges[node_idx] = closest_neighbors

    def _find_highest_degree(self, edges, layer_indices):
        # Find node with highest degree
        max_degree = -1
        best_node = None

        for node_idx in layer_indices:
            degree = len(edges.get(node_idx, []))
            if degree > max_degree:
                max_degree = degree
                best_node = node_idx
        self.init_node = best_node

    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ):
        """
        For NSW graphs, select the node with highest degree as it's likely to be well-connected.
        If edges not available or all have same degree, use random selection.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            edges: Adjacency list (used to find highest degree node)

        Returns:
            Global index of the node with highest degree
        """
        if not layer_indices:
            return None

        if edges is None:
            return np.random.choice(layer_indices)

        return (
            self.init_node
            if self.init_node is not None
            else np.random.choice(layer_indices)
        )
