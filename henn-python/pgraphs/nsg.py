from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
from typing import List, Tuple, Dict, Set
from tqdm import tqdm


class NSG(BaseProximityGraph):
    """
    Navigable Sparse Graph (NSG) implementation.

    NSG is an improvement over NSW that creates a sparse, navigable graph
    with controlled out-degree and improved connectivity. The key ideas are:
    1. Use a navigation node (medoid) as entry point
    2. Build graph with controlled out-degree using neighbor selection strategy
    3. Ensure strong connectivity through pruning and reconnection
    """

    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        super().__init__(distance, enable_logging, log_level)
        self.init_node = None

    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ):
        """
        Build a NSG (Navigable Sparse Graph) for the specified layer.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            params: Optional parameters for graph construction
                   Expected keys: 'R' (max out-degree), 'L' (search list size),
                   'C' (max candidates for selection)

        Returns:
            Dictionary mapping global indices to lists of connected global indices (adjacency list)
        """
        if params is None:
            params = {}

        # Default parameters
        R = params.get("R", 16)  # Maximum out-degree
        L = params.get("L", 100)  # Search list size during construction
        C = params.get("C", 500)  # Maximum candidates for neighbor selection

        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Step 1: Find navigation node (medoid)
        print("Finding medoid for navigation node...")
        navigation_node = self._find_medoid(henn_points, layer_indices)

        # Step 2: Build initial graph using k-NN
        print("Building initial k-NN graph...")
        initial_graph = self._build_initial_knn_graph(henn_points, layer_indices, R)

        # Step 3: Build NSG by iteratively improving connections
        print("Building NSG from initial k-NN graph...")
        nsg_graph = self._build_nsg_from_knn(
            henn_points, layer_indices, initial_graph, navigation_node, R, L, C
        )

        # Step 4: Ensure connectivity
        print("Ensuring graph connectivity...")
        final_graph = self._ensure_connectivity(
            henn_points, layer_indices, nsg_graph, navigation_node
        )

        self.init_node = navigation_node

        return final_graph

    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ):
        """
        For NSG graphs, use the medoid (navigation node) as the initial search node.
        This is the optimal entry point for NSG as it's designed to be the most central node.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            edges: Adjacency list (not used for NSG as we compute medoid)

        Returns:
            Global index of the medoid (navigation node)
        """
        if not layer_indices:
            return None

        # Find and return the medoid (same logic as used in build_graph)
        return (
            self.init_node
            if self.init_node is not None
            else np.random.choice(layer_indices)
        )

    def _find_medoid(self, henn_points: np.ndarray, layer_indices: list) -> int:
        """
        Find the medoid (most central point) to use as navigation node.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer

        Returns:
            Global index of the medoid point
        """
        if len(layer_indices) == 1:
            return layer_indices[0]

        min_sum_dist = float("inf")
        medoid_idx = layer_indices[0]

        # Calculate sum of distances for each point to all other points
        for i, idx_i in tqdm(
            enumerate(layer_indices), desc="Finding medoid", total=len(layer_indices)
        ):
            sum_dist = 0.0
            point_i = henn_points[idx_i]

            for j, idx_j in enumerate(layer_indices):
                if i != j:
                    point_j = henn_points[idx_j]
                    if self.distance == "cosine":
                        dist = 1 - np.dot(point_i, point_j)
                    else:
                        dist = np.linalg.norm(point_i - point_j)
                    sum_dist += dist

            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                medoid_idx = idx_i

        return medoid_idx

    def _build_initial_knn_graph(
        self, henn_points: np.ndarray, layer_indices: list, k: int
    ) -> Dict[int, List[int]]:
        """
        Build initial k-NN graph as starting point for NSG construction.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            k: Number of nearest neighbors

        Returns:
            Dictionary mapping global indices to lists of k nearest neighbors
        """
        knn_graph = {idx: [] for idx in layer_indices}

        for i, idx_i in tqdm(
            enumerate(layer_indices),
            desc="Building initial k-NN graph",
            total=len(layer_indices),
        ):
            point_i = henn_points[idx_i]

            # Calculate distances to all other points
            distances = []
            for j, idx_j in enumerate(layer_indices):
                if i != j:
                    point_j = henn_points[idx_j]
                    if self.distance == "cosine":
                        dist = 1 - np.dot(point_i, point_j)
                    else:
                        dist = np.linalg.norm(point_i - point_j)
                    distances.append((dist, idx_j))

            # Sort by distance and take k nearest
            distances.sort()
            knn_graph[idx_i] = [idx_j for _, idx_j in distances[:k]]

        return knn_graph

    def _build_nsg_from_knn(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        knn_graph: Dict[int, List[int]],
        navigation_node: int,
        R: int,
        L: int,
        C: int,
    ) -> Dict[int, List[int]]:
        """
        Build NSG from initial k-NN graph using neighbor selection strategy.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            knn_graph: Initial k-NN graph
            navigation_node: Global index of navigation node
            R: Maximum out-degree
            L: Search list size
            C: Maximum candidates for selection

        Returns:
            NSG graph as adjacency list
        """
        nsg_graph = {idx: [] for idx in layer_indices}

        # Process nodes in random order (except navigation node first)
        processing_order = [navigation_node] + [
            idx for idx in layer_indices if idx != navigation_node
        ]
        random.shuffle(processing_order[1:])  # Keep navigation node first

        for node_idx in tqdm(processing_order, desc="Building NSG"):
            # Find candidate neighbors using beam search
            candidates = self._beam_search_candidates(
                henn_points, layer_indices, node_idx, nsg_graph, navigation_node, L, C
            )

            # Select R best neighbors using NSG neighbor selection
            selected_neighbors = self._select_neighbors_nsg(
                henn_points, node_idx, candidates, R
            )

            # Add selected neighbors to NSG (directed edges)
            nsg_graph[node_idx] = selected_neighbors

        return nsg_graph

    def _beam_search_candidates(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        query_idx: int,
        current_graph: Dict[int, List[int]],
        navigation_node: int,
        L: int,
        C: int,
    ) -> List[int]:
        """
        Use beam search to find candidate neighbors for a node.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            query_idx: Global index of query node
            current_graph: Current state of the graph being built
            navigation_node: Global index of navigation node
            L: Search list size
            C: Maximum candidates to return

        Returns:
            List of candidate neighbor indices
        """
        query_point = henn_points[query_idx]
        visited = set()
        candidates = []

        # Start from navigation node
        if self.distance == "cosine":
            start_dist = 1 - np.dot(henn_points[navigation_node], query_point)
        else:
            start_dist = np.linalg.norm(henn_points[navigation_node] - query_point)
        candidates.append((start_dist, navigation_node))

        dynamic_candidates = []

        while candidates:
            # Get closest unvisited candidate
            candidates.sort()
            current_dist, current_idx = candidates.pop(0)

            if current_idx in visited or current_idx == query_idx:
                continue

            visited.add(current_idx)

            # Add to dynamic candidate list
            dynamic_candidates.append((current_dist, current_idx))

            # Keep only L best candidates
            if len(dynamic_candidates) > L:
                dynamic_candidates.sort()
                dynamic_candidates = dynamic_candidates[:L]

            # Explore neighbors if we haven't found enough or current is better than worst
            should_explore = len(dynamic_candidates) < L or (
                dynamic_candidates and current_dist < dynamic_candidates[-1][0]
            )

            if should_explore and current_idx in current_graph:
                for neighbor_idx in current_graph[current_idx]:
                    if neighbor_idx not in visited and neighbor_idx != query_idx:
                        if self.distance == "cosine":
                            neighbor_dist = 1 - np.dot(
                                henn_points[neighbor_idx], query_point
                            )
                        else:
                            neighbor_dist = np.linalg.norm(
                                henn_points[neighbor_idx] - query_point
                            )
                        candidates.append((neighbor_dist, neighbor_idx))

        # Return top C candidates
        dynamic_candidates.sort()
        return [idx for _, idx in dynamic_candidates[:C]]

    def _select_neighbors_nsg(
        self, henn_points: np.ndarray, node_idx: int, candidates: List[int], R: int
    ) -> List[int]:
        """
        Select R best neighbors using NSG neighbor selection strategy.
        This implements a diversity-aware selection to avoid clustering.

        Args:
            henn_points: All points in the HENN structure
            node_idx: Global index of the node
            candidates: List of candidate neighbor indices
            R: Maximum number of neighbors to select

        Returns:
            List of selected neighbor indices
        """
        if len(candidates) <= R:
            return candidates

        node_point = henn_points[node_idx]
        selected = []
        remaining = candidates.copy()

        # Calculate distances from node to all candidates
        candidate_distances = []
        for cand_idx in candidates:
            if self.distance == "cosine":
                dist = 1 - np.dot(henn_points[cand_idx], node_point)
            else:
                dist = np.linalg.norm(henn_points[cand_idx] - node_point)
            candidate_distances.append((dist, cand_idx))

        candidate_distances.sort()

        # Select neighbors using diversity strategy
        for dist, cand_idx in candidate_distances:
            if len(selected) >= R:
                break

            # Check if candidate is too close to already selected neighbors
            should_select = True
            for sel_idx in selected:
                # If distance from candidate to selected neighbor is much smaller
                # than distance from node to candidate, skip this candidate
                if self.distance == "cosine":
                    cand_to_sel_dist = 1 - np.dot(
                        henn_points[cand_idx], henn_points[sel_idx]
                    )
                else:
                    cand_to_sel_dist = np.linalg.norm(
                        henn_points[cand_idx] - henn_points[sel_idx]
                    )
                if cand_to_sel_dist < dist:
                    should_select = False
                    break

            if should_select:
                selected.append(cand_idx)

        # If we don't have enough neighbors, add closest remaining ones
        while len(selected) < R and len(selected) < len(candidates):
            for _, cand_idx in candidate_distances:
                if cand_idx not in selected:
                    selected.append(cand_idx)
                    break

        return selected[:R]

    def _ensure_connectivity(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        nsg_graph: Dict[int, List[int]],
        navigation_node: int,
    ) -> Dict[int, List[int]]:
        """
        Ensure the graph is strongly connected by adding necessary edges.
        Maintains degree constraints during connectivity enforcement.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            nsg_graph: Current NSG graph
            navigation_node: Global index of navigation node

        Returns:
            Connected NSG graph
        """
        # Find nodes that are not reachable from navigation node
        reachable = set()
        stack = [navigation_node]

        print("Checking graph connectivity...")
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)

            for neighbor in nsg_graph[current]:
                if neighbor not in reachable:
                    stack.append(neighbor)

        # For unreachable nodes, add connection to closest reachable node
        unreachable = set(layer_indices) - reachable

        for unreachable_idx in tqdm(unreachable, desc="Ensuring connectivity"):
            unreachable_point = henn_points[unreachable_idx]

            # Find closest reachable node
            min_dist = float("inf")
            closest_reachable = None

            for reachable_idx in reachable:
                if self.distance == "cosine":
                    dist = 1 - np.dot(henn_points[reachable_idx], unreachable_point)
                else:
                    dist = np.linalg.norm(
                        henn_points[reachable_idx] - unreachable_point
                    )
                if dist < min_dist:
                    min_dist = dist
                    closest_reachable = reachable_idx

            # Add connection from unreachable to closest reachable
            # (unidirectional to maintain degree constraints)
            if closest_reachable is not None:
                if closest_reachable not in nsg_graph[unreachable_idx]:
                    nsg_graph[unreachable_idx].append(closest_reachable)

                reachable.add(unreachable_idx)

        return nsg_graph
