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

    Medoid Finding Optimization:
    - "original": O(n²) implementation with nested loops (default, most accurate)
    - "vectorized": Vectorized NumPy operations (much faster for medium/large datasets)
    - "sampling": Sampling-based approximation (fastest for very large datasets)

    Connectivity Optimization:
    - "original": Original O(n²) connectivity ensuring (for small datasets)
    - "vectorized": Vectorized batch processing (default, good for most datasets)
    - "approximate": Sampling-based approximation (fastest for very large datasets >10k points)
    """

    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
        medoid_method: str = "original",
        medoid_sample_size: int = 1000,
        connectivity_optimization: str = "vectorized",
    ):
        super().__init__(distance, enable_logging, log_level)
        self.init_node = None
        self.medoid_method = medoid_method  # "original", "vectorized", "sampling"
        self.medoid_sample_size = medoid_sample_size
        self.connectivity_optimization = (
            connectivity_optimization  # "original", "vectorized", "approximate"
        )

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
        R = params.get("k", 16)  # Maximum out-degree
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
        final_graph = self._ensure_connectivity_optimized(
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

        Uses different optimization strategies based on self.medoid_method:
        - "original": Original O(n²) implementation (default)
        - "vectorized": Vectorized NumPy operations for speed
        - "sampling": Sampling-based approximation for large datasets

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer

        Returns:
            Global index of the medoid point
        """
        if len(layer_indices) == 1:
            return layer_indices[0]

        if self.medoid_method == "vectorized":
            return self._find_medoid_vectorized(henn_points, layer_indices)
        elif self.medoid_method == "sampling":
            return self._find_medoid_sampling(henn_points, layer_indices)
        else:
            return self._find_medoid_original(henn_points, layer_indices)

    def _find_medoid_original(
        self, henn_points: np.ndarray, layer_indices: list
    ) -> int:
        """
        Original medoid finding implementation (O(n²)).

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer

        Returns:
            Global index of the medoid point
        """
        min_sum_dist = float("inf")
        medoid_idx = layer_indices[0]

        # Calculate sum of distances for each point to all other points
        for i, idx_i in tqdm(
            enumerate(layer_indices),
            desc="Finding medoid (original)",
            total=len(layer_indices),
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

    def _find_medoid_vectorized(
        self, henn_points: np.ndarray, layer_indices: list
    ) -> int:
        """
        Vectorized medoid finding implementation using NumPy operations.
        Much faster than original for larger datasets.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer

        Returns:
            Global index of the medoid point
        """
        # Extract points for this layer
        layer_points = henn_points[layer_indices]  # Shape: (n, d)
        n = len(layer_indices)

        if self.distance == "cosine":
            # Normalize points for cosine distance
            # layer_points_norm = layer_points / np.linalg.norm(layer_points, axis=1, keepdims=True)

            # Compute cosine similarity matrix
            sim_matrix = np.dot(layer_points, layer_points.T)

            # Convert to cosine distance matrix
            dist_matrix = 1 - sim_matrix

        else:  # L2 distance
            # Compute pairwise L2 distances using broadcasting
            # ||a - b||² = ||a||² + ||b||² - 2*a·b
            squared_norms = np.sum(layer_points**2, axis=1)
            dot_products = np.dot(layer_points, layer_points.T)

            # Broadcasting to compute distance matrix
            dist_matrix = (
                squared_norms[:, np.newaxis]
                + squared_norms[np.newaxis, :]
                - 2 * dot_products
            )
            dist_matrix = np.sqrt(
                np.maximum(dist_matrix, 0)
            )  # Ensure non-negative due to numerical errors

        # Set diagonal to 0 (distance from point to itself)
        np.fill_diagonal(dist_matrix, 0)

        # Sum distances for each point
        sum_distances = np.sum(dist_matrix, axis=1)

        # Find medoid (point with minimum sum of distances)
        medoid_local_idx = np.argmin(sum_distances)
        medoid_global_idx = layer_indices[medoid_local_idx]

        return medoid_global_idx

    def _find_medoid_sampling(
        self, henn_points: np.ndarray, layer_indices: list
    ) -> int:
        """
        Sampling-based approximation for medoid finding.
        Useful for very large datasets where exact computation is too slow.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer

        Returns:
            Global index of the approximate medoid point
        """
        n = len(layer_indices)

        # If dataset is small, use vectorized method
        if n <= self.medoid_sample_size:
            return self._find_medoid_vectorized(henn_points, layer_indices)

        # Sample candidate points
        sample_size = min(self.medoid_sample_size, n)
        sampled_indices = np.random.choice(
            layer_indices, size=sample_size, replace=False
        )

        # For each sampled candidate, compute average distance to a subset of all points
        eval_sample_size = min(500, n)  # Sample points to evaluate against
        eval_indices = np.random.choice(
            layer_indices, size=eval_sample_size, replace=False
        )

        min_avg_dist = float("inf")
        best_candidate = sampled_indices[0]

        for candidate_idx in tqdm(sampled_indices, desc="Finding medoid (sampling)"):
            candidate_point = henn_points[candidate_idx]
            total_dist = 0.0

            for eval_idx in eval_indices:
                if candidate_idx != eval_idx:
                    eval_point = henn_points[eval_idx]
                    if self.distance == "cosine":
                        dist = 1 - np.dot(candidate_point, eval_point)
                    else:
                        dist = np.linalg.norm(candidate_point - eval_point)
                    total_dist += dist

            avg_dist = (
                total_dist / (len(eval_indices) - 1)
                if candidate_idx in eval_indices
                else total_dist / len(eval_indices)
            )

            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_candidate = candidate_idx

        return best_candidate

    def _build_initial_knn_graph(
        self, henn_points: np.ndarray, layer_indices: list, k: int
    ) -> Dict[int, List[int]]:
        """
        Build initial k-NN graph as starting point for NSG construction.
        Uses memory-efficient chunked processing similar to knn.py.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            k: Number of nearest neighbors

        Returns:
            Dictionary mapping global indices to lists of k nearest neighbors
        """
        chunk_size = 1000  # Process in chunks to save memory
        layer_points = henn_points[layer_indices]
        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Limit k to maximum possible neighbors
        k = min(k, n - 1)

        print(
            f"Building initial k-NN graph with memory-efficient chunked processing for {n} points..."
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

        knn_graph = {}

        # Process points in chunks to avoid memory overflow
        for chunk_start in tqdm(
            range(0, n, chunk_size), desc="Processing chunks for initial k-NN"
        ):
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
                knn_graph[global_idx] = [
                    layer_indices[j] for j in neighbor_local_indices
                ]

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
        Optimized version using vectorized operations for large datasets.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            nsg_graph: Current NSG graph
            navigation_node: Global index of navigation node

        Returns:
            Connected NSG graph
        """
        # Find nodes that are not reachable from navigation node using BFS
        reachable = set()
        queue = [navigation_node]
        reachable.add(navigation_node)

        print("Checking graph connectivity...")
        while queue:
            current = queue.pop(0)

            for neighbor in nsg_graph[current]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        # Find unreachable nodes
        unreachable_nodes = [idx for idx in layer_indices if idx not in reachable]

        if not unreachable_nodes:
            return nsg_graph  # Already connected

        print(
            f"Found {len(unreachable_nodes)} unreachable nodes out of {len(layer_indices)}"
        )

        # Convert to numpy arrays for vectorized operations
        reachable_list = list(reachable)
        reachable_points = henn_points[reachable_list]
        unreachable_points = henn_points[unreachable_nodes]

        # Process unreachable nodes in batches for memory efficiency
        batch_size = min(1000, len(unreachable_nodes))

        for batch_start in tqdm(
            range(0, len(unreachable_nodes), batch_size),
            desc="Ensuring connectivity (batched)",
        ):
            batch_end = min(batch_start + batch_size, len(unreachable_nodes))
            batch_unreachable = unreachable_nodes[batch_start:batch_end]
            batch_points = unreachable_points[batch_start:batch_end]

            # Vectorized distance computation for this batch
            if self.distance == "cosine":
                # Normalize points for cosine distance
                batch_norm = np.linalg.norm(batch_points, axis=1, keepdims=True)
                batch_norm[batch_norm == 0] = 1
                batch_normalized = batch_points / batch_norm

                reachable_norm = np.linalg.norm(reachable_points, axis=1, keepdims=True)
                reachable_norm[reachable_norm == 0] = 1
                reachable_normalized = reachable_points / reachable_norm

                # Compute cosine similarities: (batch_size, num_reachable)
                similarities = batch_normalized @ reachable_normalized.T
                # Convert to distances
                distances = 1 - similarities

            else:  # L2 distance
                # Compute pairwise L2 distances using broadcasting
                # Shape: (batch_size, 1, dim) and (1, num_reachable, dim)
                batch_expanded = batch_points[:, np.newaxis, :]
                reachable_expanded = reachable_points[np.newaxis, :, :]

                # Compute squared differences and sum over last dimension
                diff_squared = np.sum(
                    (batch_expanded - reachable_expanded) ** 2, axis=2
                )
                distances = np.sqrt(diff_squared)

            # Find closest reachable node for each unreachable node in batch
            closest_indices = np.argmin(distances, axis=1)

            # Add connections
            for i, unreachable_idx in enumerate(batch_unreachable):
                closest_reachable_idx = reachable_list[closest_indices[i]]

                # Add bidirectional connection for better connectivity
                if closest_reachable_idx not in nsg_graph[unreachable_idx]:
                    nsg_graph[unreachable_idx].append(closest_reachable_idx)

                # Optionally add reverse connection if it doesn't exceed degree limit
                # This helps with connectivity but maintains sparsity
                max_degree = 32  # Conservative limit to maintain sparsity
                if (
                    len(nsg_graph[closest_reachable_idx]) < max_degree
                    and unreachable_idx not in nsg_graph[closest_reachable_idx]
                ):
                    nsg_graph[closest_reachable_idx].append(unreachable_idx)

                # Add to reachable set for subsequent batches
                reachable.add(unreachable_idx)
                reachable_list.append(unreachable_idx)

            # Update reachable_points for next batch if there are more batches
            if batch_end < len(unreachable_nodes):
                reachable_points = henn_points[reachable_list]

        return nsg_graph

    def _ensure_connectivity_optimized(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        nsg_graph: Dict[int, List[int]],
        navigation_node: int,
    ) -> Dict[int, List[int]]:
        """
        Optimized connectivity ensuring that chooses the best strategy based on dataset size.

        Args:
            henn_points: All points in the HENN structure
            layer_indices: List of global indices for points in this layer
            nsg_graph: Current NSG graph
            navigation_node: Global index of navigation node

        Returns:
            Connected NSG graph
        """
        n = len(layer_indices)

        # Choose strategy based on dataset size and optimization setting
        if self.connectivity_optimization == "original" or n < 1000:
            return self._ensure_connectivity_original(
                henn_points, layer_indices, nsg_graph, navigation_node
            )
        elif self.connectivity_optimization == "approximate" and n > 10000:
            return self._ensure_connectivity_approximate(
                henn_points, layer_indices, nsg_graph, navigation_node
            )
        else:
            return self._ensure_connectivity(
                henn_points, layer_indices, nsg_graph, navigation_node
            )

    def _ensure_connectivity_original(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        nsg_graph: Dict[int, List[int]],
        navigation_node: int,
    ) -> Dict[int, List[int]]:
        """
        Original connectivity ensuring method (kept for compatibility).
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
            if closest_reachable is not None:
                if closest_reachable not in nsg_graph[unreachable_idx]:
                    nsg_graph[unreachable_idx].append(closest_reachable)

                reachable.add(unreachable_idx)

        return nsg_graph

    def _ensure_connectivity_approximate(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        nsg_graph: Dict[int, List[int]],
        navigation_node: int,
    ) -> Dict[int, List[int]]:
        """
        Approximate connectivity ensuring for very large datasets.
        Uses sampling and hierarchical approaches to reduce complexity.
        """
        # Find nodes that are not reachable from navigation node using BFS
        reachable = set()
        queue = [navigation_node]
        reachable.add(navigation_node)

        print("Checking graph connectivity...")
        while queue:
            current = queue.pop(0)

            for neighbor in nsg_graph[current]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        # Find unreachable nodes
        unreachable_nodes = [idx for idx in layer_indices if idx not in reachable]

        if not unreachable_nodes:
            return nsg_graph

        print(
            f"Found {len(unreachable_nodes)} unreachable nodes out of {len(layer_indices)}"
        )

        # For very large datasets, use sampling-based approach
        reachable_list = list(reachable)

        # Sample representative points from reachable set for efficiency
        max_reachable_sample = min(2000, len(reachable_list))
        if len(reachable_list) > max_reachable_sample:
            sampled_reachable = np.random.choice(
                reachable_list, size=max_reachable_sample, replace=False
            ).tolist()
        else:
            sampled_reachable = reachable_list

        # Process unreachable nodes in large batches
        batch_size = min(2000, len(unreachable_nodes))

        for batch_start in tqdm(
            range(0, len(unreachable_nodes), batch_size),
            desc="Ensuring connectivity (approximate)",
        ):
            batch_end = min(batch_start + batch_size, len(unreachable_nodes))
            batch_unreachable = unreachable_nodes[batch_start:batch_end]

            # Use sampled reachable points for distance computation
            self._connect_batch_to_reachable(
                henn_points, batch_unreachable, sampled_reachable, nsg_graph
            )

            # Add newly connected nodes to reachable set
            reachable.update(batch_unreachable)

        return nsg_graph

    def _connect_batch_to_reachable(
        self,
        henn_points: np.ndarray,
        batch_unreachable: List[int],
        sampled_reachable: List[int],
        nsg_graph: Dict[int, List[int]],
    ):
        """
        Connect a batch of unreachable nodes to sampled reachable nodes efficiently.
        """
        batch_points = henn_points[batch_unreachable]
        reachable_points = henn_points[sampled_reachable]

        # Vectorized distance computation
        if self.distance == "cosine":
            # Normalize points
            batch_norm = np.linalg.norm(batch_points, axis=1, keepdims=True)
            batch_norm[batch_norm == 0] = 1
            batch_normalized = batch_points / batch_norm

            reachable_norm = np.linalg.norm(reachable_points, axis=1, keepdims=True)
            reachable_norm[reachable_norm == 0] = 1
            reachable_normalized = reachable_points / reachable_norm

            # Compute cosine similarities
            similarities = batch_normalized @ reachable_normalized.T
            distances = 1 - similarities

        else:  # L2 distance
            # For very large batches, use chunked computation to save memory
            chunk_size = 500
            distances = np.zeros((len(batch_unreachable), len(sampled_reachable)))

            for i in range(0, len(batch_unreachable), chunk_size):
                end_i = min(i + chunk_size, len(batch_unreachable))
                chunk_batch = batch_points[i:end_i]

                # Compute distances for this chunk
                chunk_expanded = chunk_batch[:, np.newaxis, :]
                reachable_expanded = reachable_points[np.newaxis, :, :]
                chunk_distances = np.sqrt(
                    np.sum((chunk_expanded - reachable_expanded) ** 2, axis=2)
                )
                distances[i:end_i] = chunk_distances

        # Find closest reachable node for each unreachable node
        closest_indices = np.argmin(distances, axis=1)

        # Add connections
        for i, unreachable_idx in enumerate(batch_unreachable):
            closest_reachable_idx = sampled_reachable[closest_indices[i]]

            if closest_reachable_idx not in nsg_graph[unreachable_idx]:
                nsg_graph[unreachable_idx].append(closest_reachable_idx)
