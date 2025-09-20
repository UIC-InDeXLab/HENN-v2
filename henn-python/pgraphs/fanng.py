from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
import heapq
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from numba import jit, prange
import numba


@jit(nopython=True, fastmath=True)
def compute_l2_distances_batch(query_point, points, indices):
    """Vectorized L2 distance computation."""
    n = len(indices)
    distances = np.empty(n, dtype=np.float32)
    for i in prange(n):
        idx = indices[i]
        diff = points[idx] - query_point
        distances[i] = np.sqrt(np.sum(diff * diff))
    return distances


@jit(nopython=True, fastmath=True)
def compute_cosine_distances_batch(query_point, points, indices):
    """Vectorized cosine distance computation."""
    n = len(indices)
    distances = np.empty(n, dtype=np.float32)
    query_norm = np.sqrt(np.sum(query_point * query_point))

    for i in prange(n):
        idx = indices[i]
        point = points[idx]
        dot_product = np.sum(query_point * point)
        point_norm = np.sqrt(np.sum(point * point))

        if query_norm > 0 and point_norm > 0:
            cosine_sim = dot_product / (query_norm * point_norm)
            distances[i] = 1.0 - cosine_sim
        else:
            distances[i] = 1.0
    return distances


@jit(nopython=True, fastmath=True)
def fast_neighbor_selection(
    distances, candidate_indices, points, node_point, K, alpha, distance_type
):
    """Optimized neighbor selection with diversity."""
    n_candidates = len(candidate_indices)
    if n_candidates <= K:
        return candidate_indices[:n_candidates]

    selected = np.empty(K, dtype=np.int32)
    selected_count = 0
    used = np.zeros(n_candidates, dtype=np.bool_)

    # Pre-compute all pairwise distances for candidates
    candidate_points = np.empty((n_candidates, points.shape[1]), dtype=np.float32)
    for i in range(n_candidates):
        candidate_points[i] = points[candidate_indices[i]]

    while selected_count < K:
        best_idx = -1
        best_score = np.inf

        for i in range(n_candidates):
            if used[i]:
                continue

            dist_to_node = distances[i]
            max_similarity = 0.0

            # Calculate diversity score against selected neighbors
            for j in range(selected_count):
                selected_point = points[selected[j]]

                if distance_type == 0:  # L2
                    diff = candidate_points[i] - selected_point
                    cand_to_sel = np.sqrt(np.sum(diff * diff))
                else:  # cosine
                    dot_prod = np.sum(candidate_points[i] * selected_point)
                    norm1 = np.sqrt(np.sum(candidate_points[i] * candidate_points[i]))
                    norm2 = np.sqrt(np.sum(selected_point * selected_point))
                    if norm1 > 0 and norm2 > 0:
                        cand_to_sel = 1.0 - (dot_prod / (norm1 * norm2))
                    else:
                        cand_to_sel = 1.0

                if cand_to_sel > 0:
                    similarity = dist_to_node / cand_to_sel
                    max_similarity = max(max_similarity, similarity)

            score = dist_to_node + alpha * max_similarity

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected[selected_count] = candidate_indices[best_idx]
            used[best_idx] = True
            selected_count += 1
        else:
            break

    return selected[:selected_count]


class FANNG(BaseProximityGraph):
    """
    Fast Approximate Nearest Neighbor Graph (FANNG) implementation.

    FANNG builds a sparse, navigable graph optimized for fast approximate
    nearest neighbor search. It uses incremental construction with candidate
    selection and pruning strategies to maintain graph quality.
    """

    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        super().__init__(distance, enable_logging, log_level)
        self.init_node = None
        self.distance_type = 0 if distance == "l2" else 1  # 0 for L2, 1 for cosine

    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ) -> Dict[int, List[int]]:
        """
        Build FANNG graph for the given points with optimizations.

        Args:
            henn_points: Array of points
            layer_indices: Indices of points in this layer
            params: Parameters for FANNG construction:
                - K: Number of neighbors per node (default: 16)
                - L: Candidate pool size during search (default: 32)
                - R: Number of reverse links to consider (default: 16)
                - alpha: Pruning parameter (default: 1.2)

        Returns:
            Dictionary mapping node indices to neighbor lists
        """
        if params is None:
            params = {}

        K = params.get("K", 16)  # Target degree
        L = params.get("L", 32)  # Search candidate pool size
        R = params.get("R", 16)  # Reverse link consideration
        alpha = params.get("alpha", 1.2)  # Pruning parameter

        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Convert to numpy arrays for faster access
        henn_points = np.asarray(henn_points, dtype=np.float32)
        layer_indices_array = np.array(layer_indices, dtype=np.int32)

        # Initialize adjacency list
        graph = {idx: [] for idx in layer_indices}

        # Randomize insertion order for better graph quality
        insertion_order = layer_indices.copy()
        random.shuffle(insertion_order)

        # Insert points incrementally with optimizations
        inserted_nodes = []

        # Pre-allocate arrays for efficiency
        max_candidates = min(L * 2, n)
        candidate_buffer = np.empty(max_candidates, dtype=np.int32)
        distance_buffer = np.empty(max_candidates, dtype=np.float32)

        for i, node_idx in tqdm(
            enumerate(insertion_order), desc="Inserting nodes", total=n
        ):
            if i == 0:
                # First node has no neighbors
                inserted_nodes.append(node_idx)
                continue

            # Find candidate neighbors using optimized search
            candidates = self._search_candidates_optimized(
                henn_points,
                node_idx,
                inserted_nodes,
                graph,
                L,
                candidate_buffer,
                distance_buffer,
            )

            # Select best neighbors using optimized FANNG selection
            selected_neighbors = self._select_neighbors_optimized(
                henn_points, node_idx, candidates, K, alpha
            )

            # Add forward links
            graph[node_idx] = selected_neighbors.tolist()

            # Add reverse links with batch updates
            self._add_reverse_links_batch(graph, node_idx, selected_neighbors)

            # Update reverse links for existing nodes (less frequently)
            if i % 10 == 0 or len(selected_neighbors) > K // 2:  # Adaptive frequency
                self._update_reverse_links_optimized(
                    henn_points, node_idx, inserted_nodes, graph, R, K, alpha
                )

            inserted_nodes.append(node_idx)

        # Find best entry point for searches
        self._find_entry_point(graph, layer_indices)

        return graph

    def _search_candidates_optimized(
        self,
        henn_points: np.ndarray,
        query_idx: int,
        inserted_nodes: List[int],
        graph: Dict[int, List[int]],
        L: int,
        candidate_buffer: np.ndarray,
        distance_buffer: np.ndarray,
    ) -> np.ndarray:
        """
        Optimized candidate search using vectorized distance computation.
        """
        if not inserted_nodes:
            return np.array([], dtype=np.int32)

        query_point = henn_points[query_idx]

        # Start with multiple entry points for better coverage
        n_entries = min(3, len(inserted_nodes))
        entry_points = random.sample(inserted_nodes, n_entries)

        # Use set for faster membership testing
        visited = set()
        candidates_heap = []  # min-heap for exploration
        working_set = []  # max-heap for results

        # Initialize with entry points
        for entry_point in entry_points:
            if entry_point not in visited:
                if self.distance_type == 0:  # L2
                    dist = np.linalg.norm(henn_points[entry_point] - query_point)
                else:  # cosine
                    dist = 1 - np.dot(henn_points[entry_point], query_point)

                heapq.heappush(candidates_heap, (dist, entry_point))
                heapq.heappush(working_set, (-dist, entry_point))
                visited.add(entry_point)

        # Greedy search with early termination
        iterations = 0
        max_iterations = min(len(inserted_nodes) * 2, 1000)  # Limit iterations

        while candidates_heap and iterations < max_iterations:
            current_dist, current_idx = heapq.heappop(candidates_heap)
            iterations += 1

            # Early termination condition
            if working_set and current_dist > -working_set[0][0] * 1.5:
                break

            # Explore neighbors
            neighbors = graph.get(current_idx, [])
            if len(neighbors) > 20:  # Limit neighbor exploration for high-degree nodes
                neighbors = random.sample(neighbors, 20)

            for neighbor_idx in neighbors:
                if neighbor_idx not in visited and neighbor_idx != query_idx:
                    visited.add(neighbor_idx)

                    if self.distance_type == 0:  # L2
                        neighbor_dist = np.linalg.norm(
                            henn_points[neighbor_idx] - query_point
                        )
                    else:  # cosine
                        neighbor_dist = 1 - np.dot(
                            henn_points[neighbor_idx], query_point
                        )

                    # Add to exploration queue
                    heapq.heappush(candidates_heap, (neighbor_dist, neighbor_idx))

                    # Update working set
                    if len(working_set) < L:
                        heapq.heappush(working_set, (-neighbor_dist, neighbor_idx))
                    elif neighbor_dist < -working_set[0][0]:
                        heapq.heapreplace(working_set, (-neighbor_dist, neighbor_idx))

        # Convert to sorted array
        result_size = len(working_set)
        result_indices = np.empty(result_size, dtype=np.int32)
        result_distances = np.empty(result_size, dtype=np.float32)

        for i in range(result_size):
            dist, idx = heapq.heappop(working_set)
            result_distances[i] = -dist
            result_indices[i] = idx

        # Sort by distance
        sort_order = np.argsort(result_distances)
        return result_indices[sort_order]

    def _select_neighbors_optimized(
        self,
        henn_points: np.ndarray,
        node_idx: int,
        candidate_indices: np.ndarray,
        K: int,
        alpha: float,
    ) -> np.ndarray:
        """
        Optimized neighbor selection using numba-compiled function.
        """
        if len(candidate_indices) <= K:
            return candidate_indices

        # Compute distances to all candidates
        query_point = henn_points[node_idx]
        if self.distance_type == 0:  # L2
            distances = compute_l2_distances_batch(
                query_point, henn_points, candidate_indices
            )
        else:  # cosine
            distances = compute_cosine_distances_batch(
                query_point, henn_points, candidate_indices
            )

        # Use optimized selection
        selected = fast_neighbor_selection(
            distances,
            candidate_indices,
            henn_points,
            query_point,
            K,
            alpha,
            self.distance_type,
        )

        return selected

    def _add_reverse_links_batch(
        self, graph: Dict[int, List[int]], node_idx: int, selected_neighbors: np.ndarray
    ):
        """
        Efficiently add reverse links in batch.
        """
        for neighbor_idx in selected_neighbors:
            neighbor_list = graph[neighbor_idx]
            if node_idx not in neighbor_list:
                neighbor_list.append(node_idx)

    def _update_reverse_links_optimized(
        self,
        henn_points: np.ndarray,
        new_node_idx: int,
        inserted_nodes: List[int],
        graph: Dict[int, List[int]],
        R: int,
        K: int,
        alpha: float,
    ):
        """
        Optimized reverse link updates with vectorized distance computation.
        """
        if len(inserted_nodes) <= 1:
            return

        new_point = henn_points[new_node_idx]

        # Vectorized distance computation to all inserted nodes
        inserted_indices = np.array(
            [idx for idx in inserted_nodes if idx != new_node_idx], dtype=np.int32
        )
        if len(inserted_indices) == 0:
            return

        if self.distance_type == 0:  # L2
            distances = compute_l2_distances_batch(
                new_point, henn_points, inserted_indices
            )
        else:  # cosine
            distances = compute_cosine_distances_batch(
                new_point, henn_points, inserted_indices
            )

        # Find R closest nodes efficiently
        if len(distances) <= R:
            close_indices = inserted_indices
        else:
            closest_indices = np.argpartition(distances, R)[:R]
            close_indices = inserted_indices[closest_indices]

        # Update reverse links for close nodes
        for node_idx in close_indices:
            current_neighbors = graph[node_idx]

            if new_node_idx not in current_neighbors:
                current_neighbors.append(new_node_idx)

                # Prune if necessary (less aggressive pruning)
                if len(current_neighbors) > K * 1.5:  # Allow some degree variance
                    # Quick pruning - remove farthest neighbor
                    node_point = henn_points[node_idx]
                    max_dist = -1
                    worst_neighbor = None

                    for neighbor_idx in current_neighbors:
                        if self.distance_type == 0:
                            dist = np.linalg.norm(
                                henn_points[neighbor_idx] - node_point
                            )
                        else:
                            dist = 1 - np.dot(henn_points[neighbor_idx], node_point)

                        if dist > max_dist:
                            max_dist = dist
                            worst_neighbor = neighbor_idx

                    if worst_neighbor is not None:
                        current_neighbors.remove(worst_neighbor)

    def _find_entry_point(self, graph: Dict[int, List[int]], layer_indices: list):
        """
        Find the best entry point for searches (node with highest degree).
        """
        max_degree = -1
        best_node = None

        for node_idx in layer_indices:
            degree = len(graph.get(node_idx, []))
            if degree > max_degree:
                max_degree = degree
                best_node = node_idx

        self.init_node = best_node if best_node is not None else layer_indices[0]

    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ) -> int:
        """
        Get the initial search node for this layer.

        Returns the entry point found during graph construction,
        or the highest degree node if edges are provided.
        """
        # Use stored entry point
        if self.init_node is not None and self.init_node in layer_indices:
            return self.init_node

        # Fallback to first node
        return layer_indices[0] if layer_indices else None
