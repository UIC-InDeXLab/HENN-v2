from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
import heapq
from collections import defaultdict


class KGraph(BaseProximityGraph):
    """
    KGraph implementation using NN-Descent algorithm.

    KGraph builds high-quality k-nearest neighbor graphs using the NN-Descent
    (Nearest Neighbor Descent) algorithm. This iterative algorithm starts with
    a random or approximate k-NN graph and progressively improves it by having
    nodes share information about their neighbors.

    The algorithm is particularly effective for large-scale datasets where
    brute-force k-NN construction would be too expensive.
    """

    def __init__(
        self,
        distance: str = "l2",
        enable_logging: bool = False,
        log_level: str = "INFO",
    ):
        super().__init__(distance, enable_logging, log_level)
        self.init_node = None

    def _compute_distance_vectorized(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between two sets of points using vectorized operations.

        Args:
            points1: Array of shape (n, d)
            points2: Array of shape (n, d)

        Returns:
            Array of distances of shape (n,)
        """
        if self.distance == "cosine":
            # Vectorized cosine distance
            dots = np.sum(points1 * points2, axis=1)
            return 1 - dots
        else:  # Default to L2 distance
            # Vectorized L2 distance
            diffs = points1 - points2
            return np.linalg.norm(diffs, axis=1)

    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ) -> Dict[int, List[int]]:
        """
        Build KGraph using NN-Descent algorithm.

        Args:
            henn_points: Array of points
            layer_indices: Indices of points in this layer
            params: Parameters for KGraph construction:
                - k: Number of nearest neighbors (default: 10)
                - rho: Sample rate for NN-Descent (default: 1.0)
                - delta: Early termination threshold (default: 0.001)
                - max_iterations: Maximum iterations (default: 30)
                - recall_target: Target recall for quality (default: 0.9)

        Returns:
            Dictionary mapping node indices to neighbor lists
        """
        if params is None:
            params = {}

        k = params.get("k", 16)
        rho = params.get("rho", 0.5)  # Increased sample rate for better convergence
        delta = params.get("delta", 0.001)  # Early termination threshold
        max_iterations = params.get("max_iterations", 10)

        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Limit k to maximum possible neighbors
        k = min(k, n - 1)

        # Step 1: Initialize with random k-NN graph
        print("Initializing random k-NN graph...")
        knn_graph = self._initialize_random_knn(henn_points, layer_indices, k)

        # Step 2: Run NN-Descent iterations
        print("Running NN-Descent...")
        final_graph = self._nn_descent(
            henn_points, layer_indices, knn_graph, k, rho, delta, max_iterations
        )

        # Step 3: Find best entry point
        self._find_entry_point(final_graph, layer_indices)

        return final_graph

    def _initialize_random_knn(
        self, henn_points: np.ndarray, layer_indices: list, k: int
    ) -> Dict[int, List[Tuple[float, int]]]:
        """
        Initialize with random k-NN graph using memory-efficient chunked processing.

        Returns graph with (distance, neighbor_idx) tuples for easy sorting.
        """
        n = len(layer_indices)
        knn_graph = {}

        # Process in chunks to manage memory for large datasets
        chunk_size = min(1000, max(100, n // 100))

        print(f"Initializing random k-NN graph with chunk size: {chunk_size}")

        # Convert layer_indices to numpy array for faster indexing
        layer_array = np.array(layer_indices)

        for chunk_start in tqdm(
            range(0, n, chunk_size), desc="Initializing random k-NN"
        ):
            chunk_end = min(chunk_start + chunk_size, n)

            for i in range(chunk_start, chunk_end):
                idx = layer_indices[i]

                # Get all possible neighbors (excluding self)
                available_mask = np.ones(n, dtype=bool)
                available_mask[i] = False  # Exclude self
                available_neighbors = layer_array[available_mask]

                actual_k = min(k, len(available_neighbors))

                if actual_k > 0:
                    # Randomly sample k neighbors
                    neighbor_indices = np.random.choice(
                        available_neighbors, size=actual_k, replace=False
                    )

                    # Calculate distances in smaller batches to avoid memory issues
                    neighbor_list = []
                    batch_size = min(100, len(neighbor_indices))

                    for batch_start in range(0, len(neighbor_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(neighbor_indices))
                        batch_neighbors = neighbor_indices[batch_start:batch_end]

                        # Vectorized distance calculation for batch
                        node_point = henn_points[
                            idx : idx + 1
                        ]  # Keep 2D shape for broadcasting
                        neighbor_points = henn_points[batch_neighbors]

                        # Use vectorized distance computation
                        distances = self._compute_distance_vectorized(
                            np.tile(node_point, (len(batch_neighbors), 1)),
                            neighbor_points,
                        )

                        # Add to neighbor list
                        batch_neighbors_with_dist = [
                            (dist, neighbor_idx)
                            for dist, neighbor_idx in zip(distances, batch_neighbors)
                        ]
                        neighbor_list.extend(batch_neighbors_with_dist)

                    neighbor_list.sort()
                    knn_graph[idx] = neighbor_list
                else:
                    knn_graph[idx] = []

        return knn_graph

    def _nn_descent(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        knn_graph: Dict[int, List[Tuple[float, int]]],
        k: int,
        rho: float,
        delta: float,
        max_iterations: int,
    ) -> Dict[int, List[int]]:
        """
        Memory-efficient NN-Descent algorithm to iteratively improve the k-NN graph.
        Processes nodes in chunks to avoid memory overflow with large datasets.
        """
        n = len(layer_indices)
        updates_count = 0

        # Memory management parameters
        chunk_size = min(1000, max(100, n // 100))  # Adaptive chunk size
        distance_batch_size = 10000  # Limit concurrent distance calculations

        print(
            f"Using chunk size: {chunk_size}, distance batch size: {distance_batch_size}"
        )

        # Convert to more efficient data structures
        # Use max heaps for efficient neighbor management
        neighbor_heaps = {}
        for node_idx in layer_indices:
            # Convert to max heap (negate distances for max heap behavior)
            heap = [(-dist, neighbor_idx) for dist, neighbor_idx in knn_graph[node_idx]]
            heapq.heapify(heap)
            neighbor_heaps[node_idx] = heap

        # Track which neighbors are new vs old for better sampling
        new_flags = {
            node_idx: [True] * len(knn_graph[node_idx]) for node_idx in layer_indices
        }

        for iteration in tqdm(range(max_iterations), desc="Running NN-Descent"):
            iteration_updates = 0

            # Process nodes in chunks to manage memory
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                chunk_indices = layer_indices[chunk_start:chunk_end]

                # Collect candidate pairs for this chunk only
                chunk_candidate_pairs = []

                # Sample neighbors more efficiently for chunk
                for node_idx in chunk_indices:
                    current_neighbors = neighbor_heaps[node_idx]
                    node_new_flags = new_flags[node_idx]

                    # Limit sampling to reduce memory usage
                    max_sample_size = min(
                        10, len(current_neighbors)
                    )  # Reduced from rho-based calculation

                    # Sample new neighbors (higher probability)
                    new_sample_count = min(
                        len([f for f in node_new_flags if f]),
                        max(1, max_sample_size // 2),
                    )

                    # Sample old neighbors (lower probability)
                    old_sample_count = min(
                        len([f for f in node_new_flags if not f]),
                        max(1, max_sample_size // 4),
                    )

                    sampled_neighbors = []

                    # Get new neighbors first
                    new_count = 0
                    old_count = 0
                    for i, (neg_dist, neighbor_idx) in enumerate(current_neighbors):
                        if (
                            new_count < new_sample_count
                            and i < len(node_new_flags)
                            and node_new_flags[i]
                        ):
                            sampled_neighbors.append(neighbor_idx)
                            new_count += 1
                        elif (
                            old_count < old_sample_count
                            and i < len(node_new_flags)
                            and not node_new_flags[i]
                        ):
                            sampled_neighbors.append(neighbor_idx)
                            old_count += 1

                    # Generate candidate pairs from sampled neighbors (limited)
                    for i, neighbor1 in enumerate(sampled_neighbors):
                        # Limit to avoid quadratic explosion
                        for j, neighbor2 in enumerate(
                            sampled_neighbors[i + 1 : i + 6], i + 1
                        ):  # Only check next 5
                            if neighbor1 != neighbor2:
                                chunk_candidate_pairs.append((neighbor1, neighbor2))
                                chunk_candidate_pairs.append((neighbor2, neighbor1))

                        # Add limited neighbors of neighbors
                        neighbor_heap = neighbor_heaps.get(neighbor1, [])
                        for neg_dist, candidate_idx in neighbor_heap[
                            :3
                        ]:  # Reduced from 5
                            if (
                                candidate_idx != node_idx
                                and candidate_idx not in sampled_neighbors
                            ):
                                chunk_candidate_pairs.append((node_idx, candidate_idx))

                # Process distance calculations in smaller batches
                chunk_updates = 0
                for batch_start in range(
                    0, len(chunk_candidate_pairs), distance_batch_size
                ):
                    batch_end = min(
                        batch_start + distance_batch_size, len(chunk_candidate_pairs)
                    )
                    batch_pairs = chunk_candidate_pairs[batch_start:batch_end]

                    if not batch_pairs:
                        continue

                    # Calculate distances for this batch
                    candidate_distances = {}

                    try:
                        nodes1, nodes2 = zip(*batch_pairs)
                        points1 = henn_points[list(nodes1)]
                        points2 = henn_points[list(nodes2)]

                        if self.distance == "cosine":
                            # Vectorized cosine distance
                            dots = np.sum(points1 * points2, axis=1)
                            distances = 1 - dots
                        else:
                            # Vectorized L2 distance
                            diffs = points1 - points2
                            distances = np.linalg.norm(diffs, axis=1)

                        for i, (node1, node2) in enumerate(batch_pairs):
                            candidate_distances[(node1, node2)] = distances[i]

                    except MemoryError:
                        # Fallback to individual calculations if batch is too large
                        print(
                            "Memory error in batch processing, falling back to individual calculations"
                        )
                        for node1, node2 in batch_pairs:
                            point1 = henn_points[node1 : node1 + 1]
                            point2 = henn_points[node2 : node2 + 1]

                            if self.distance == "cosine":
                                dot = np.sum(point1 * point2)
                                dist = 1 - dot
                            else:
                                diff = point1 - point2
                                dist = np.linalg.norm(diff)

                            candidate_distances[(node1, node2)] = dist

                    # Update neighbor lists for nodes in current chunk
                    for node_idx in chunk_indices:
                        current_heap = neighbor_heaps[node_idx]
                        heap_changed = False

                        # Check relevant candidates for this node from current batch
                        for (
                            cand_node1,
                            cand_node2,
                        ), candidate_dist in candidate_distances.items():
                            if cand_node1 != node_idx:
                                continue

                            candidate_idx = cand_node2
                            if candidate_idx == node_idx:
                                continue

                            # Check if candidate is already a neighbor
                            existing_neighbors = {
                                neighbor_idx for _, neighbor_idx in current_heap
                            }
                            if candidate_idx in existing_neighbors:
                                continue

                            # Update heap if we have room or candidate is better than worst
                            if len(current_heap) < k:
                                heapq.heappush(
                                    current_heap, (-candidate_dist, candidate_idx)
                                )
                                heap_changed = True
                                chunk_updates += 1
                            elif (
                                len(current_heap) > 0
                                and -candidate_dist > current_heap[0][0]
                            ):
                                # Replace worst neighbor (heap[0] is max in max-heap)
                                heapq.heapreplace(
                                    current_heap, (-candidate_dist, candidate_idx)
                                )
                                heap_changed = True
                                chunk_updates += 1

                        # Update new flags - mark all as old after processing
                        if heap_changed:
                            new_flags[node_idx] = [True] * len(current_heap)
                        else:
                            new_flags[node_idx] = [False] * len(current_heap)

                    # Clear batch data to free memory
                    del candidate_distances

                iteration_updates += chunk_updates

                # Clear chunk data to free memory
                del chunk_candidate_pairs

            # Check for convergence
            if iteration_updates == 0:
                print(f"NN-Descent converged after {iteration + 1} iterations")
                break

            updates_count += iteration_updates

            # Early termination based on update rate
            if iteration > 0 and iteration_updates < delta * n * k:
                print(f"NN-Descent early termination after {iteration + 1} iterations")
                break

        # Convert back to simple adjacency list format
        final_graph = {}
        for node_idx, heap in neighbor_heaps.items():
            # Sort by distance (convert back from negative distances)
            neighbors = sorted(
                [(-neg_dist, neighbor_idx) for neg_dist, neighbor_idx in heap]
            )
            final_graph[node_idx] = [neighbor_idx for _, neighbor_idx in neighbors[:k]]

        print(f"NN-Descent completed with {updates_count} total updates")
        return final_graph

    def _find_entry_point(self, graph: Dict[int, List[int]], layer_indices: list):
        """
        Find the best entry point for searches.

        For KGraph, we choose the node with highest degree (most connections).
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
