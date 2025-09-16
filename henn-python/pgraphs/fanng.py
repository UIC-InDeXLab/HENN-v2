from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
import heapq
from tqdm import tqdm
from typing import List, Tuple, Dict, Set


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

    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ) -> Dict[int, List[int]]:
        """
        Build FANNG graph for the given points.

        Args:
            henn_points: Array of points
            layer_indices: Indices of points in this layer
            params: Parameters for FANNG construction:
                - K: Number of neighbors per node (default: 8)
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

        # Initialize adjacency list
        graph = {idx: [] for idx in layer_indices}

        # Randomize insertion order for better graph quality
        insertion_order = layer_indices.copy()
        random.shuffle(insertion_order)

        # Insert points incrementally
        inserted_nodes = []

        for i, node_idx in tqdm(
            enumerate(insertion_order), desc="Inserting nodes", total=n
        ):
            if i == 0:
                # First node has no neighbors
                inserted_nodes.append(node_idx)
                continue

            # Find candidate neighbors using search
            candidates = self._search_candidates(
                henn_points, node_idx, inserted_nodes, graph, L
            )

            # Select best neighbors using FANNG selection strategy
            selected_neighbors = self._select_neighbors_fanng(
                henn_points, node_idx, candidates, K, alpha
            )

            # Add forward links
            graph[node_idx] = selected_neighbors

            # Add reverse links (bidirectional connections)
            for neighbor_idx in selected_neighbors:
                if node_idx not in graph[neighbor_idx]:
                    graph[neighbor_idx].append(node_idx)

            # Prune neighbors that exceed degree limit
            # for neighbor_idx in selected_neighbors:
            #     if len(graph[neighbor_idx]) > K:
            #         graph[neighbor_idx] = self._prune_neighbors(
            #             henn_points, neighbor_idx, graph[neighbor_idx], K, alpha
            #         )

            # Update reverse links for existing nodes
            self._update_reverse_links(
                henn_points, node_idx, inserted_nodes, graph, R, K, alpha
            )

            inserted_nodes.append(node_idx)

        # Find best entry point for searches
        self._find_entry_point(graph, layer_indices)

        return graph

    def _search_candidates(
        self,
        henn_points: np.ndarray,
        query_idx: int,
        inserted_nodes: List[int],
        graph: Dict[int, List[int]],
        L: int,
    ) -> List[Tuple[float, int]]:
        """
        Search for candidate neighbors using greedy search.

        Returns list of (distance, node_idx) tuples.
        """
        if not inserted_nodes:
            return []

        query_point = henn_points[query_idx]

        # Start with random node
        entry_point = random.choice(inserted_nodes)

        # Dynamic candidate list (min-heap for closest candidates)
        candidates = []
        # Working set (max-heap for farthest candidates)
        working_set = []
        visited = set()

        # Initialize with entry point
        if self.distance == "cosine":
            entry_dist = 1 - np.dot(henn_points[entry_point], query_point)
        else:  # Default to L2 distance
            entry_dist = np.linalg.norm(henn_points[entry_point] - query_point)
        heapq.heappush(candidates, (entry_dist, entry_point))
        heapq.heappush(working_set, (-entry_dist, entry_point))
        visited.add(entry_point)

        while candidates:
            current_dist, current_idx = heapq.heappop(candidates)

            # If current candidate is farther than farthest in working set, stop
            if working_set and current_dist > -working_set[0][0]:
                break

            # Explore neighbors
            for neighbor_idx in graph.get(current_idx, []):
                if neighbor_idx not in visited and neighbor_idx != query_idx:
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

                    # Update working set
                    if len(working_set) < L:
                        heapq.heappush(working_set, (-neighbor_dist, neighbor_idx))
                    elif neighbor_dist < -working_set[0][0]:
                        heapq.heapreplace(working_set, (-neighbor_dist, neighbor_idx))

        # Convert working set to sorted list of candidates
        result = [(-dist, idx) for dist, idx in working_set]
        result.sort()
        return result

    def _select_neighbors_fanng(
        self,
        henn_points: np.ndarray,
        node_idx: int,
        candidates: List[Tuple[float, int]],
        K: int,
        alpha: float,
    ) -> List[int]:
        """
        Select neighbors using FANNG's diversity-based selection.

        This implements a greedy algorithm that selects neighbors to maximize
        diversity while maintaining proximity.
        """
        if len(candidates) <= K:
            return [idx for _, idx in candidates]

        node_point = henn_points[node_idx]
        selected = []

        # Convert to list for easier manipulation
        remaining_candidates = candidates.copy()

        while len(selected) < K and remaining_candidates:
            best_candidate = None
            best_score = float("inf")

            for i, (dist_to_node, candidate_idx) in enumerate(remaining_candidates):
                candidate_point = henn_points[candidate_idx]

                # Calculate diversity score
                max_similarity = 0.0
                for selected_idx in selected:
                    selected_point = henn_points[selected_idx]

                    # Distance from candidate to selected neighbor
                    if self.distance == "cosine":
                        cand_to_sel = 1 - np.dot(candidate_point, selected_point)
                    else:  # Default to L2 distance
                        cand_to_sel = np.linalg.norm(candidate_point - selected_point)

                    # Similarity score (lower distance = higher similarity)
                    if cand_to_sel > 0:
                        similarity = dist_to_node / cand_to_sel
                        max_similarity = max(max_similarity, similarity)

                # Score combines distance and diversity
                # Lower score is better
                score = dist_to_node + alpha * max_similarity

                if score < best_score:
                    best_score = score
                    best_candidate = i

            if best_candidate is not None:
                _, selected_idx = remaining_candidates.pop(best_candidate)
                selected.append(selected_idx)

        return selected

    # def _prune_neighbors(
    #     self,
    #     henn_points: np.ndarray,
    #     node_idx: int,
    #     neighbors: List[int],
    #     K: int,
    #     alpha: float,
    # ) -> List[int]:
    #     """
    #     Prune neighbors to maintain degree limit while preserving quality.
    #     """
    #     if len(neighbors) <= K:
    #         return neighbors

    #     node_point = henn_points[node_idx]

    #     # Calculate distances to all neighbors
    #     neighbor_distances = []
    #     for neighbor_idx in neighbors:
    #         dist = np.linalg.norm(henn_points[neighbor_idx] - node_point)
    #         neighbor_distances.append((dist, neighbor_idx))

    #     # Use same selection strategy as neighbor selection
    #     selected = self._select_neighbors_fanng(
    #         henn_points, node_idx, neighbor_distances, K, alpha
    #     )

    #     return selected

    def _update_reverse_links(
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
        Update reverse links for existing nodes when a new node is added.
        """
        new_point = henn_points[new_node_idx]

        # Find R closest existing nodes to consider for reverse linking
        distances = []
        for node_idx in inserted_nodes:
            if node_idx != new_node_idx:
                if self.distance == "cosine":
                    dist = 1 - np.dot(henn_points[node_idx], new_point)
                else:  # Default to L2 distance
                    dist = np.linalg.norm(henn_points[node_idx] - new_point)
                distances.append((dist, node_idx))

        distances.sort()
        close_nodes = distances[:R]

        # For each close node, consider adding new_node as neighbor
        for dist, node_idx in close_nodes:
            current_neighbors = graph[node_idx].copy()

            # Add new node as candidate
            if new_node_idx not in current_neighbors:
                current_neighbors.append(new_node_idx)

                # If exceeds degree limit, prune
                if len(current_neighbors) > K:
                    # Calculate distances for pruning
                    neighbor_distances = []
                    node_point = henn_points[node_idx]
                    for neighbor_idx in current_neighbors:
                        if self.distance == "cosine":
                            neighbor_dist = 1 - np.dot(
                                henn_points[neighbor_idx], node_point
                            )
                        else:  # Default to L2 distance
                            neighbor_dist = np.linalg.norm(
                                henn_points[neighbor_idx] - node_point
                            )
                        neighbor_distances.append((neighbor_dist, neighbor_idx))

                    # Select best neighbors
                    selected = self._select_neighbors_fanng(
                        henn_points, node_idx, neighbor_distances, K, alpha
                    )
                    graph[node_idx] = selected
                else:
                    graph[node_idx] = current_neighbors

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
