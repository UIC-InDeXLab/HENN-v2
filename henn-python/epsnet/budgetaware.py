import numpy as np
import math
import random
from tqdm import tqdm
from typing import List, Set, Tuple
from epsnet.base_epsnet import BaseEPSNet


class BudgetAware(BaseEPSNet):
    """
    Budget-aware EPSNet construction algorithm.

    This algorithm builds an epsilon-net by:
    1. Starting with a random subset of size N = get_size(n, eps, d) - budget
    2. For B (budget) steps:
        - Select a diverse subset of 1/eps (or n/size) points
        - Create balls around each point containing eps*n points
        - Find unhit balls and add points from them
    """

    def __init__(self, distance="l2", random_seed=None, fast_diversity=False):
        self.distance = distance
        self.random_seed = random_seed
        self.fast_diversity = (
            fast_diversity  # Use faster but less accurate diversity selection
        )

    def build_epsnet(self, points: np.ndarray, eps=None, size=None, budget=None):
        """
        Build eps-net using budget-aware algorithm.

        Args:
            points: Input points array (n, d)
            eps: Epsilon parameter
            size: Target size of the eps-net (if provided, overrides eps calculation)
            budget: Number of improvement steps (default: size // 10)

        Returns:
            List of indices representing the eps-net
        """
        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        n, d = points.shape

        # Determine target size
        if size is None:
            if eps is None:
                raise ValueError("Either eps or size must be provided")
            size = BaseEPSNet.get_size(n, eps, d)

        # Estimate eps if not provided but size is given
        if eps is None:
            eps = size / n  # Simple estimation

        size = min(size, n)

        if budget is None:
            budget = max(1, size)  # Default budget is 100% of size
            
        print(f"Budget-aware epsnet: budget={budget}, netsize={size}, n={n}")
        
        if size == n:
            return list(range(n))

        # Initial sample size N = size - budget
        initial_n = max(1, size - budget)

        if initial_n >= n:
            return list(range(n))

        # Step 1: Start with random subset of size N
        current_sample = set(np.random.choice(n, size=initial_n, replace=False))

        # Step 2: Budget-aware improvement
        for step in tqdm(range(budget)):
            # Find unhit balls and add points from them
            new_points = self._find_unhit_balls(points, current_sample, eps, size)

            # Add all new points to current sample (up to the size limit)
            if new_points:
                for new_point in new_points:
                    if len(current_sample) >= size:
                        break  # Stop if we've reached the target size
                    current_sample.add(new_point)

        # Ensure we have exactly 'size' points
        # If we couldn't find enough through budget process, add random points
        remaining_points = set(range(n)) - current_sample
        while len(current_sample) < size and remaining_points:
            random_point = random.choice(list(remaining_points))
            current_sample.add(random_point)
            remaining_points.remove(random_point)

        return list(current_sample)

    def _find_unhit_balls(
        self,
        points: np.ndarray,
        current_sample: Set[int],
        eps: float,
        size: int,
    ) -> List[int]:
        """
        Find points from unhit balls using the new diverse sampling approach.

        Steps:
        1. Randomly select a diverse subset of 1/eps (or n/size) points
        2. Create balls around each point containing eps*n points
        3. Find unhit balls and add points from them

        Args:
            points: All points
            current_sample: Current sample indices
            eps: Epsilon parameter
            size: Target size of eps-net

        Returns:
            List of indices from unhit balls
        """
        n = points.shape[0]

        # Calculate number of diverse points to sample
        # Use 1/eps if available, otherwise use n/size
        if eps > 0:
            num_diverse_points = min(n, max(1, int(1.0 / eps)))
        else:
            num_diverse_points = min(n, max(1, n // size))

        # Select diverse subset using k-means++ initialization strategy
        if self.fast_diversity:
            diverse_indices = self._select_diverse_points_fast(
                points, num_diverse_points
            )
        else:
            diverse_indices = self._select_diverse_points(points, num_diverse_points)

        # Calculate ball size (eps * n points per ball)
        ball_size = max(1, int(eps * n))

        # Create balls around each diverse point
        unhit_points = []
        for center_idx in diverse_indices:
            # Find ball_size nearest points to this center
            ball_points = self._find_ball_points(points, center_idx, ball_size)

            # Check if this ball is unhit (no points from current sample in the ball)
            is_hit = any(idx in current_sample for idx in ball_points)

            if not is_hit and ball_points:
                # Choose one random point from this unhit ball
                random_point = random.choice(ball_points)
                unhit_points.append(random_point)

        return unhit_points

    def _select_diverse_points(self, points: np.ndarray, k: int) -> List[int]:
        """
        Select k diverse points using an optimized k-means++ strategy.

        This version uses vectorized operations and sampling for better performance
        while maintaining good diversity properties.

        Args:
            points: All points array (n, d)
            k: Number of diverse points to select

        Returns:
            List of indices of selected diverse points
        """
        n = points.shape[0]
        if k >= n:
            return list(range(n))

        selected_indices = []

        # Step 1: Choose first point randomly
        first_idx = random.randint(0, n - 1)
        selected_indices.append(first_idx)

        # Pre-compute squared norms for cosine distance optimization
        if self.distance == "cosine":
            # For cosine distance, we can use dot products more efficiently
            points_normalized = points / (
                np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
            )
            selected_points = points_normalized[selected_indices]
        else:
            selected_points = points[selected_indices]

        # Step 2: Choose remaining points with optimized distance calculations
        for step in range(k - 1):
            if self.distance == "cosine":
                # Vectorized cosine distance calculation
                # Distance = 1 - dot_product for normalized vectors
                similarities = np.dot(points_normalized, selected_points.T)
                distances = 1 - similarities
                min_distances_sq = np.min(distances, axis=1) ** 2
            else:
                # Vectorized L2 distance calculation
                # Broadcasting: (n, 1, d) - (1, num_selected, d) -> (n, num_selected, d)
                diffs = points[:, np.newaxis, :] - selected_points[np.newaxis, :, :]
                distances_sq = np.sum(diffs**2, axis=2)
                min_distances_sq = np.min(distances_sq, axis=1)

            # Zero out distances for already selected points
            min_distances_sq[selected_indices] = 0.0

            # Early termination: if we have many candidates, sample from top candidates
            # This maintains diversity while being much faster
            if n > 10000 and step > k // 4:
                # For large datasets and after selecting some diverse points,
                # sample from the top 10% most distant points
                num_candidates = max(100, n // 10)
                top_indices = np.argpartition(min_distances_sq, -num_candidates)[
                    -num_candidates:
                ]
                candidate_distances = min_distances_sq[top_indices]

                # Weighted random selection from candidates
                if np.sum(candidate_distances) > 0:
                    probabilities = candidate_distances / np.sum(candidate_distances)
                    chosen_candidate_idx = np.random.choice(
                        len(top_indices), p=probabilities
                    )
                    next_idx = top_indices[chosen_candidate_idx]
                else:
                    # Fallback to random selection from candidates
                    remaining = [i for i in range(n) if i not in selected_indices]
                    if remaining:
                        next_idx = random.choice(remaining)
                    else:
                        break
            else:
                # Standard k-means++ selection for smaller datasets or initial selections
                total_dist_sq = np.sum(min_distances_sq)
                if total_dist_sq == 0:
                    # All remaining points are identical to selected ones
                    remaining = [i for i in range(n) if i not in selected_indices]
                    if remaining:
                        next_idx = random.choice(remaining)
                    else:
                        break
                else:
                    probabilities = min_distances_sq / total_dist_sq
                    next_idx = np.random.choice(n, p=probabilities)

            selected_indices.append(next_idx)

            # Update selected points for next iteration
            if self.distance == "cosine":
                selected_points = points_normalized[selected_indices]
            else:
                selected_points = points[selected_indices]

        return selected_indices

    def _select_diverse_points_fast(self, points: np.ndarray, k: int) -> List[int]:
        """
        Fast diverse point selection using approximate methods.

        This method is significantly faster than the standard k-means++ approach
        but may not achieve optimal diversity. Good for when speed is more important
        than perfect diversity.

        Args:
            points: All points array (n, d)
            k: Number of diverse points to select

        Returns:
            List of indices of selected diverse points
        """
        n = points.shape[0]
        if k >= n:
            return list(range(n))

        # Strategy 1: For very large datasets, use grid-based sampling
        if n > 50000:
            return self._grid_based_sampling(points, k)

        # Strategy 2: Random sampling with distance filtering
        # Sample more candidates than needed, then filter by distance
        oversample_factor = min(5, max(2, n // k))
        num_candidates = min(n, k * oversample_factor)

        # Random sample of candidates
        candidate_indices = np.random.choice(n, size=num_candidates, replace=False)
        candidate_points = points[candidate_indices]

        # Select diverse points from candidates using greedy approach
        selected_local_indices = [0]  # Start with first candidate

        for _ in range(min(k - 1, len(candidate_indices) - 1)):
            selected_points = candidate_points[selected_local_indices]

            # Calculate distances from all candidates to selected points
            if self.distance == "cosine":
                # Normalize for cosine distance
                candidates_norm = candidate_points / (
                    np.linalg.norm(candidate_points, axis=1, keepdims=True) + 1e-8
                )
                selected_norm = selected_points / (
                    np.linalg.norm(selected_points, axis=1, keepdims=True) + 1e-8
                )
                similarities = np.dot(candidates_norm, selected_norm.T)
                distances = 1 - similarities
                min_distances = np.min(distances, axis=1)
            else:
                # L2 distance
                diffs = (
                    candidate_points[:, np.newaxis, :]
                    - selected_points[np.newaxis, :, :]
                )
                distances_sq = np.sum(diffs**2, axis=2)
                min_distances = np.sqrt(np.min(distances_sq, axis=1))

            # Zero out already selected points
            min_distances[selected_local_indices] = 0

            # Select the point with maximum minimum distance (farthest first)
            next_local_idx = np.argmax(min_distances)
            selected_local_indices.append(next_local_idx)

        # Convert back to original indices
        return [candidate_indices[i] for i in selected_local_indices]

    def _grid_based_sampling(self, points: np.ndarray, k: int) -> List[int]:
        """
        Grid-based sampling for very large datasets.

        Divides space into grid cells and samples from different cells.
        """
        n, d = points.shape

        # Determine grid size - aim for roughly sqrt(k) cells per dimension
        cells_per_dim = max(2, int(np.ceil(k ** (1 / d))))

        # Find min/max bounds for each dimension
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        ranges = maxs - mins

        # Avoid division by zero
        ranges = np.where(ranges == 0, 1, ranges)

        # Assign points to grid cells
        cell_coords = np.floor((points - mins) / ranges * cells_per_dim).astype(int)
        cell_coords = np.clip(cell_coords, 0, cells_per_dim - 1)

        # Create cell identifiers
        cell_ids = np.sum(cell_coords * (cells_per_dim ** np.arange(d)), axis=1)

        # Sample one point from each unique cell, prioritizing cells with more points
        unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)

        # Sort cells by count (descending) to prioritize denser regions first
        sorted_indices = np.argsort(-cell_counts)
        selected_indices = []

        for cell_idx in unique_cells[sorted_indices]:
            if len(selected_indices) >= k:
                break

            # Find all points in this cell
            cell_points = np.where(cell_ids == cell_idx)[0]

            # Select a random point from this cell
            selected_point = np.random.choice(cell_points)
            selected_indices.append(selected_point)

        # If we don't have enough points, fill with random selection
        if len(selected_indices) < k:
            remaining_points = [i for i in range(n) if i not in selected_indices]
            additional_needed = k - len(selected_indices)
            if remaining_points:
                additional = np.random.choice(
                    remaining_points,
                    size=min(additional_needed, len(remaining_points)),
                    replace=False,
                )
                selected_indices.extend(additional)

        return selected_indices[:k]

    def _find_ball_points(
        self, points: np.ndarray, center_idx: int, ball_size: int
    ) -> List[int]:
        """
        Find the ball_size nearest points to the center point.

        Args:
            points: All points array
            center_idx: Index of center point
            ball_size: Number of points to include in the ball

        Returns:
            List of indices of points in the ball
        """
        n = points.shape[0]
        center = points[center_idx]

        # Calculate distances to all points
        distances = []
        for i in range(n):
            dist = self._distance(center, points[i])
            distances.append((dist, i))

        # Sort by distance and take the ball_size nearest points
        distances.sort()
        ball_points = [idx for _, idx in distances[:ball_size]]

        return ball_points

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two points."""
        if self.distance == "cosine":
            return 1 - np.dot(a, b)
        else:  # Default to L2
            return np.linalg.norm(a - b)

    def verify_coverage(
        self, points: np.ndarray, selected_indices: List[int], eps: float
    ) -> Tuple[bool, float]:
        """
        Verify coverage quality of the selected points.

        Args:
            points: Original points array
            selected_indices: Indices of selected points
            eps: Epsilon parameter

        Returns:
            Tuple of (is_valid_coverage, coverage_ratio)
        """
        n = points.shape[0]
        covered_points = set()

        # For each selected point, find all points within eps distance
        for sel_idx in selected_indices:
            center = points[sel_idx]
            for i in range(n):
                if self._distance(center, points[i]) <= eps:
                    covered_points.add(i)

        coverage_ratio = len(covered_points) / n
        is_valid = (
            coverage_ratio >= 1.0
        )  # All points should be covered for valid eps-net

        return is_valid, coverage_ratio
