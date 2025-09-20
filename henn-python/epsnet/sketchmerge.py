import numpy as np
import math
import random
from tqdm import tqdm
from typing import List, Set, Tuple
from epsnet.base_epsnet import BaseEPSNet


class SketchMerge(BaseEPSNet):
    """
    Sketch-and-merge EPSNet construction using discrepancy minimization.

    This algorithm builds an epsilon-net by:
    1. Partitioning points into smaller groups
    2. Recursively merging pairs and applying discrepancy halving
    3. Final halving to achieve desired size
    """

    def __init__(self, distance="l2", random_seed=None):
        self.distance = distance
        self.random_seed = random_seed

    def build_epsnet(self, points: np.ndarray, eps=None, size=None, ranges=None):
        """
        Build eps-net using sketch-and-merge algorithm.

        Args:
            points: Input points array (n, d)
            eps: Epsilon parameter (not used directly, size is prioritized)
            size: Target size of the eps-net

        Returns:
            List of indices representing the eps-net
        """
        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        n, d = points.shape

        if size is None:
            if eps is None:
                raise ValueError("Either eps or size must be provided")
            size = BaseEPSNet.get_size(n, eps, d)

        size = min(size, n)

        print(f"Size of epsnet: {size} / {n}")

        if size >= n:
            return list(range(n))

        # Generate ball ranges for all points
        if ranges is None:
            ranges = self._generate_ball_ranges(points, eps)

        # Calculate partition size - should be power of 2
        c1 = 2  # constant for partition size calculation
        p = size * (2**c1)  # size of each partition
        p = 2 ** math.ceil(math.log2(p))  # round to nearest power of 2
        p = min(p, n)  # don't exceed total points

        # Create partitions
        partitions = []
        indices = list(range(n))
        random.shuffle(indices)  # randomize for better distribution

        for i in range(0, n, p):
            partition = indices[i : i + p]
            partitions.append(partition)

        # Sketch-and-merge phase
        print('Merging partitions...')
        root = self._sketch_merge(partitions, points, ranges)

        # Final halving to achieve target size
        print('Final halving...')
        while len(root) > 2 * size:
            print(f"Current size: {len(root)}, halving...")
            root = self._random_halving(root, points, ranges)

        return root[:size]  # Return exactly 'size' points

    def _generate_ball_ranges(self, points: np.ndarray, eps: float) -> List[Set[int]]:
        """
        Generate ball ranges around each point.

        Args:
            points: Input points array
            eps: Radius for ball ranges

        Returns:
            List of sets, where each set contains indices of points within eps distance
        """
        n = points.shape[0]
        ranges = []

        if eps is None:
            # Use adaptive epsilon based on point distribution
            eps = self._estimate_epsilon(points)

        for i in tqdm(range(n), desc="Generating ball ranges"):
            ball_range = set()
            center = points[i]

            for j in range(n):
                if self._distance(center, points[j]) <= eps:
                    ball_range.add(j)

            ranges.append(ball_range)

        return ranges

    def _estimate_epsilon(self, points: np.ndarray) -> float:
        """Estimate a reasonable epsilon based on point distribution."""
        n = points.shape[0]
        sample_size = min(100, n)
        sample_indices = random.sample(range(n), sample_size)

        distances = []
        for i in sample_indices:
            for j in sample_indices:
                if i != j:
                    dist = self._distance(points[i], points[j])
                    distances.append(dist)

        # Use median distance as a reasonable epsilon
        distances.sort()
        return distances[len(distances) // 2] if distances else 1.0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two points."""
        if self.distance == "cosine":
            return 1 - np.dot(a, b)
        else:  # Default to L2
            return np.linalg.norm(a - b)

    def _sketch_merge(
        self, partitions: List[List[int]], points: np.ndarray, ranges: List[Set[int]]
    ) -> List[int]:
        """
        Recursively merge partitions using discrepancy halving.

        Args:
            partitions: List of point index partitions
            points: Original points array
            ranges: Ball ranges for discrepancy calculation

        Returns:
            Final merged partition
        """
        while len(partitions) > 1:
            new_partitions = []

            # Process pairs of partitions
            for i in tqdm(range(0, len(partitions), 2), desc="Merging partitions"):
                if i + 1 < len(partitions):
                    # Merge two partitions
                    merged = list(set(partitions[i]) | set(partitions[i + 1]))
                    # Apply discrepancy halving
                    halved = self._random_halving(merged, points, ranges)
                    new_partitions.append(halved)
                else:
                    # Odd partition remains
                    new_partitions.append(partitions[i])

            partitions = new_partitions

        return partitions[0] if partitions else []

    def _random_halving(
        self, point_indices: List[int], points: np.ndarray, ranges: List[Set[int]]
    ) -> List[int]:
        """
        Apply discrepancy-minimizing random halving.

        Args:
            point_indices: Indices of points to halve
            points: Original points array
            ranges: Ball ranges for discrepancy calculation

        Returns:
            Halved set of point indices
        """
        if len(point_indices) <= 1:
            return point_indices

        # Shuffle and pair points
        shuffled = point_indices.copy()
        random.shuffle(shuffled)

        # Ensure even number of points
        if len(shuffled) % 2 == 1:
            shuffled = shuffled[:-1]

        # Create pairs
        pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]

        # Apply greedy discrepancy halving
        return self._greedy_discrepancy_halving(pairs, ranges)

    def _greedy_discrepancy_halving(
        self, pairs: List[Tuple[int, int]], ranges: List[Set[int]]
    ) -> List[int]:
        """
        Greedy discrepancy halving to minimize maximum discrepancy.

        Args:
            pairs: List of point index pairs
            ranges: Ball ranges for discrepancy calculation

        Returns:
            Selected points that minimize discrepancy
        """
        coloring = {}
        selected = []

        for pair in pairs:
            p1, p2 = pair

            # Try both assignments and choose the one with lower max discrepancy

            # Option 1: select p1
            coloring[p1] = 1
            coloring[p2] = -1
            max_disc_1 = self._calculate_max_discrepancy(coloring, ranges)

            # Option 2: select p2
            coloring[p1] = -1
            coloring[p2] = 1
            max_disc_2 = self._calculate_max_discrepancy(coloring, ranges)

            # Choose the better option
            if max_disc_1 <= max_disc_2:
                coloring[p1] = 1
                coloring[p2] = -1
                selected.append(p1)
            else:
                coloring[p1] = -1
                coloring[p2] = 1
                selected.append(p2)

        return selected

    def _calculate_max_discrepancy(
        self, coloring: dict, ranges: List[Set[int]]
    ) -> float:
        """
        Calculate maximum discrepancy over all ranges.

        Args:
            coloring: Dictionary mapping point indices to {-1, 1}
            ranges: Ball ranges

        Returns:
            Maximum absolute discrepancy
        """
        max_discrepancy = 0

        for range_set in ranges:
            discrepancy = sum(coloring.get(point_idx, 0) for point_idx in range_set)
            max_discrepancy = max(max_discrepancy, abs(discrepancy))

        return max_discrepancy

    def verify_epsnet(
        self, points: np.ndarray, selected_indices: List[int], eps: float
    ) -> bool:
        """
        Verify if the selected points form a valid eps-net.

        Args:
            points: Original points array
            selected_indices: Indices of selected points
            eps: Epsilon parameter

        Returns:
            True if valid eps-net, False otherwise
        """
        ranges = self._generate_ball_ranges(points, eps)

        for range_set in ranges:
            # Check if this range is hit by at least one selected point
            if not any(idx in selected_indices for idx in range_set):
                return False

        return True
