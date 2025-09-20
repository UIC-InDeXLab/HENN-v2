import numpy as np
import math
from tqdm import tqdm
from typing import List, Set


class BaseEPSNet:
    def build_epsnet(self, points: np.ndarray, eps=None, size=None):
        """Return indices of selected points for EPSNet layer."""
        raise NotImplementedError("Subclasses must implement build_epsnet method")

    @classmethod
    def get_eps(cls, n, size, d):
        raise NotImplementedError("get_eps method")

    @classmethod
    def get_size(cls, n, eps, d, success_prob=0.9):
        c1 = 1  # constant
        phi = 1 - success_prob
        m = max((4 / eps) * math.log2(4 / phi), (8 * d / eps) * math.log2(16 / eps))
        m = m / c1
        return math.ceil(m)

    @classmethod
    def verify(
        cls,
        epsnet: List[int],
        points: np.ndarray,
        eps: float,
        distance: str = "l2",
        ranges=None,
    ) -> bool:
        """
        Verify if the selected points form a valid eps-net.

        An eps-net is valid if every point in the dataset has at least one point
        from the eps-net within its eps-fraction ball.

        Args:
            epsnet: List of indices of selected points
            points: Original points array (n, d)
            eps: Epsilon parameter (fraction of points to include in ball)
            distance: Distance metric ("l2" or "cosine")

        Returns:
            True if valid eps-net, False otherwise
        """
        # Generate ball ranges for all points
        if ranges is None:
            ranges = cls._generate_ball_ranges(points, eps, distance)

        # Check if each range is hit by at least one selected point
        for range_set in ranges:
            # Check if this range is hit by at least one selected point
            if not any(idx in epsnet for idx in range_set):
                return False, ranges

        return True, ranges

    @classmethod
    def _generate_ball_ranges(
        cls, points: np.ndarray, eps: float, distance: str = "l2"
    ) -> List[Set[int]]:
        """
        Generate ball ranges around each point.

        Args:
            points: Input points array
            eps: Fraction of points to include in each ball (0 < eps <= 1)
            distance: Distance metric ("l2" or "cosine")

        Returns:
            List of sets, where each set contains indices of the closest eps fraction of points
        """
        n = points.shape[0]
        ranges = []

        # Calculate how many points to include in each ball
        ball_size = max(1, int(eps * n))

        for i in tqdm(range(n), desc="Generating ball ranges"):
            center = points[i]

            # Calculate distances to all points
            distances = []
            for j in range(n):
                dist = cls._distance(center, points[j], distance)
                distances.append((dist, j))

            # Sort by distance and take the closest eps fraction of points
            distances.sort(key=lambda x: x[0])
            ball_range = set()

            for k in range(ball_size):
                ball_range.add(distances[k][1])

            ranges.append(ball_range)

        return ranges

    @classmethod
    def _distance(cls, a: np.ndarray, b: np.ndarray, distance: str = "l2") -> float:
        """Calculate distance between two points."""
        if distance == "cosine":
            return 1 - np.dot(a, b)
        else:  # Default to L2
            return np.linalg.norm(a - b)
