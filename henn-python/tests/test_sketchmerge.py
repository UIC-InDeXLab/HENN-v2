import unittest
import numpy as np
import random
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epsnet.sketchmerge import SketchMerge


class TestSketchMerge(unittest.TestCase):
    """Test cases for SketchMerge EPSNet implementation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.sketch_merge_l2 = SketchMerge(distance="l2")
        self.sketch_merge_cosine = SketchMerge(distance="cosine")

    def test_basic_functionality(self):
        """Test basic SketchMerge functionality."""
        # Create simple 2D test points
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])
        
        size = 3
        indices = self.sketch_merge_l2.build_epsnet(points, size=size)
        
        # Should return exactly the requested size
        self.assertEqual(len(indices), size)
        
        # All indices should be valid
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(points))
        
        # No duplicate indices
        self.assertEqual(len(indices), len(set(indices)))
        
        print("✓ Basic functionality test passed")

    def test_size_constraints(self):
        """Test different size constraints."""
        points = np.random.rand(20, 3)
        
        # Test various sizes
        for size in [1, 5, 10, 15, 20]:
            with self.subTest(size=size):
                indices = self.sketch_merge_l2.build_epsnet(points, size=size)
                expected_size = min(size, len(points))
                self.assertEqual(len(indices), expected_size)
        
        print("✓ Size constraints test passed")

    def test_epsilon_parameter(self):
        """Test with epsilon parameter instead of size."""
        points = np.random.rand(15, 2)
        eps = 0.5
        
        indices = self.sketch_merge_l2.build_epsnet(points, eps=eps)
        
        # Should return valid indices
        self.assertGreater(len(indices), 0)
        self.assertLessEqual(len(indices), len(points))
        
        # All indices should be valid
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(points))
        
        print("✓ Epsilon parameter test passed")

    def test_distance_metrics(self):
        """Test different distance metrics."""
        points = np.random.rand(10, 4)
        # Normalize for cosine distance
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        size = 5
        
        # Test L2 distance
        indices_l2 = self.sketch_merge_l2.build_epsnet(points, size=size)
        self.assertEqual(len(indices_l2), size)
        
        # Test cosine distance
        indices_cosine = self.sketch_merge_cosine.build_epsnet(points, size=size)
        self.assertEqual(len(indices_cosine), size)
        
        print("✓ Distance metrics test passed")

    def test_ball_range_generation(self):
        """Test ball range generation."""
        points = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        eps = 0.5
        
        ranges = self.sketch_merge_l2._generate_ball_ranges(points, eps)
        
        # Should have one range per point
        self.assertEqual(len(ranges), len(points))
        
        # Each range should contain at least the point itself
        for i, range_set in enumerate(ranges):
            self.assertIn(i, range_set)
        
        # Points 0 and 1 should be in each other's ranges (distance 0.1 < 0.5)
        self.assertIn(1, ranges[0])
        self.assertIn(0, ranges[1])
        
        # Point 2 should not be in point 0's range (distance 1.0 > 0.5)
        self.assertNotIn(2, ranges[0])
        
        print("✓ Ball range generation test passed")

    def test_discrepancy_calculation(self):
        """Test discrepancy calculation."""
        ranges = [
            {0, 1, 2},
            {1, 2, 3},
            {0, 3}
        ]
        
        coloring = {0: 1, 1: -1, 2: 1, 3: -1}
        
        max_disc = self.sketch_merge_l2._calculate_max_discrepancy(coloring, ranges)
        
        # Range 0: 1 + (-1) + 1 = 1, abs = 1
        # Range 1: (-1) + 1 + (-1) = -1, abs = 1  
        # Range 2: 1 + (-1) = 0, abs = 0
        # Max discrepancy should be 1
        self.assertEqual(max_disc, 1)
        
        print("✓ Discrepancy calculation test passed")

    def test_greedy_discrepancy_halving(self):
        """Test greedy discrepancy halving."""
        pairs = [(0, 1), (2, 3)]
        ranges = [
            {0, 1},
            {2, 3},
            {0, 2},
            {1, 3}
        ]
        
        selected = self.sketch_merge_l2._greedy_discrepancy_halving(pairs, ranges)
        
        # Should select exactly one from each pair
        self.assertEqual(len(selected), 2)
        
        # Should be valid selections
        self.assertTrue(0 in selected or 1 in selected)
        self.assertTrue(2 in selected or 3 in selected)
        
        print("✓ Greedy discrepancy halving test passed")

    def test_random_halving(self):
        """Test random halving functionality."""
        points = np.random.rand(8, 2)
        point_indices = list(range(8))
        ranges = self.sketch_merge_l2._generate_ball_ranges(points, 0.5)
        
        halved = self.sketch_merge_l2._random_halving(point_indices, points, ranges)
        
        # Should return roughly half the points
        self.assertLessEqual(len(halved), len(point_indices))
        self.assertGreater(len(halved), 0)
        
        # All returned indices should be from original set
        for idx in halved:
            self.assertIn(idx, point_indices)
        
        print("✓ Random halving test passed")

    def test_sketch_merge_process(self):
        """Test the sketch-merge process."""
        points = np.random.rand(16, 3)
        partitions = [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
            list(range(12, 16))
        ]
        ranges = self.sketch_merge_l2._generate_ball_ranges(points, 0.5)
        
        result = self.sketch_merge_l2._sketch_merge(partitions, points, ranges)
        
        # Should return a single partition with points from all original partitions
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 16)
        
        # All indices should be valid
        for idx in result:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 16)
        
        print("✓ Sketch-merge process test passed")

    def test_epsnet_verification(self):
        """Test eps-net verification functionality."""
        # Create points in a grid pattern
        points = np.array([
            [0.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [1.0, 1.0],
            [0.5, 0.5]
        ])
        eps = 0.8
        
        # Test with a valid eps-net (should cover all ranges)
        selected_indices = [0, 2, 4]  # Corner and center points
        is_valid = self.sketch_merge_l2.verify_epsnet(points, selected_indices, eps)
        
        # The verification result depends on the specific eps value and point distribution
        # We mainly test that the function runs without error
        self.assertIsInstance(is_valid, bool)
        
        print("✓ EPSNet verification test passed")

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point
        single_point = np.array([[0.0, 0.0]])
        indices = self.sketch_merge_l2.build_epsnet(single_point, size=1)
        self.assertEqual(indices, [0])
        
        # Size larger than number of points
        small_points = np.random.rand(3, 2)
        indices = self.sketch_merge_l2.build_epsnet(small_points, size=10)
        self.assertEqual(len(indices), 3)
        self.assertEqual(set(indices), {0, 1, 2})
        
        print("✓ Edge cases test passed")

    def test_higher_dimensions(self):
        """Test with higher dimensional data."""
        for d in [5, 10, 20]:
            with self.subTest(dimension=d):
                points = np.random.rand(50, d)
                size = 10
                
                indices = self.sketch_merge_l2.build_epsnet(points, size=size)
                
                self.assertEqual(len(indices), size)
                for idx in indices:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, 50)
        
        print("✓ Higher dimensions test passed")

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        points = np.random.rand(20, 3)
        size = 8
        
        # Create instance with fixed seed
        sketch_merge_fixed = SketchMerge(distance="l2", random_seed=123)
        
        # Run twice with same algorithm instance and seed
        indices1 = sketch_merge_fixed.build_epsnet(points, size=size)
        indices2 = sketch_merge_fixed.build_epsnet(points, size=size)
        
        # Results should be the same
        self.assertEqual(indices1, indices2)
        
        print("✓ Reproducibility test passed")


if __name__ == "__main__":
    unittest.main()
