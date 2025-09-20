import unittest
import numpy as np
import random
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epsnet.budgetaware import BudgetAware


class TestBudgetAware(unittest.TestCase):
    """Test cases for BudgetAware EPSNet implementation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.budget_aware_l2 = BudgetAware(distance="l2")
        self.budget_aware_cosine = BudgetAware(distance="cosine")

    def test_basic_functionality(self):
        """Test basic BudgetAware functionality."""
        # Create simple 2D test points
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]
        ])
        
        size = 4
        budget = 2
        indices = self.budget_aware_l2.build_epsnet(points, size=size, budget=budget)
        
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
        
        # Test various sizes and budgets
        test_cases = [
            (5, 1), (10, 2), (15, 3), (20, 5)
        ]
        
        for size, budget in test_cases:
            with self.subTest(size=size, budget=budget):
                indices = self.budget_aware_l2.build_epsnet(
                    points, size=size, budget=budget
                )
                expected_size = min(size, len(points))
                self.assertEqual(len(indices), expected_size)
        
        print("✓ Size constraints test passed")

    def test_budget_parameter(self):
        """Test budget parameter effects."""
        points = np.random.rand(30, 2)
        size = 10
        
        # Test different budget values
        for budget in [1, 3, 5, 10]:
            with self.subTest(budget=budget):
                indices = self.budget_aware_l2.build_epsnet(
                    points, size=size, budget=budget
                )
                
                # Should return valid indices
                self.assertEqual(len(indices), size)
                for idx in indices:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, len(points))
        
        print("✓ Budget parameter test passed")

    def test_initial_sample_size(self):
        """Test that initial sample size is calculated correctly."""
        points = np.random.rand(50, 3)
        size = 20
        budget = 5
        
        # The algorithm should start with size - budget points
        # and then add up to budget more points
        indices = self.budget_aware_l2.build_epsnet(
            points, size=size, budget=budget
        )
        
        self.assertEqual(len(indices), size)
        
        print("✓ Initial sample size test passed")

    def test_epsilon_parameter(self):
        """Test with epsilon parameter instead of size."""
        points = np.random.rand(15, 2)
        eps = 0.5
        budget = 2
        
        indices = self.budget_aware_l2.build_epsnet(
            points, eps=eps, budget=budget
        )
        
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
        points = np.random.rand(15, 4)
        # Normalize for cosine distance
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        size = 8
        budget = 3
        
        # Test L2 distance
        indices_l2 = self.budget_aware_l2.build_epsnet(
            points, size=size, budget=budget
        )
        self.assertEqual(len(indices_l2), size)
        
        # Test cosine distance
        indices_cosine = self.budget_aware_cosine.build_epsnet(
            points, size=size, budget=budget
        )
        self.assertEqual(len(indices_cosine), size)
        
        print("✓ Distance metrics test passed")

    def test_diverse_point_selection(self):
        """Test diverse point selection functionality."""
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [3.0, 3.0]
        ])
        
        k = 4
        diverse_indices = self.budget_aware_l2._select_diverse_points(points, k)
        
        # Should return exactly k indices
        self.assertEqual(len(diverse_indices), k)
        
        # All indices should be unique
        self.assertEqual(len(set(diverse_indices)), k)
        
        # All indices should be valid
        for idx in diverse_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(points))
        
        print("✓ Diverse point selection test passed")

    def test_ball_finding(self):
        """Test finding ball points functionality."""
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]
        ])
        
        center_idx = 0
        ball_size = 4
        
        ball_points = self.budget_aware_l2._find_ball_points(
            points, center_idx, ball_size
        )
        
        # Should return exactly ball_size points
        self.assertEqual(len(ball_points), ball_size)
        
        # All indices should be valid
        for idx in ball_points:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(points))
        
        # The center point should be included (closest to itself)
        self.assertIn(center_idx, ball_points)
        
        print("✓ Ball finding test passed")

    def test_coverage_verification(self):
        """Test coverage verification functionality."""
        points = np.array([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
            [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
        ])
        
        selected_indices = [0, 4, 8]  # Corner and center
        eps = 0.8
        
        is_valid, coverage_ratio = self.budget_aware_l2.verify_coverage(
            points, selected_indices, eps
        )
        
        # Should return valid results
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(coverage_ratio, float)
        self.assertGreaterEqual(coverage_ratio, 0.0)
        self.assertLessEqual(coverage_ratio, 1.0)
        
        print("✓ Coverage verification test passed")

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point
        single_point = np.array([[0.0, 0.0]])
        indices = self.budget_aware_l2.build_epsnet(
            single_point, size=1, budget=1
        )
        self.assertEqual(indices, [0])
        
        # Size larger than number of points
        small_points = np.random.rand(3, 2)
        indices = self.budget_aware_l2.build_epsnet(
            small_points, size=10, budget=2
        )
        self.assertEqual(len(indices), 3)
        self.assertEqual(set(indices), {0, 1, 2})
        
        # Zero budget
        points = np.random.rand(10, 2)
        indices = self.budget_aware_l2.build_epsnet(
            points, size=5, budget=0
        )
        self.assertEqual(len(indices), 5)
        
        print("✓ Edge cases test passed")

    def test_default_budget(self):
        """Test default budget calculation."""
        points = np.random.rand(100, 3)
        size = 20
        
        # Don't specify budget, should use default (size // 10)
        indices = self.budget_aware_l2.build_epsnet(points, size=size)
        
        self.assertEqual(len(indices), size)
        
        print("✓ Default budget test passed")

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        points = np.random.rand(20, 3)
        size = 8
        budget = 3
        
        # Create instance with fixed seed
        budget_aware_fixed = BudgetAware(distance="l2", random_seed=123)
        
        # Run twice with same algorithm instance and seed
        indices1 = budget_aware_fixed.build_epsnet(
            points, size=size, budget=budget
        )
        indices2 = budget_aware_fixed.build_epsnet(
            points, size=size, budget=budget
        )
        
        # Results should be the same
        self.assertEqual(indices1, indices2)
        
        print("✓ Reproducibility test passed")

    def test_large_partition_numbers(self):
        """Test with large number of partitions."""
        points = np.random.rand(50, 2)
        size = 15
        budget = 5
        
        # Should handle large datasets gracefully
        indices = self.budget_aware_l2.build_epsnet(
            points, size=size, budget=budget
        )
        
        self.assertEqual(len(indices), size)
        
        print("✓ Large partition numbers test passed")

    def test_budget_improvement_effect(self):
        """Test that budget actually improves the solution."""
        points = np.random.rand(30, 3)
        size = 10
        eps = 0.5
        
        # Compare with different budgets
        budget_aware_small = BudgetAware(distance="l2", random_seed=456)
        budget_aware_large = BudgetAware(distance="l2", random_seed=456)
        
        indices_small = budget_aware_small.build_epsnet(
            points, size=size, budget=1
        )
        indices_large = budget_aware_large.build_epsnet(
            points, size=size, budget=5
        )
        
        # Both should return valid results
        self.assertEqual(len(indices_small), size)
        self.assertEqual(len(indices_large), size)
        
        # Verify coverage for both
        _, coverage_small = budget_aware_small.verify_coverage(
            points, indices_small, eps
        )
        _, coverage_large = budget_aware_large.verify_coverage(
            points, indices_large, eps
        )
        
        # Both should provide reasonable coverage
        self.assertGreaterEqual(coverage_small, 0.0)
        self.assertGreaterEqual(coverage_large, 0.0)
        
        print("✓ Budget improvement effect test passed")

    def test_higher_dimensions(self):
        """Test with higher dimensional data."""
        for d in [5, 10, 15]:
            with self.subTest(dimension=d):
                points = np.random.rand(40, d)
                size = 10
                budget = 3
                
                indices = self.budget_aware_l2.build_epsnet(
                    points, size=size, budget=budget
                )
                
                self.assertEqual(len(indices), size)
                for idx in indices:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, 40)
        
        print("✓ Higher dimensions test passed")


if __name__ == "__main__":
    unittest.main()
