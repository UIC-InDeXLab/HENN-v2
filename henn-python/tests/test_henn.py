import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import henn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from henn import HENN, HENNConfig, Layer
from pgraphs.knn import Knn


class TestLayer(unittest.TestCase):
    """Test cases for Layer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_points = np.random.rand(10, 3)  # 10 points in 3D
        self.layer = Layer(self.test_points)

    def test_layer_initialization(self):
        """Test layer initialization."""
        self.assertIsNone(self.layer.indices)
        self.assertIsNone(self.layer.edges)
        self.assertEqual(self.layer.n, 0)

    def test_add_indices(self):
        """Test adding indices to layer."""
        indices = [0, 2, 5, 7]
        self.layer.add_indices(indices)
        self.assertEqual(self.layer.indices, indices)
        self.assertEqual(self.layer.n, 4)

    def test_search_empty_layer(self):
        """Test search on empty layer."""
        query_point = np.array([0.5, 0.5, 0.5])
        result = self.layer.search(query_point, k=3)
        self.assertEqual(result, [])


class TestHENN(unittest.TestCase):
    """Test cases for HENN class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        np.random.seed(42)  # For reproducible tests
        self.test_points = np.random.rand(100, 3)  # 100 points in 3D
        self.query_point = np.array([0.5, 0.5, 0.5])

    def test_henn_initialization_default_config(self):
        """Test HENN initialization with default config."""
        henn = HENN(self.test_points)
        self.assertEqual(henn.n, 100)
        self.assertEqual(henn.d, 3)
        self.assertEqual(henn.exp_decay, 2)
        np.testing.assert_array_equal(henn.points, self.test_points)
        self.assertEqual(len(henn.layers), 0)  # Not built yet

    def test_henn_initialization_custom_config(self):
        """Test HENN initialization with custom config."""
        config = HENNConfig(enable_logging=True, log_level="DEBUG")
        henn = HENN(self.test_points, config=config, exp_decay=3)
        self.assertEqual(henn.exp_decay, 3)
        self.assertFalse(henn.logger.disabled)

    def test_logging_enable_disable(self):
        """Test enabling and disabling logging."""
        henn = HENN(self.test_points)

        # Initially disabled
        self.assertTrue(henn.logger.disabled)

        # Enable logging
        henn.enable_logging("INFO")
        self.assertFalse(henn.logger.disabled)

        # Disable logging
        henn.disable_logging()
        self.assertTrue(henn.logger.disabled)

    def test_query_before_build(self):
        """Test query before building the index."""
        henn = HENN(self.test_points)
        result = henn.query(self.query_point, k=5)
        self.assertEqual(result, [])

    def test_layer_count_calculation(self):
        """Test correct calculation of number of layers."""
        # For 100 points with exp_decay=2: L = floor(log_2(100)) = floor(6.64) = 6
        henn = HENN(self.test_points, exp_decay=2)
        expected_layers = int(np.floor(np.log(100) / np.log(2)))
        self.assertEqual(henn.L, expected_layers)

        # For 1000 points with exp_decay=3: L = floor(log_3(1000)) = floor(6.29) = 6
        large_points = np.random.rand(1000, 3)
        henn_large = HENN(large_points, exp_decay=3)
        expected_layers_large = int(np.floor(np.log(1000) / np.log(3)))


class TestHENNFunctionality(unittest.TestCase):
    """Test HENN functionality with mock implementations."""

    def setUp(self):
        """Set up test fixtures with mocks."""

        # Mock EpsNet that just returns the first 'size' indices
        class MockEpsNet:
            def build_epsnet(self, points, size=None):
                n = len(points)
                if size is None or size >= n:
                    return list(range(n))
                return list(range(min(size, n)))

        self.mock_epsnet = MockEpsNet()
        self.knn_pgraph = Knn()

    def test_henn_2d_points(self):
        """Test HENN with 2D points."""
        np.random.seed(42)
        points_2d = np.random.rand(50, 2)
        query_point = np.array([0.5, 0.5])

        config = HENNConfig(enable_logging=True, log_level="DEBUG")
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        config.pgraph_params = {"k": 3}  # Use k=3 for KNN graph

        henn = HENN(points_2d, config=config)

        # Test basic properties
        self.assertEqual(henn.d, 2)
        self.assertEqual(henn.n, 50)

        # Build the index
        henn.build()
        self.assertGreater(len(henn.layers), 0)

        # Test query for different k values
        for k in [1, 3, 5]:
            result = henn.query(query_point, k=k)
            self.assertLessEqual(len(result), k)
            self.assertGreater(len(result), 0)

            # Check that all returned indices are valid
            for idx in result:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, 50)

            # Verify results are sorted by distance
            distances = [np.linalg.norm(points_2d[idx] - query_point) for idx in result]
            self.assertEqual(distances, sorted(distances))

        print(f"✓ 2D test passed: {henn.n} points, {len(henn.layers)} layers")

    def test_henn_4d_points(self):
        """Test HENN with 4D points."""
        np.random.seed(123)
        points_4d = np.random.rand(80, 4)
        query_point = np.array([0.3, 0.7, 0.2, 0.8])

        config = HENNConfig(enable_logging=True, log_level="INFO")
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        config.pgraph_params = {"k": 5}  # Use k=5 for KNN graph

        henn = HENN(points_4d, config=config)

        # Test basic properties
        self.assertEqual(henn.d, 4)
        self.assertEqual(henn.n, 80)

        # Build the index
        henn.build()
        self.assertGreater(len(henn.layers), 0)

        # Test query for different k values
        for k in [1, 5, 10]:
            result = henn.query(query_point, k=k, ef=15)
            self.assertLessEqual(len(result), k)
            self.assertGreater(len(result), 0)

            # Check that all returned indices are valid
            for idx in result:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, 80)

            # Verify results are sorted by distance
            distances = [np.linalg.norm(points_4d[idx] - query_point) for idx in result]
            self.assertEqual(distances, sorted(distances))

        print(f"✓ 4D test passed: {henn.n} points, {len(henn.layers)} layers")

    def test_henn_accuracy_check(self):
        """Test HENN accuracy by comparing with brute force."""
        np.random.seed(456)
        points = np.random.rand(30, 3)
        query_point = np.array([0.4, 0.6, 0.5])
        k = 5

        # Brute force nearest neighbors
        distances = [
            (np.linalg.norm(point - query_point), i) for i, point in enumerate(points)
        ]
        distances.sort()
        true_nn = [idx for _, idx in distances[:k]]

        # HENN nearest neighbors
        config = HENNConfig(enable_logging=True)
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        config.pgraph_params = {"k": 4}  # Use k=4 for KNN graph

        henn = HENN(points, config=config)
        henn.build()
        henn_nn = henn.query(query_point, k=k)

        # Check that HENN found valid neighbors
        self.assertEqual(len(henn_nn), k)
        for idx in henn_nn:
            self.assertIn(idx, list(range(30)))

        # Calculate overlap with true nearest neighbors
        overlap = len(set(true_nn) & set(henn_nn))
        overlap_ratio = overlap / k

        print(f"✓ Accuracy test: {overlap}/{k} overlap ({overlap_ratio:.1%})")
        print(f"  True NN: {true_nn}")
        print(f"  HENN NN: {henn_nn}")

        # Should have some overlap (exact match depends on graph connectivity)
        self.assertGreater(overlap_ratio, 0.2)  # At least 20% overlap

    def test_layer_structure(self):
        """Test that layers have decreasing sizes."""
        np.random.seed(789)
        points = np.random.rand(64, 3)  # 64 = 2^6 for clean layer sizes

        config = HENNConfig(enable_logging=True, log_level="DEBUG")
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        config.pgraph_params = {"k": 3}  # Use k=3 for KNN graph

        henn = HENN(points, config=config, exp_decay=2)
        henn.build()

        # Check that layer sizes decrease
        layer_sizes = [layer.n for layer in henn.layers]
        print(f"✓ Layer sizes: {layer_sizes}")

        # Each layer should be smaller than the previous (except possibly the last few)
        for i in range(len(layer_sizes) - 1):
            self.assertLessEqual(layer_sizes[i + 1], layer_sizes[i])

        # First layer should have the most points
        self.assertEqual(layer_sizes[0], 64)

        # All layers should have some points
        for size in layer_sizes:
            self.assertGreater(size, 0)

    def test_different_ef_values(self):
        """Test HENN with different ef values."""
        np.random.seed(321)
        points = np.random.rand(40, 3)
        query_point = np.array([0.3, 0.7, 0.4])
        k = 5

        config = HENNConfig(enable_logging=False)
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        config.pgraph_params = {"k": 4}

        henn = HENN(points, config=config)
        henn.build()

        # Test different ef values
        ef_values = [5, 10, 20, 30]
        results = {}

        for ef in ef_values:
            result = henn.query(query_point, k=k, ef=ef)
            results[ef] = result

            # Should always return exactly k points
            self.assertEqual(len(result), k)

            # All indices should be valid
            for idx in result:
                self.assertIn(idx, list(range(40)))

            # Results should be sorted by distance
            distances = [np.linalg.norm(points[idx] - query_point) for idx in result]
            self.assertEqual(distances, sorted(distances))

        print(f"✓ Different ef values test passed")
        for ef, result in results.items():
            print(f"  ef={ef}: {result}")

    def test_different_dataset_sizes(self):
        """Test HENN with different dataset sizes resulting in different layer counts."""
        test_cases = [
            (16, 2),  # 16 points, exp_decay=2 -> 4 layers
            (32, 2),  # 32 points, exp_decay=2 -> 5 layers
            (64, 2),  # 64 points, exp_decay=2 -> 6 layers
            (27, 3),  # 27 points, exp_decay=3 -> 3 layers
            (81, 3),  # 81 points, exp_decay=3 -> 4 layers
        ]

        for n, exp_decay in test_cases:
            with self.subTest(n=n, exp_decay=exp_decay):
                np.random.seed(42 + n)  # Different seed for each test
                points = np.random.rand(n, 3)
                query_point = np.array([0.5, 0.5, 0.5])

                config = HENNConfig(enable_logging=False)
                config.epsnet_algorithm = self.mock_epsnet
                config.pgraph_algorithm = self.knn_pgraph
                config.pgraph_params = {"k": min(3, n - 1)}  # Ensure k < n

                henn = HENN(points, config=config, exp_decay=exp_decay)

                # Check expected layer count
                expected_layers = int(np.floor(np.log(n) / np.log(exp_decay)))
                self.assertEqual(henn.L, expected_layers)

                # Build and test
                henn.build()
                actual_layers = len(henn.layers)

                # Query should work
                result = henn.query(query_point, k=min(3, n))
                self.assertGreater(len(result), 0)
                self.assertLessEqual(len(result), min(3, n))

                # Check layer sizes decrease
                layer_sizes = [layer.n for layer in henn.layers]
                for i in range(len(layer_sizes) - 1):
                    self.assertLessEqual(layer_sizes[i + 1], layer_sizes[i])

                print(
                    f"✓ n={n}, exp_decay={exp_decay}: {actual_layers} layers, sizes={layer_sizes}"
                )

    def test_comprehensive_dimensions(self):
        """Test HENN comprehensively with different dimensions."""
        dimensions = [2, 3, 4, 5, 8]

        for d in dimensions:
            with self.subTest(dimension=d):
                np.random.seed(100 + d)
                n_points = 50
                points = np.random.rand(n_points, d)
                query_point = np.random.rand(d)

                config = HENNConfig(enable_logging=False)
                config.epsnet_algorithm = self.mock_epsnet
                config.pgraph_algorithm = self.knn_pgraph
                config.pgraph_params = {"k": 4}

                henn = HENN(points, config=config)

                # Test basic properties
                self.assertEqual(henn.d, d)
                self.assertEqual(henn.n, n_points)

                # Build and query
                henn.build()

                for k in [1, 3, 7]:
                    result = henn.query(query_point, k=k)

                    # Should return exactly k points
                    self.assertEqual(len(result), k)

                    # All indices should be valid
                    for idx in result:
                        self.assertIn(idx, list(range(n_points)))

                    # Results should be sorted by distance
                    distances = [
                        np.linalg.norm(points[idx] - query_point) for idx in result
                    ]
                    self.assertEqual(distances, sorted(distances))

                print(f"✓ {d}D test passed: {len(henn.layers)} layers")

    def test_sample_points_output_verification(self):
        """Test with specific sample points and verify exact k-point output."""
        # Create a known set of points
        points = np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.0, 1.0],  # 2
                [1.0, 1.0],  # 3
                [0.5, 0.5],  # 4
                [0.2, 0.3],  # 5
                [0.8, 0.7],  # 6
                [0.1, 0.9],  # 7
                [0.9, 0.1],  # 8
                [0.6, 0.4],  # 9
            ]
        )

        query_point = np.array([0.5, 0.5])  # Center point

        config = HENNConfig(enable_logging=True, log_level="INFO")
        config.epsnet_algorithm = self.mock_epsnet
        config.pgraph_algorithm = self.knn_pgraph
        # config.pgraph_params = {"k": 3}
        config.pgraph_params = {"k": 5}

        henn = HENN(points, config=config)
        henn.build()

        # Test different k values
        for k in [1, 3, 5, 8]:
            result = henn.query(query_point, k=k)

            # Must return exactly k points
            self.assertEqual(len(result), k, f"Expected {k} points, got {len(result)}")

            # All indices must be valid
            for idx in result:
                self.assertIn(idx, list(range(10)), f"Invalid index {idx}")

            # No duplicates
            self.assertEqual(len(result), len(set(result)), "Duplicate indices found")

            # Calculate distances to verify sorting
            distances = [np.linalg.norm(points[idx] - query_point) for idx in result]
            self.assertEqual(
                distances, sorted(distances), "Results not sorted by distance"
            )

            print(
                f"  k={k}: indices={result}, distances={[f'{d:.3f}' for d in distances]}"
            )

        print("✓ Sample points verification passed")


if __name__ == "__main__":
    unittest.main()
