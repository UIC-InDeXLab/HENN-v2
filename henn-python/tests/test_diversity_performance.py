#!/usr/bin/env python3
"""
Test script to compare performance of different diversity selection methods.
"""

import numpy as np
import time
from epsnet.budgetaware import BudgetAware


def test_diversity_performance():
    """Test performance of different diversity selection methods."""
    
    # Generate test data
    print("Generating test data...")
    n_points = 10000
    n_dims = 128
    k_diverse = 50
    
    # Create synthetic data with some clusters
    np.random.seed(42)
    points = np.random.randn(n_points, n_dims)
    
    # Add some clusters to make it more realistic
    cluster_centers = np.random.randn(5, n_dims) * 3
    for i, center in enumerate(cluster_centers):
        start_idx = i * (n_points // 5)
        end_idx = (i + 1) * (n_points // 5)
        points[start_idx:end_idx] += center
    
    print(f"Data shape: {points.shape}")
    print(f"Selecting {k_diverse} diverse points from {n_points} points")
    print()
    
    # Test original method (optimized)
    print("Testing optimized k-means++ method...")
    budget_aware_standard = BudgetAware(distance="l2", random_seed=42, fast_diversity=False)
    
    start_time = time.time()
    diverse_indices_standard = budget_aware_standard._select_diverse_points(points, k_diverse)
    standard_time = time.time() - start_time
    
    print(f"Standard method: {standard_time:.3f} seconds")
    print(f"Selected {len(diverse_indices_standard)} points")
    
    # Test fast method
    print("\nTesting fast diversity method...")
    budget_aware_fast = BudgetAware(distance="l2", random_seed=42, fast_diversity=True)
    
    start_time = time.time()
    diverse_indices_fast = budget_aware_fast._select_diverse_points_fast(points, k_diverse)
    fast_time = time.time() - start_time
    
    print(f"Fast method: {fast_time:.3f} seconds")
    print(f"Selected {len(diverse_indices_fast)} points")
    print(f"Speedup: {standard_time / fast_time:.1f}x")
    
    # Compare diversity quality (approximate)
    def calculate_avg_min_distance(points, indices):
        """Calculate average minimum distance between selected points."""
        selected_points = points[indices]
        n_selected = len(selected_points)
        total_min_dist = 0
        
        for i in range(n_selected):
            min_dist = float('inf')
            for j in range(n_selected):
                if i != j:
                    dist = np.linalg.norm(selected_points[i] - selected_points[j])
                    min_dist = min(min_dist, dist)
            total_min_dist += min_dist
        
        return total_min_dist / n_selected
    
    print("\nDiversity quality comparison:")
    standard_diversity = calculate_avg_min_distance(points, diverse_indices_standard)
    fast_diversity = calculate_avg_min_distance(points, diverse_indices_fast)
    
    print(f"Standard method avg min distance: {standard_diversity:.3f}")
    print(f"Fast method avg min distance: {fast_diversity:.3f}")
    print(f"Diversity ratio (fast/standard): {fast_diversity/standard_diversity:.3f}")


def test_with_different_sizes():
    """Test performance with different dataset sizes."""
    print("\n" + "="*60)
    print("Testing with different dataset sizes...")
    print("="*60)
    
    sizes = [1000, 5000, 10000, 20000]
    k_diverse = 20
    
    for n_points in sizes:
        print(f"\nDataset size: {n_points}")
        
        # Generate data
        np.random.seed(42)
        points = np.random.randn(n_points, 64)
        
        # Standard method
        budget_aware_standard = BudgetAware(distance="l2", random_seed=42, fast_diversity=False)
        start_time = time.time()
        _ = budget_aware_standard._select_diverse_points(points, k_diverse)
        standard_time = time.time() - start_time
        
        # Fast method
        budget_aware_fast = BudgetAware(distance="l2", random_seed=42, fast_diversity=True)
        start_time = time.time()
        _ = budget_aware_fast._select_diverse_points_fast(points, k_diverse)
        fast_time = time.time() - start_time
        
        print(f"  Standard: {standard_time:.3f}s, Fast: {fast_time:.3f}s, Speedup: {standard_time/fast_time:.1f}x")


if __name__ == "__main__":
    test_diversity_performance()
    test_with_different_sizes()
    
    print("\n" + "="*60)
    print("Usage examples:")
    print("="*60)
    print("# Use standard optimized method (better diversity)")
    print("budget_aware = BudgetAware(distance='l2', fast_diversity=False)")
    print()
    print("# Use fast method (faster, slightly less diverse)")
    print("budget_aware = BudgetAware(distance='l2', fast_diversity=True)")
    print()
    print("# Then build epsnet as usual")
    print("epsnet = budget_aware.build_epsnet(points, eps=0.1)")
