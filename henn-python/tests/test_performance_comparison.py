#!/usr/bin/env python3
"""
Performance comparison test for diversity selection methods.
"""

import numpy as np
import time
import sys
import os

# Add the henn-python directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from epsnet.budgetaware import BudgetAware


def time_function(func, *args, **kwargs):
    """Time a function call and return result and time taken."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def test_performance_comparison():
    """Compare performance of different diversity selection methods."""
    
    print("Performance Comparison: Diversity Selection Methods")
    print("=" * 60)
    
    # Test with different dataset sizes
    test_configs = [
        (500, 32, 10),
        (1000, 64, 15),
        (2000, 128, 20),
        (5000, 128, 25),
    ]
    
    for n_points, n_dims, k_diverse in test_configs:
        print(f"\nDataset: {n_points} points, {n_dims} dims, selecting {k_diverse} diverse points")
        print("-" * 50)
        
        # Generate test data
        np.random.seed(42)
        points = np.random.randn(n_points, n_dims)
        
        # Test optimized standard method
        ba_standard = BudgetAware(distance="l2", random_seed=42, fast_diversity=False)
        indices_std, time_std = time_function(ba_standard._select_diverse_points, points, k_diverse)
        
        # Test fast method
        ba_fast = BudgetAware(distance="l2", random_seed=42, fast_diversity=True)
        indices_fast, time_fast = time_function(ba_fast._select_diverse_points_fast, points, k_diverse)
        
        # Calculate basic diversity metrics
        def avg_pairwise_distance(points, indices):
            selected_points = points[indices]
            n = len(selected_points)
            if n < 2:
                return 0.0
            
            total_dist = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(selected_points[i] - selected_points[j])
                    total_dist += dist
                    count += 1
            return total_dist / count if count > 0 else 0.0
        
        diversity_std = avg_pairwise_distance(points, indices_std)
        diversity_fast = avg_pairwise_distance(points, indices_fast)
        
        print(f"Standard method: {time_std:.3f}s, avg distance: {diversity_std:.3f}")
        print(f"Fast method:     {time_fast:.3f}s, avg distance: {diversity_fast:.3f}")
        
        if time_fast > 0:
            speedup = time_std / time_fast
            diversity_ratio = diversity_fast / diversity_std if diversity_std > 0 else 1.0
            print(f"Speedup: {speedup:.1f}x, Diversity retention: {diversity_ratio:.2f}")


def test_full_epsnet_performance():
    """Test performance of full epsnet construction with different diversity methods."""
    
    print("\n\nFull EPSNet Construction Performance")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_points = 2000
    n_dims = 64
    points = np.random.randn(n_points, n_dims)
    
    eps = 0.05
    size = 100
    budget = 10
    
    print(f"Dataset: {n_points} points, {n_dims} dims")
    print(f"EPSNet params: eps={eps}, size={size}, budget={budget}")
    print("-" * 50)
    
    # Test with standard diversity selection
    print("Testing with standard diversity selection...")
    ba_standard = BudgetAware(distance="l2", random_seed=42, fast_diversity=False)
    epsnet_std, time_std = time_function(ba_standard.build_epsnet, points, eps=eps, size=size, budget=budget)
    
    # Test with fast diversity selection
    print("\nTesting with fast diversity selection...")
    ba_fast = BudgetAware(distance="l2", random_seed=42, fast_diversity=True)
    epsnet_fast, time_fast = time_function(ba_fast.build_epsnet, points, eps=eps, size=size, budget=budget)
    
    print(f"\nResults:")
    print(f"Standard method: {time_std:.3f}s, epsnet size: {len(epsnet_std)}")
    print(f"Fast method:     {time_fast:.3f}s, epsnet size: {len(epsnet_fast)}")
    
    if time_fast > 0:
        speedup = time_std / time_fast
        print(f"Total speedup: {speedup:.1f}x")
    
    # Verify coverage for both
    print(f"\nCoverage verification:")
    is_valid_std, coverage_std = ba_standard.verify_coverage(points, epsnet_std, eps)
    is_valid_fast, coverage_fast = ba_fast.verify_coverage(points, epsnet_fast, eps)
    
    print(f"Standard method: Valid={is_valid_std}, Coverage={coverage_std:.3f}")
    print(f"Fast method:     Valid={is_valid_fast}, Coverage={coverage_fast:.3f}")


def usage_examples():
    """Show usage examples."""
    print("\n\nUsage Examples")
    print("=" * 60)
    
    print("# For best diversity (slower):")
    print("budget_aware = BudgetAware(distance='l2', fast_diversity=False)")
    print("epsnet = budget_aware.build_epsnet(points, eps=0.1)")
    print()
    
    print("# For faster execution (slightly less diverse):")
    print("budget_aware = BudgetAware(distance='l2', fast_diversity=True)")
    print("epsnet = budget_aware.build_epsnet(points, eps=0.1)")
    print()
    
    print("# The fast_diversity flag affects the internal diversity selection")
    print("# but maintains the same overall algorithm structure and guarantees.")


if __name__ == "__main__":
    test_performance_comparison()
    test_full_epsnet_performance()
    usage_examples()
