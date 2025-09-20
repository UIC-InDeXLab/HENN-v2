#!/usr/bin/env python3
"""
Test script to verify both partition methods in BudgetAware algorithm.
"""

import numpy as np
import sys
import os

# Add the henn-python directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from epsnet.budgetaware import BudgetAware

def test_budgetaware_methods():
    """Test the new BudgetAware implementation with diverse sampling and unhit balls."""
    
    # Generate some test data
    np.random.seed(42)
    n_points = 100
    dimensions = 2
    points = np.random.randn(n_points, dimensions)
    
    eps = 0.1
    size = 20
    budget = 5
    
    print("Testing BudgetAware with new unhit balls method")
    print(f"Dataset: {n_points} points, {dimensions} dimensions")
    print(f"Target size: {size}, Budget: {budget}, Eps: {eps}")
    print("=" * 60)
    
    # Test with new implementation
    print("\n1. Testing new diverse sampling + unhit balls method:")
    budgetaware = BudgetAware(random_seed=42)
    
    try:
        epsnet = budgetaware.build_epsnet(
            points, eps=eps, size=size, budget=budget
        )
        print(f"   Result: Selected {len(epsnet)} points")
        print(f"   Indices: {sorted(epsnet[:10])}{'...' if len(epsnet) > 10 else ''}")
        
        # Test coverage verification
        print("\n2. Coverage verification:")
        is_valid, coverage = budgetaware.verify_coverage(points, epsnet, eps)
        print(f"   Valid eps-net: {is_valid}, Coverage: {coverage:.3f}")
        
        # Test with size-only parameter
        print("\n3. Testing with size parameter only:")
        epsnet_size_only = budgetaware.build_epsnet(points, size=size, budget=budget)
        print(f"   Result: Selected {len(epsnet_size_only)} points")
        print(f"   Indices: {sorted(epsnet_size_only[:10])}{'...' if len(epsnet_size_only) > 10 else ''}")
        
        # Test diverse point selection
        print("\n4. Testing diverse point selection:")
        diverse_points = budgetaware._select_diverse_points(points, 10)
        print(f"   Selected {len(diverse_points)} diverse points: {diverse_points}")
        
        # Test ball finding
        print("\n5. Testing ball finding:")
        ball_points = budgetaware._find_ball_points(points, 0, 15)
        print(f"   Ball around point 0 with size 15: {len(ball_points)} points")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_budgetaware_methods()
