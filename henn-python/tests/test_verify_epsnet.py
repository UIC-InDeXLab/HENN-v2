#!/usr/bin/env python3
"""
Test script to verify the eps-net verification functionality.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from epsnet.base_epsnet import BaseEPSNet

def test_verify_epsnet():
    """Test the verify method with simple cases."""
    
    # Test case 1: Simple 2D points
    print("Test Case 1: Simple 2D points")
    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 2]
    ])
    
    # Valid eps-net: all points with eps=2.0 (should cover everything)
    eps = 2.0
    epsnet_all = [0, 1, 2, 3, 4]  # All points
    result = BaseEPSNet.verify(epsnet_all, points, eps)
    print(f"All points as eps-net (eps={eps}): {result}")  # Should be True
    
    # Valid eps-net: single point that can cover all with large enough eps
    eps = 3.0
    epsnet_single = [0]  # Just the first point
    result = BaseEPSNet.verify(epsnet_single, points, eps)
    print(f"Single point as eps-net (eps={eps}): {result}")  # Should be True
    
    # Invalid eps-net: single point with small eps
    eps = 0.5
    epsnet_single_small = [0]  # Just the first point
    result = BaseEPSNet.verify(epsnet_single_small, points, eps)
    print(f"Single point as eps-net (eps={eps}): {result}")  # Should be False
    
    # Valid eps-net: two strategic points
    eps = 1.5
    epsnet_two = [0, 4]  # First and last points
    result = BaseEPSNet.verify(epsnet_two, points, eps)
    print(f"Two strategic points as eps-net (eps={eps}): {result}")  # Should be True
    
    print()

def test_verify_epsnet_cosine():
    """Test with cosine distance."""
    print("Test Case 2: Cosine distance")
    
    # Normalized vectors for cosine distance
    points = np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [0.707, 0.707]  # 45 degrees
    ])
    
    # Normalize the vectors
    for i in range(len(points)):
        points[i] = points[i] / np.linalg.norm(points[i])
    
    eps = 0.5  # Small epsilon for cosine distance
    epsnet = [0, 1, 2, 3]  # Four cardinal directions
    result = BaseEPSNet.verify(epsnet, points, eps, distance="cosine")
    print(f"Four cardinal directions as eps-net (cosine, eps={eps}): {result}")
    
    # Larger epsilon that should cover all
    eps = 1.5
    epsnet_single = [0]  # Just one point
    result = BaseEPSNet.verify(epsnet_single, points, eps, distance="cosine")
    print(f"Single point as eps-net (cosine, eps={eps}): {result}")
    
    print()

def test_edge_cases():
    """Test edge cases."""
    print("Test Case 3: Edge cases")
    
    # Single point
    points = np.array([[0, 0]])
    epsnet = [0]
    result = BaseEPSNet.verify(epsnet, points, 1.0)
    print(f"Single point dataset: {result}")  # Should be True
    
    # Empty epsnet (should fail unless no points)
    points = np.array([[0, 0], [1, 1]])
    epsnet = []
    result = BaseEPSNet.verify(epsnet, points, 1.0)
    print(f"Empty eps-net: {result}")  # Should be False
    
    print()

if __name__ == "__main__":
    test_verify_epsnet()
    test_verify_epsnet_cosine()
    test_edge_cases()
    print("All tests completed!")
