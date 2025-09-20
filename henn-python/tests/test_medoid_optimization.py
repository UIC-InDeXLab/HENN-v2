#!/usr/bin/env python3
"""
Test script to compare medoid finding performance between different optimization methods.
"""

import numpy as np
import time
from pgraphs.nsg import NSG


def generate_test_data(n_points=1000, n_dims=128, distance_type="l2"):
    """Generate synthetic test data."""
    np.random.seed(42)  # For reproducible results
    
    if distance_type == "cosine":
        # Generate random vectors and normalize for cosine distance
        points = np.random.randn(n_points, n_dims)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    else:
        # Generate random vectors for L2 distance
        points = np.random.randn(n_points, n_dims)
    
    layer_indices = list(range(n_points))
    return points, layer_indices


def benchmark_medoid_methods(n_points=1000, n_dims=128, distance_type="l2"):
    """Benchmark different medoid finding methods."""
    print(f"Benchmarking medoid finding methods:")
    print(f"Dataset: {n_points} points, {n_dims} dimensions, {distance_type} distance")
    print("-" * 60)
    
    # Generate test data
    points, layer_indices = generate_test_data(n_points, n_dims, distance_type)
    
    methods = ["original", "vectorized", "sampling"]
    results = {}
    
    for method in methods:
        print(f"Testing {method} method...")
        
        # Create NSG instance with specific method
        nsg = NSG(
            distance=distance_type,
            medoid_method=method,
            medoid_sample_size=min(500, n_points // 2)
        )
        
        # Measure time
        start_time = time.time()
        medoid_idx = nsg._find_medoid(points, layer_indices)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        results[method] = {
            'time': elapsed_time,
            'medoid_idx': medoid_idx
        }
        
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  Medoid index: {medoid_idx}")
        print()
    
    # Compare speedups
    original_time = results['original']['time']
    print("Performance comparison:")
    print(f"Original method: {original_time:.4f}s (baseline)")
    
    for method in ['vectorized', 'sampling']:
        speedup = original_time / results[method]['time']
        print(f"{method.capitalize()} method: {results[method]['time']:.4f}s "
              f"({speedup:.2f}x speedup)")
    
    return results


def verify_medoid_quality(n_points=500, n_dims=64, distance_type="l2"):
    """Verify that optimized methods find reasonable medoids."""
    print(f"\nVerifying medoid quality:")
    print(f"Dataset: {n_points} points, {n_dims} dimensions, {distance_type} distance")
    print("-" * 60)
    
    points, layer_indices = generate_test_data(n_points, n_dims, distance_type)
    
    methods = ["original", "vectorized", "sampling"]
    medoids = {}
    
    # Find medoids using different methods
    for method in methods:
        nsg = NSG(
            distance=distance_type,
            medoid_method=method,
            medoid_sample_size=min(300, n_points // 2)
        )
        medoids[method] = nsg._find_medoid(points, layer_indices)
    
    # Calculate sum of distances for each medoid found
    def calculate_medoid_quality(medoid_idx):
        total_dist = 0.0
        medoid_point = points[medoid_idx]
        
        for idx in layer_indices:
            if idx != medoid_idx:
                point = points[idx]
                if distance_type == "cosine":
                    dist = 1 - np.dot(medoid_point, point)
                else:
                    dist = np.linalg.norm(medoid_point - point)
                total_dist += dist
        
        return total_dist
    
    print("Medoid quality (sum of distances to all other points):")
    qualities = {}
    for method in methods:
        quality = calculate_medoid_quality(medoids[method])
        qualities[method] = quality
        print(f"{method.capitalize()}: {quality:.4f} (medoid index: {medoids[method]})")
    
    # Compare quality
    original_quality = qualities['original']
    print("\nQuality comparison (lower is better):")
    for method in methods:
        if method == 'original':
            print(f"{method.capitalize()}: baseline")
        else:
            quality_ratio = qualities[method] / original_quality
            print(f"{method.capitalize()}: {quality_ratio:.4f} relative quality "
                  f"({'better' if quality_ratio < 1 else 'worse'})")


if __name__ == "__main__":
    print("NSG Medoid Finding Optimization Test")
    print("=" * 50)
    
    # Test with different dataset sizes and distance metrics
    test_configs = [
        (500, 64, "l2"),
        (1000, 128, "l2"),
        (500, 64, "cosine"),
        (1000, 128, "cosine"),
    ]
    
    for n_points, n_dims, distance_type in test_configs:
        print(f"\n{'='*50}")
        benchmark_medoid_methods(n_points, n_dims, distance_type)
        
        if n_points <= 500:  # Only verify quality for smaller datasets
            verify_medoid_quality(n_points, n_dims, distance_type)
    
    print(f"\n{'='*50}")
    print("Recommendations:")
    print("- Use 'original' for small datasets (<1000 points) or when exact medoid is crucial")
    print("- Use 'vectorized' for medium datasets (1000-10000 points) for best speed/accuracy tradeoff")
    print("- Use 'sampling' for large datasets (>10000 points) when speed is critical")
