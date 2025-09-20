#!/usr/bin/env python3
"""
Example usage of NSG with different medoid finding optimization methods.
"""

import numpy as np
from pgraphs.nsg import NSG


def example_usage():
    """Demonstrate how to use different medoid optimization methods."""
    
    # Generate some example data
    np.random.seed(42)
    n_points = 1000
    n_dims = 128
    henn_points = np.random.randn(n_points, n_dims)
    layer_indices = list(range(n_points))
    
    print("NSG Medoid Optimization Methods Example")
    print("=" * 50)
    
    # Example 1: Default (original) method
    print("\n1. Using default (original) method:")
    nsg_original = NSG(distance="l2")  # medoid_method defaults to "original"
    # or explicitly: nsg_original = NSG(distance="l2", medoid_method="original")
    
    # Build graph (this will use the original medoid finding method)
    graph_original = nsg_original.build_graph(henn_points, layer_indices)
    print(f"   Built graph with {len(graph_original)} nodes")
    print(f"   Navigation node (medoid): {nsg_original.init_node}")
    
    # Example 2: Vectorized method (faster)
    print("\n2. Using vectorized method (recommended for medium datasets):")
    nsg_vectorized = NSG(
        distance="l2",
        medoid_method="vectorized"
    )
    
    graph_vectorized = nsg_vectorized.build_graph(henn_points, layer_indices)
    print(f"   Built graph with {len(graph_vectorized)} nodes")
    print(f"   Navigation node (medoid): {nsg_vectorized.init_node}")
    
    # Example 3: Sampling method (fastest for large datasets)
    print("\n3. Using sampling method (recommended for large datasets):")
    nsg_sampling = NSG(
        distance="l2",
        medoid_method="sampling",
        medoid_sample_size=500  # Sample 500 candidates
    )
    
    graph_sampling = nsg_sampling.build_graph(henn_points, layer_indices)
    print(f"   Built graph with {len(graph_sampling)} nodes")
    print(f"   Navigation node (medoid): {nsg_sampling.init_node}")
    
    # Example 4: Cosine distance with vectorized method
    print("\n4. Using cosine distance with vectorized method:")
    # Normalize points for cosine distance
    henn_points_normalized = henn_points / np.linalg.norm(henn_points, axis=1, keepdims=True)
    
    nsg_cosine = NSG(
        distance="cosine",
        medoid_method="vectorized"
    )
    
    graph_cosine = nsg_cosine.build_graph(henn_points_normalized, layer_indices)
    print(f"   Built graph with {len(graph_cosine)} nodes")
    print(f"   Navigation node (medoid): {nsg_cosine.init_node}")
    
    print("\n" + "=" * 50)
    print("Method Selection Guidelines:")
    print("- original: Most accurate, use for small datasets (<1000 points)")
    print("- vectorized: Best balance of speed/accuracy (1000-10000 points)")
    print("- sampling: Fastest approximation for large datasets (>10000 points)")
    print("\nNote: The 'original' method is the default to maintain backward compatibility.")


if __name__ == "__main__":
    example_usage()
