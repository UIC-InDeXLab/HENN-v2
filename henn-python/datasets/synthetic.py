import numpy as np
from typing import Tuple, Literal
from sklearn.preprocessing import normalize


def generate_synthetic_dataset(
    n_samples: int,
    dimension: int,
    distribution: Literal["random", "mixture_gaussians"] = "random",
    distance_metric: Literal["cosine", "l2"] = "l2",
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for ANN search.

    Args:
        n_samples: Number of samples to generate
        dimension: Dimensionality of the vectors
        distribution: Type of distribution ("random" or "mixture_gaussians")
        distance_metric: Distance metric ("cosine" or "l2")
        n_clusters: Number of clusters for mixture of gaussians
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (data, labels) where labels indicate cluster membership
    """
    np.random.seed(random_state)

    if distribution == "random":
        # Generate random uniform data
        data = np.random.uniform(-1, 1, size=(n_samples, dimension))
        labels = np.zeros(n_samples)  # No meaningful clusters

    elif distribution == "mixture_gaussians":
        # Generate mixture of anisotropic gaussians
        data = []
        labels = []
        samples_per_cluster = n_samples // n_clusters

        for i in range(n_clusters):
            # Random center for each cluster
            center = np.random.uniform(-2, 2, dimension)

            # Create anisotropic covariance matrix
            # Random eigenvalues with different scales
            eigenvals = np.random.uniform(0.1, 2.0, dimension)
            # Random rotation matrix
            Q = np.random.randn(dimension, dimension)
            Q, _ = np.linalg.qr(Q)

            # Construct covariance matrix: Q * diag(eigenvals) * Q.T
            cov_matrix = Q @ np.diag(eigenvals) @ Q.T

            # Generate samples from this gaussian
            n_samples_cluster = samples_per_cluster + (
                1 if i < n_samples % n_clusters else 0
            )
            cluster_data = np.random.multivariate_normal(
                center, cov_matrix, n_samples_cluster
            )

            data.append(cluster_data)
            labels.extend([i] * n_samples_cluster)

        data = np.vstack(data)
        labels = np.array(labels)

    # Normalize for cosine distance
    if distance_metric == "cosine":
        data = normalize(data, norm="l2", axis=1)

    return data


# Example usage
if __name__ == "__main__":
    # Generate random dataset for L2 distance
    X_random = generate_synthetic_dataset(
        n_samples=1000, dimension=128, distribution="random", distance_metric="l2"
    )

    # Generate mixture of gaussians for cosine similarity
    X_gauss = generate_synthetic_dataset(
        n_samples=1000,
        dimension=128,
        distribution="mixture_gaussians",
        distance_metric="cosine",
        n_clusters=10,
    )

    print(f"Random dataset shape: {X_random.shape}")
    print(f"Gaussian mixture dataset shape: {X_gauss.shape}")
