import numpy as np
import os
import urllib.request
import gzip
import tarfile
import h5py
from typing import Tuple, Optional
from sklearn.preprocessing import normalize


class ANNDatasetLoader:
    """
    A comprehensive loader for standard ANN benchmark datasets.

    Supports loading:
    - glove-100-angular
    - glove-25-angular
    - nytimes-256-angular
    - fashion-mnist-784-euclidean
    - gist-960-euclidean
    - sift-128-euclidean

    All datasets are returned as numpy arrays.
    """

    def __init__(self, data_dir: str = "./datasets/ann_datasets"):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Dataset metadata
        self.dataset_info = {
            "glove-100-angular": {
                "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
                "filename": "glove-100-angular.hdf5",
                "distance": "angular",
            },
            "glove-25-angular": {
                "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
                "filename": "glove-25-angular.hdf5",
                "distance": "angular",
            },
            "nytimes-256-angular": {
                "url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
                "filename": "nytimes-256-angular.hdf5",
                "distance": "angular",
            },
            "fashion-mnist-784-euclidean": {
                "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
                "filename": "fashion-mnist-784-euclidean.hdf5",
                "distance": "euclidean",
            },
            "gist-960-euclidean": {
                "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
                "filename": "gist-960-euclidean.hdf5",
                "distance": "euclidean",
            },
            "sift-128-euclidean": {
                "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
                "filename": "sift-128-euclidean.hdf5",
                "distance": "euclidean",
            },
        }

    def _download_dataset(self, dataset_name: str) -> str:
        """
        Download dataset if not already present.

        Args:
            dataset_name: Name of the dataset to download

        Returns:
            Path to the downloaded file
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        info = self.dataset_info[dataset_name]
        filepath = os.path.join(self.data_dir, info["filename"])

        if not os.path.exists(filepath):
            print(f"Downloading {dataset_name}...")
            try:
                urllib.request.urlretrieve(info["url"], filepath)
                print(f"Downloaded {dataset_name} to {filepath}")
            except Exception as e:
                print(f"Error downloading {dataset_name}: {e}")
                print(f"Please manually download from {info['url']} to {filepath}")
                raise
        else:
            # Verify file integrity by trying to open it
            try:
                with h5py.File(filepath, "r") as f:
                    # Just check if we can read the file structure
                    list(f.keys())
            except Exception as e:
                print(f"Corrupted file detected for {dataset_name}. Re-downloading...")
                os.remove(filepath)
                try:
                    urllib.request.urlretrieve(info["url"], filepath)
                    print(f"Re-downloaded {dataset_name} to {filepath}")
                except Exception as download_error:
                    print(f"Error re-downloading {dataset_name}: {download_error}")
                    raise

        return filepath

    def _load_hdf5_dataset(
        self, filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset from HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Tuple of (train_data, test_data, neighbors)
        """
        with h5py.File(filepath, "r") as f:
            train = np.array(f["train"])
            test = np.array(f["test"])
            neighbors = np.array(f["neighbors"])

        return train, test, neighbors

    def load_glove_100_angular(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load GloVe 100-dimensional angular dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("glove-100-angular")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Normalize for angular distance
        train = normalize(train, norm="l2", axis=1)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = normalize(test, norm="l2", axis=1)
            return train, test, neighbors
        else:
            return (train,)

    def load_glove_25_angular(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load GloVe 25-dimensional angular dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("glove-25-angular")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Normalize for angular distance
        train = normalize(train, norm="l2", axis=1)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = normalize(test, norm="l2", axis=1)
            return train, test, neighbors
        else:
            return (train,)

    def load_nytimes_256_angular(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load NYTimes 256-dimensional angular dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("nytimes-256-angular")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Normalize for angular distance
        train = normalize(train, norm="l2", axis=1)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = normalize(test, norm="l2", axis=1)
            return train, test, neighbors
        else:
            return (train,)

    def load_fashion_mnist_784_euclidean(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load Fashion-MNIST 784-dimensional euclidean dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("fashion-mnist-784-euclidean")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Convert to float32 for efficiency
        train = train.astype(np.float32)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = test.astype(np.float32)
            return train, test, neighbors
        else:
            return (train,)

    def load_gist_960_euclidean(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load GIST 960-dimensional euclidean dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("gist-960-euclidean")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Convert to float32 for efficiency
        train = train.astype(np.float32)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = test.astype(np.float32)
            return train, test, neighbors
        else:
            return (train,)

    def load_sift_128_euclidean(
        self, return_test: bool = False, subset_size: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Load SIFT 128-dimensional euclidean dataset.

        Args:
            return_test: Whether to return test queries and ground truth
            subset_size: Number of training samples to return (None for full dataset)

        Returns:
            If return_test=False: (train_data,)
            If return_test=True: (train_data, test_queries, ground_truth_neighbors)
        """
        filepath = self._download_dataset("sift-128-euclidean")
        train, test, neighbors = self._load_hdf5_dataset(filepath)

        # Convert to float32 for efficiency
        train = train.astype(np.float32)

        # Apply subset if requested
        if subset_size is not None:
            train = train[:subset_size]

        if return_test:
            test = test.astype(np.float32)
            return train, test, neighbors
        else:
            return (train,)

    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return self.dataset_info[dataset_name]

    def list_available_datasets(self) -> list:
        """
        List all available datasets.

        Returns:
            List of dataset names
        """
        return list(self.dataset_info.keys())


# Convenience functions for direct loading
def load_glove_100_angular(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load GloVe 100-dimensional angular dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_glove_100_angular(return_test, subset_size)


def load_glove_25_angular(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load GloVe 25-dimensional angular dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_glove_25_angular(return_test, subset_size)


def load_nytimes_256_angular(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load NYTimes 256-dimensional angular dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_nytimes_256_angular(return_test, subset_size)


def load_fashion_mnist_784_euclidean(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load Fashion-MNIST 784-dimensional euclidean dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_fashion_mnist_784_euclidean(return_test, subset_size)


def load_gist_960_euclidean(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load GIST 960-dimensional euclidean dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_gist_960_euclidean(return_test, subset_size)


def load_sift_128_euclidean(
    data_dir: str = "./datasets/ann_datasets",
    return_test: bool = False,
    subset_size: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Load SIFT 128-dimensional euclidean dataset."""
    loader = ANNDatasetLoader(data_dir)
    return loader.load_sift_128_euclidean(return_test, subset_size)


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ANNDatasetLoader()

    # List available datasets
    print("Available datasets:")
    for dataset in loader.list_available_datasets():
        info = loader.get_dataset_info(dataset)
        print(f"  - {dataset} ({info['distance']} distance)")

    # Example: Load SIFT dataset
    print("\nLoading SIFT dataset...")
    data = load_sift_128_euclidean()
    print(f"SIFT train data shape: {data[0].shape}")
    print(f"SIFT data type: {data[0].dtype}")

    # Example: Load subset of SIFT dataset
    print("\nLoading SIFT dataset subset (first 1000 samples)...")
    data_subset = load_sift_128_euclidean(subset_size=1000)
    print(f"SIFT subset train data shape: {data_subset[0].shape}")

    # Example: Load with test data
    print("\nLoading GloVe-100 with test data...")
    train, test, neighbors = load_glove_100_angular(return_test=True)
    print(f"GloVe-100 train shape: {train.shape}")
    print(f"GloVe-100 test shape: {test.shape}")
    print(f"GloVe-100 neighbors shape: {neighbors.shape}")

    # Example: Load subset with test data
    print("\nLoading GloVe-100 subset (first 5000 samples) with test data...")
    train_subset, test, neighbors = load_glove_100_angular(
        return_test=True, subset_size=5000
    )
    print(f"GloVe-100 subset train shape: {train_subset.shape}")
    print(f"GloVe-100 test shape: {test.shape}")
    print(f"GloVe-100 neighbors shape: {neighbors.shape}")
