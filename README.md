# HENN-v2

**Hierarchical EpsNet Navigation Graph for Approximate Nearest Neighbor Search with Guarantees**

This repository contains both Python and C++ implementations of HENN (Hierarchical EpsNet Navigation Graph), a state-of-the-art algorithm for approximate nearest neighbor search with theoretical guarantees.

## Overview

HENN is an efficient algorithm that combines the concepts of EpsNet and proximity graphs to provide fast approximate nearest neighbor search with provable guarantees. The hierarchical structure enables scalable search across large datasets while maintaining accuracy bounds.

## Repository Structure

### `henn-python/`
Python implementation of HENN algorithms including:
- **EpsNet algorithms**: Random sample and budget-aware variants
- **Main HENN implementation**: Located in `henn.py`
- **Proximity graphs**: Implementation in `pgraphs/` directory
- **Datasets**: Test datasets in `datasets/`
- **Tests**: Unit tests in `tests/`

### `henn-faiss/`
C++ implementation integrated with the FAISS library:
- **Core HENN implementations**: `faiss/HENN.cpp` and `faiss/IndexHENN.cpp`
- **Examples and benchmarks**: Located in `experiments/` directory
- **Build system**: CMake configuration and build scripts
- **Documentation**: API documentation and installation guides

## Getting Started

### Python Implementation
```bash
cd henn-python
pip install -r requirements.txt
python henn.py
```

### C++ Implementation
```bash
cd henn-faiss
./build_henn_faiss.sh
```

See the respective directories for detailed installation and usage instructions.