#!/usr/bin/env python3
"""
Test runner for HENN Python implementation.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py TestHENN     # Run specific test class
"""

import unittest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_tests():
    """Discover and run all tests."""
    # Set up test discovery
    loader = unittest.TestLoader()

    # Discover tests in the current directory
    test_dir = os.path.dirname(__file__)
    suite = loader.discover(test_dir, pattern="test_*.py")

    # Set up test runner
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)

    # Run tests
    result = runner.run(suite)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
