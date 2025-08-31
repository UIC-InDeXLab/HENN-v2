#!/usr/bin/env python3
"""
Test runner for HENN Python implementation.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py TestHENN     # Run specific test class
    python run_tests.py --list       # List all available tests
"""

import unittest
import sys
import os
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def list_tests():
    """List all available test classes and methods."""
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(__file__)
    suite = loader.discover(test_dir, pattern="test_*.py")

    print("Available test files and classes:")
    print("=" * 50)

    test_count = 0

    def count_and_list_tests(test_suite, indent=""):
        nonlocal test_count
        for test in test_suite:
            if isinstance(test, unittest.TestSuite):
                count_and_list_tests(test, indent)
            else:
                # This is an individual test case
                class_name = test.__class__.__name__
                module_name = test.__class__.__module__
                method_name = test._testMethodName

                # Print class name only once per class
                class_key = f"{module_name}.{class_name}"
                if not hasattr(list_tests, "_printed_classes"):
                    list_tests._printed_classes = set()

                if class_key not in list_tests._printed_classes:
                    print(f"\n{module_name}.py::{class_name}")
                    list_tests._printed_classes.add(class_key)

                print(f"  - {method_name}")
                test_count += 1

    count_and_list_tests(suite)
    print(f"\nTotal tests found: {test_count}")
    return test_count


def run_tests():
    """Discover and run all tests."""
    start_time = time.time()

    # Handle command line arguments
    if "--list" in sys.argv:
        list_tests()
        return 0

    print("HENN Test Suite Runner")
    print("=" * 50)

    # Set up test discovery
    loader = unittest.TestLoader()

    # Discover tests in the current directory
    test_dir = os.path.dirname(__file__)

    # If specific test class is specified, run only that
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        test_pattern = sys.argv[1]
        print(f"Running specific test: {test_pattern}")
        try:
            suite = loader.loadTestsFromName(test_pattern)
        except Exception as e:
            print(f"Error loading test {test_pattern}: {e}")
            print("Use --list to see available tests")
            return 1
    else:
        print("Discovering all tests in tests directory...")
        suite = loader.discover(test_dir, pattern="test_*.py")

        # Count total tests
        test_count = suite.countTestCases()
        print(f"Found {test_count} tests")

    print("-" * 50)

    # Set up test runner
    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        buffer=True,  # Capture stdout/stderr during tests
    )

    # Run tests
    print("Running tests...")
    result = runner.run(suite)

    end_time = time.time()
    duration = end_time - start_time

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Duration: {duration:.2f} seconds")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} TESTS FAILED!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = run_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
