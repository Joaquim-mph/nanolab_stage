#!/usr/bin/env python3
"""
Test if worker processes can even start and return simple results.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def simple_task(x):
    """Simplest possible task - just return the input."""
    return x * 2


def test_basic_multiprocessing():
    """Test if ProcessPoolExecutor works at all."""
    print("Testing basic ProcessPoolExecutor functionality...")
    print()

    with ProcessPoolExecutor(max_workers=8) as executor:
        print("Submitting 100 simple tasks...")
        futures = {executor.submit(simple_task, i): i for i in range(100)}

        print("Waiting for results with as_completed()...")
        results = []
        for future in as_completed(futures):
            result = future.result(timeout=5)
            results.append(result)
            if len(results) % 10 == 0:
                print(f"  Completed {len(results)}/100")

        print()
        print(f"✓ All {len(results)} tasks completed successfully!")
        print(f"  First few results: {sorted(results)[:10]}")


if __name__ == "__main__":
    test_basic_multiprocessing()
