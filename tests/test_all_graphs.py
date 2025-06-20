#!/usr/bin/env python3
"""
Test script to run bus_access edge connectivity tests on all graph files.
Provides a summary report for all graphs.
"""
import os
import sys
import subprocess
from pathlib import Path

def test_all_graphs():
    """Test all graphml files in the graphml directory."""
    graphml_dir = Path("graphml")
    if not graphml_dir.exists():
        print(" graphml directory not found!")
        return {}
    
    graph_files = list(graphml_dir.glob("*.graphml"))
    if not graph_files:
        print(" No .graphml files found in graphml directory!")
        return {}
    
    print("=" * 80)
    print("COMPREHENSIVE BUS ACCESS EDGE TESTING")
    print("=" * 80)
    print(f"Found {len(graph_files)} graph files to test:")
    for f in graph_files:
        print(f"  • {f.name}")
    print()
    
    results = {}
    
    for graph_file in sorted(graph_files):
        print(f"Testing {graph_file.name}...")
        print("-" * 60)
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, "tests/test_bus_access_edges.py", str(graph_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                status = " PASS"
            else:
                status = " FAIL"
            
            results[graph_file.name] = {
                'status': status,
                'returncode': result.returncode,
                'output': result.stdout,
                'error': result.stderr
            }
            
            print(f"Status: {status}")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
        except subprocess.TimeoutExpired:
            results[graph_file.name] = {
                'status': " TIMEOUT",
                'returncode': -1,
                'output': "",
                'error': "Test timed out after 30 seconds"
            }
            print("Test timed out")
        except Exception as e:
            results[graph_file.name] = {
                'status': " ERROR",
                'returncode': -1,
                'output': "",
                'error': str(e)
            }
            print(f" Error: {e}")
        
        print()
    
    # Summary report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r['status'] == " PASS")
    failed = len(results) - passed
    
    print(f"Total graphs tested: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if passed == len(results):
        print(" ALL GRAPHS PASSED THE BUS ACCESS EDGE TESTS!")
    else:
        print("  SOME GRAPHS HAVE ISSUES:")
        for name, result in results.items():
            if result['status'] != " PASS":
                print(f"  • {name}: {result['status']}")
    
    print()
    print("Detailed results:")
    for name, result in results.items():
        print(f"  {name}: {result['status']}")
    
    return results

if __name__ == '__main__':
    results = test_all_graphs()
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results.values() if r['status'] != " PASS")
    sys.exit(failed_count) 