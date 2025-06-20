#!/usr/bin/env python3
"""
Test script to verify bus stop connectivity validation.
Tests the validate_bus_stop_connectivity function to ensure it correctly
identifies problematic bus stops (dead-ends, spurs, etc.).
"""
import sys
import networkx as nx
from typing import Dict, List, Tuple
from rota_aco.data.preprocess import load_graph, get_bus_stops
from rota_aco.graph.build_meta import validate_bus_stop_connectivity


def create_test_graph() -> Tuple[nx.MultiDiGraph, List]:
    """
    Create a test graph with various bus stop connectivity patterns:
    - Valid stops with through-traffic
    - Dead-end stops
    - Spur stops (both edges lead to same node)
    - Loop stops (in and out same way)
    """
    G = nx.MultiDiGraph()
    
    # Add nodes with coordinates
    nodes = {
        'A': (0, 0),    # Main intersection
        'B': (1, 0),    # Valid bus stop
        'C': (2, 0),    # Valid bus stop
        'D': (0, 1),    # Dead-end bus stop
        'E': (1, 1),    # Spur bus stop
        'F': (2, 1),    # Loop bus stop
        'G': (3, 0),    # Through node
        'H': (4, 0),    # Through node
    }
    
    for node, (x, y) in nodes.items():
        G.add_node(node, x=x, y=y)
        if node in ['B', 'C', 'D', 'E', 'F']:
            G.nodes[node]['bus_stop'] = 'true'
    
    # Add edges
    # Main road: A -> B -> C -> G -> H
    G.add_edge('A', 'B', highway='primary', length=100)
    G.add_edge('B', 'C', highway='primary', length=100)
    G.add_edge('C', 'G', highway='primary', length=100)
    G.add_edge('G', 'H', highway='primary', length=100)
    
    # Bus access edges for valid stops
    G.add_edge('A', 'B', highway='bus_access', length=50)
    G.add_edge('B', 'A', highway='bus_access', length=50)
    G.add_edge('B', 'C', highway='bus_access', length=50)
    G.add_edge('C', 'B', highway='bus_access', length=50)
    G.add_edge('C', 'G', highway='bus_access', length=50)
    G.add_edge('G', 'C', highway='bus_access', length=50)
    
    # Dead-end bus stop D (only one connection)
    G.add_edge('A', 'D', highway='bus_access', length=50)
    G.add_edge('D', 'A', highway='bus_access', length=50)
    
    # Spur bus stop E (both edges lead to same node)
    G.add_edge('A', 'E', highway='bus_access', length=50)
    G.add_edge('E', 'A', highway='bus_access', length=50)
    G.add_edge('A', 'E', highway='bus_access', length=50)  # Second edge to same node
    G.add_edge('E', 'A', highway='bus_access', length=50)  # Second edge to same node
    
    # Loop bus stop F (in and out same way, no through-traffic)
    G.add_edge('A', 'F', highway='bus_access', length=50)
    G.add_edge('F', 'A', highway='bus_access', length=50)
    
    bus_stops = ['B', 'C', 'D', 'E', 'F']
    return G, bus_stops


def test_connectivity_validation():
    """Test the connectivity validation function."""
    print("Testing bus stop connectivity validation...")
    print("=" * 60)
    
    # Create test graph
    G, bus_stops = create_test_graph()
    print(f"Created test graph with {len(bus_stops)} bus stops")
    
    # Run validation
    valid_stops, problematic_stops = validate_bus_stop_connectivity(G, bus_stops)
    
    # Expected results
    expected_valid = ['B', 'C']  # These should have through-traffic
    expected_problematic = ['D', 'E', 'F']  # These should be problematic
    
    print(f"\nVALIDATION RESULTS:")
    print(f"Valid stops found: {valid_stops}")
    print(f"Problematic stops found: {problematic_stops}")
    print(f"Expected valid: {expected_valid}")
    print(f"Expected problematic: {expected_problematic}")
    
    # Check results
    valid_correct = set(valid_stops) == set(expected_valid)
    problematic_correct = set(problematic_stops) == set(expected_problematic)
    
    print(f"\nVALIDATION CHECK:")
    print(f"Valid stops correct: {valid_correct}")
    print(f"Problematic stops correct: {problematic_correct}")
    
    if valid_correct and problematic_correct:
        print(" All tests passed!")
        return True
    else:
        print(" Some tests failed!")
        return False


def test_real_graph(graphml_path: str):
    """Test connectivity validation on a real graph file."""
    print(f"\nTesting real graph: {graphml_path}")
    print("=" * 60)
    
    try:
        from rota_aco.data.preprocess import load_graph, get_bus_stops
        
        # Load graph
        G = load_graph(graphml_path)
        bus_stops = get_bus_stops(G)
        
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"Found {len(bus_stops)} bus stops")
        
        # Run validation
        valid_stops, problematic_stops = validate_bus_stop_connectivity(G, bus_stops)
        
        print(f"\nVALIDATION RESULTS:")
        print(f"Valid stops: {len(valid_stops)}/{len(bus_stops)} ({len(valid_stops)/len(bus_stops)*100:.1f}%)")
        print(f"Problematic stops: {len(problematic_stops)}")
        
        if problematic_stops:
            print(f"\nProblematic stops:")
            for stop in problematic_stops:
                print(f"  • {stop}")
        else:
            print(f"\n All bus stops have valid connectivity!")
        
        return len(problematic_stops) == 0
        
    except Exception as e:
        print(f"Error testing real graph: {e}")
        return False


def main():
    """Main test function."""
    print("BUS STOP CONNECTIVITY VALIDATION TEST")
    print("=" * 60)
    
    # Test with synthetic graph
    synthetic_passed = test_connectivity_validation()
    
    # Test with real graph if provided
    real_passed = True
    if len(sys.argv) > 1:
        graphml_file = sys.argv[1]
        real_passed = test_real_graph(graphml_file)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Synthetic test: {'PASS' if synthetic_passed else 'FAIL'}")
    if len(sys.argv) > 1:
        print(f"Real graph test: {'PASS' if real_passed else 'FAIL'}")
    
    all_passed = synthetic_passed and real_passed
    print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 