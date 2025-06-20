#!/usr/bin/env python3
"""
Test script to check if all bus stops have proper bus_access edge connectivity.
Checks for:
1. Each bus stop has at least one incoming bus_access edge
2. Each bus stop has at least one outgoing bus_access edge
3. Identifies bus stops with missing edges
4. Provides statistics on bus_access edge distribution
5. Checks for potential issues in bus_access edge generation
"""
import sys
import networkx as nx
from typing import Dict, List, Set, Tuple
from rota_aco.data.preprocess import load_graph, get_bus_stops
from rota_aco.data.opposites import is_bus_access_edge


def analyze_bus_access_connectivity(G: nx.MultiDiGraph, bus_stops: List) -> Dict:
    """
    Analyze bus_access edge connectivity for all bus stops.
    
    Returns:
        Dictionary with analysis results including:
        - stops_with_incoming: set of stops with incoming bus_access edges
        - stops_with_outgoing: set of stops with outgoing bus_access edges
        - stops_with_both: set of stops with both incoming and outgoing
        - stops_missing_incoming: set of stops missing incoming edges
        - stops_missing_outgoing: set of stops missing outgoing edges
        - stops_missing_both: set of stops missing both
        - edge_counts: dict mapping stop -> (incoming_count, outgoing_count)
    """
    stops_with_incoming = set()
    stops_with_outgoing = set()
    edge_counts = {}
    
    for stop in bus_stops:
        incoming_count = 0
        outgoing_count = 0
        
        # Check incoming edges
        for pred, _, data in G.in_edges(stop, data=True):
            if is_bus_access_edge(data):
                incoming_count += 1
                stops_with_incoming.add(stop)
        
        # Check outgoing edges
        for _, succ, data in G.out_edges(stop, data=True):
            if is_bus_access_edge(data):
                outgoing_count += 1
                stops_with_outgoing.add(stop)
        
        edge_counts[stop] = (incoming_count, outgoing_count)
    
    stops_with_both = stops_with_incoming & stops_with_outgoing
    stops_missing_incoming = set(bus_stops) - stops_with_incoming
    stops_missing_outgoing = set(bus_stops) - stops_with_outgoing
    stops_missing_both = set(bus_stops) - stops_with_incoming - stops_with_outgoing
    
    return {
        'stops_with_incoming': stops_with_incoming,
        'stops_with_outgoing': stops_with_outgoing,
        'stops_with_both': stops_with_both,
        'stops_missing_incoming': stops_missing_incoming,
        'stops_missing_outgoing': stops_missing_outgoing,
        'stops_missing_both': stops_missing_both,
        'edge_counts': edge_counts
    }


def check_bus_access_edge_quality(G: nx.MultiDiGraph, bus_stops: List) -> Dict:
    """
    Check for potential issues in bus_access edge generation.
    
    Returns:
        Dictionary with quality check results including:
        - self_loops: list of bus stops with self-loops
        - missing_reverse_edges: list of edges missing reverse direction
        - inconsistent_oneway: list of edges with inconsistent oneway flags
        - edge_attributes: dict with edge attribute statistics
    """
    self_loops = []
    missing_reverse_edges = []
    inconsistent_oneway = []
    edge_attributes = {}
    
    # Check for self-loops on bus stops
    for stop in bus_stops:
        if G.has_edge(stop, stop):
            self_loops.append(stop)
    
    # Check for missing reverse edges and oneway consistency
    bus_access_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if is_bus_access_edge(data):
            bus_access_edges.append((u, v, key, data))
            
            # Check if reverse edge exists
            if not G.has_edge(v, u, key):
                missing_reverse_edges.append((u, v, key))
            
            # Check oneway consistency
            oneway = data.get('oneway', False)
            if oneway:
                inconsistent_oneway.append((u, v, key, "Should not be oneway for bus_access"))
    
    # Analyze edge attributes
    if bus_access_edges:
        sample_edge = bus_access_edges[0]
        edge_attributes = {
            'total_bus_access_edges': len(bus_access_edges),
            'sample_attributes': dict(sample_edge[3]),
            'oneway_count': sum(1 for _, _, _, data in bus_access_edges if data.get('oneway', False)),
            'length_stats': {
                'min': min(data.get('length', 0) for _, _, _, data in bus_access_edges),
                'max': max(data.get('length', 0) for _, _, _, data in bus_access_edges),
                'avg': sum(data.get('length', 0) for _, _, _, data in bus_access_edges) / len(bus_access_edges)
            }
        }
    
    return {
        'self_loops': self_loops,
        'missing_reverse_edges': missing_reverse_edges,
        'inconsistent_oneway': inconsistent_oneway,
        'edge_attributes': edge_attributes
    }


def print_connectivity_report(analysis: Dict, bus_stops: List, G: nx.MultiDiGraph):
    """Print a detailed connectivity report."""
    print("=" * 60)
    print("BUS ACCESS EDGE CONNECTIVITY ANALYSIS")
    print("=" * 60)
    
    total_stops = len(bus_stops)
    print(f"Total bus stops: {total_stops}")
    
    # Statistics
    print(f"\nCONNECTIVITY STATISTICS:")
    print(f"  • Stops with incoming bus_access edges: {len(analysis['stops_with_incoming'])}/{total_stops} ({len(analysis['stops_with_incoming'])/total_stops*100:.1f}%)")
    print(f"  • Stops with outgoing bus_access edges: {len(analysis['stops_with_outgoing'])}/{total_stops} ({len(analysis['stops_with_outgoing'])/total_stops*100:.1f}%)")
    print(f"  • Stops with BOTH incoming and outgoing: {len(analysis['stops_with_both'])}/{total_stops} ({len(analysis['stops_with_both'])/total_stops*100:.1f}%)")
    
    # Missing edges
    print(f"\nMISSING EDGES:")
    print(f"  • Stops missing incoming edges: {len(analysis['stops_missing_incoming'])}")
    print(f"  • Stops missing outgoing edges: {len(analysis['stops_missing_outgoing'])}")
    print(f"  • Stops missing BOTH: {len(analysis['stops_missing_both'])}")
    
    # Detailed edge counts
    print(f"\nDETAILED EDGE COUNTS:")
    for stop in bus_stops:
        incoming, outgoing = analysis['edge_counts'][stop]
        status = ""
        if incoming == 0 and outgoing == 0:
            status = " [NO EDGES]"
        elif incoming == 0:
            status = " [NO INCOMING]"
        elif outgoing == 0:
            status = " [NO OUTGOING]"
        else:
            status = " [OK]"
        
        print(f"  • {stop}: {incoming} incoming, {outgoing} outgoing{status}")
    
    # List problematic stops
    if analysis['stops_missing_both']:
        print(f"\nSTOPS WITH NO BUS_ACCESS EDGES:")
        for stop in sorted(analysis['stops_missing_both']):
            print(f"  • {stop}")
    
    if analysis['stops_missing_incoming']:
        print(f"\nSTOPS MISSING INCOMING EDGES:")
        for stop in sorted(analysis['stops_missing_incoming']):
            print(f"  • {stop}")
    
    if analysis['stops_missing_outgoing']:
        print(f"\nSTOPS MISSING OUTGOING EDGES:")
        for stop in sorted(analysis['stops_missing_outgoing']):
            print(f"  • {stop}")


def print_quality_report(quality: Dict):
    """Print a detailed quality report for bus_access edges."""
    print(f"\n" + "=" * 60)
    print("BUS ACCESS EDGE QUALITY ANALYSIS")
    print("=" * 60)
    
    # Self-loops
    if quality['self_loops']:
        print(f"\nSELF-LOOPS DETECTED:")
        for stop in quality['self_loops']:
            print(f"  • Bus stop {stop} has a self-loop edge")
    else:
        print(f"\nNo self-loops detected")
    
    # Missing reverse edges
    if quality['missing_reverse_edges']:
        print(f"\nMISSING REVERSE EDGES:")
        for u, v, key in quality['missing_reverse_edges']:
            print(f"  • Edge {u} → {v} (key={key}) missing reverse direction")
    else:
        print(f"\nAll bus_access edges have reverse directions")
    
    # Inconsistent oneway flags
    if quality['inconsistent_oneway']:
        print(f"\nINCONSISTENT ONEWAY FLAGS:")
        for u, v, key, reason in quality['inconsistent_oneway']:
            print(f"  • Edge {u} → {v} (key={key}): {reason}")
    else:
        print(f"\nAll bus_access edges have consistent oneway flags")
    
    # Edge attributes
    if quality['edge_attributes']:
        attrs = quality['edge_attributes']
        print(f"\nEDGE ATTRIBUTES:")
        print(f"  • Total bus_access edges: {attrs['total_bus_access_edges']}")
        print(f"  • Oneway edges: {attrs['oneway_count']}")
        print(f"  • Length statistics:")
        print(f"    - Min: {attrs['length_stats']['min']:.2f}")
        print(f"    - Max: {attrs['length_stats']['max']:.2f}")
        print(f"    - Avg: {attrs['length_stats']['avg']:.2f}")
        print(f"  • Sample attributes: {attrs['sample_attributes']}")


def suggest_fixes(analysis: Dict, quality: Dict, G: nx.MultiDiGraph, bus_stops: List):
    """Suggest fixes for missing bus_access edges and quality issues."""
    print(f"\nSUGGESTED FIXES:")
    
    # For stops with no edges at all
    if analysis['stops_missing_both']:
        print(f"\n1. STOPS WITH NO EDGES - Need to add bus_access edges:")
        for stop in analysis['stops_missing_both']:
            print(f"   • {stop}: Add bus_access edges to/from nearby nodes")
    
    # For stops missing incoming edges
    if analysis['stops_missing_incoming']:
        print(f"\n2. STOPS MISSING INCOMING EDGES:")
        for stop in analysis['stops_missing_incoming']:
            # Find potential sources
            potential_sources = []
            for node in G.nodes():
                if node != stop and node not in bus_stops:
                    # Check if there's a path to this stop
                    try:
                        path = nx.shortest_path(G, node, stop)
                        if len(path) <= 3:  # Reasonable distance
                            potential_sources.append(node)
                    except nx.NetworkXNoPath:
                        continue
            
            print(f"   • {stop}: Add incoming bus_access edge from one of {len(potential_sources)} potential sources")
    
    # For stops missing outgoing edges
    if analysis['stops_missing_outgoing']:
        print(f"\n3. STOPS MISSING OUTGOING EDGES:")
        for stop in analysis['stops_missing_outgoing']:
            # Find potential destinations
            potential_dests = []
            for node in G.nodes():
                if node != stop and node not in bus_stops:
                    # Check if there's a path from this stop
                    try:
                        path = nx.shortest_path(G, stop, node)
                        if len(path) <= 3:  # Reasonable distance
                            potential_dests.append(node)
                    except nx.NetworkXNoPath:
                        continue
            
            print(f"   • {stop}: Add outgoing bus_access edge to one of {len(potential_dests)} potential destinations")
    
    # Quality fixes
    if quality['self_loops']:
        print(f"\n4. REMOVE SELF-LOOPS:")
        for stop in quality['self_loops']:
            print(f"   • Remove self-loop edge from bus stop {stop}")
    
    if quality['missing_reverse_edges']:
        print(f"\n5. ADD MISSING REVERSE EDGES:")
        for u, v, key in quality['missing_reverse_edges']:
            print(f"   • Add reverse edge {v} → {u} (key={key})")
    
    if quality['inconsistent_oneway']:
        print(f"\n6. FIX ONEWAY FLAGS:")
        for u, v, key, reason in quality['inconsistent_oneway']:
            print(f"   • Set oneway=False for edge {u} → {v} (key={key})")


def main(graphml_path: str):
    """Main function to run the bus_access connectivity test."""
    print(f"Testing bus_access edge connectivity for: {graphml_path}")
    
    # Load graph and get bus stops
    G = load_graph(graphml_path)
    bus_stops = get_bus_stops(G)
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Found {len(bus_stops)} bus stops")
    
    # Analyze connectivity
    analysis = analyze_bus_access_connectivity(G, bus_stops)
    
    # Check quality
    quality = check_bus_access_edge_quality(G, bus_stops)
    
    # Print reports
    print_connectivity_report(analysis, bus_stops, G)
    print_quality_report(quality)
    
    # Suggest fixes
    suggest_fixes(analysis, quality, G, bus_stops)
    
    # Summary
    print(f"\nSUMMARY:")
    total_problems = len(analysis['stops_missing_incoming']) + len(analysis['stops_missing_outgoing']) - len(analysis['stops_missing_both'])
    quality_problems = len(quality['self_loops']) + len(quality['missing_reverse_edges']) + len(quality['inconsistent_oneway'])
    
    if total_problems == 0 and quality_problems == 0:
        print("All bus stops have proper bus_access edge connectivity and quality!")
    else:
        print(f"Found {total_problems} connectivity issues and {quality_problems} quality issues that need to be fixed.")
    
    return analysis, quality


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_bus_access_edges.py <graphml_file>")
        sys.exit(1)
    
    graphml_file = sys.argv[1]
    try:
        analysis, quality = main(graphml_file)
        total_issues = len(analysis['stops_missing_both']) + len(quality['self_loops']) + len(quality['missing_reverse_edges'])
        sys.exit(0 if total_issues == 0 else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 