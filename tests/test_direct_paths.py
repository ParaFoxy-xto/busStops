#!/usr/bin/env python3
"""
Test script to check if routes unnecessarily pass through bus stops when direct street paths are available.
This is especially important for opposite stops that should be avoided.
"""
import sys
import networkx as nx
from typing import Dict, List, Tuple, Set, Any
from rota_aco.data.preprocess import load_graph, get_bus_stops
from rota_aco.data.opposites import find_opposites_by_access, is_bus_access_edge


def find_direct_paths_vs_bus_stop_paths(G: nx.MultiDiGraph, bus_stops: List[Any]) -> Dict:
    """
    Find cases where routes unnecessarily pass through bus stops when direct paths exist.
    
    Returns:
        Dictionary with analysis results including:
        - unnecessary_bus_stop_paths: list of (start, end, via_bus_stop, direct_path, bus_stop_path)
        - opposite_violations: list of paths that go through opposite stops unnecessarily
        - direct_path_opportunities: list of (start, end, direct_length, bus_stop_path_length)
    """
    unnecessary_bus_stop_paths = []
    opposite_violations = []
    direct_path_opportunities = []
    
    # Get opposites for checking violations
    opposites_access = find_opposites_by_access(G, bus_stops)
    
    # Check all pairs of bus stops
    for i, start_stop in enumerate(bus_stops):
        for end_stop in bus_stops[i+1:]:
            if start_stop == end_stop:
                continue
                
            try:
                # Find shortest path that might go through other bus stops
                full_path = nx.shortest_path(G, start_stop, end_stop, weight='length')
                full_length = nx.shortest_path_length(G, start_stop, end_stop, weight='length')
                
                # Check if this path goes through any other bus stops
                intermediate_bus_stops = [node for node in full_path[1:-1] if node in bus_stops]
                
                if intermediate_bus_stops:
                    # Try to find a direct path avoiding bus stops
                    # Create a subgraph without the intermediate bus stops
                    nodes_to_remove = set(intermediate_bus_stops)
                    subgraph = G.copy()
                    subgraph.remove_nodes_from(nodes_to_remove)
                    
                    try:
                        direct_path = nx.shortest_path(subgraph, start_stop, end_stop, weight='length')
                        direct_length = nx.shortest_path_length(subgraph, start_stop, end_stop, weight='length')
                        
                        # Check if direct path is significantly shorter or avoids opposites
                        is_opposite_violation = False
                        for intermediate_stop in intermediate_bus_stops:
                            if (start_stop in opposites_access.get(intermediate_stop, []) or 
                                intermediate_stop in opposites_access.get(start_stop, [])):
                                is_opposite_violation = True
                                break
                        
                        if direct_length < full_length * 1.2:  # Direct path is not much longer
                            unnecessary_bus_stop_paths.append({
                                'start': start_stop,
                                'end': end_stop,
                                'via_bus_stops': intermediate_bus_stops,
                                'direct_path': direct_path,
                                'bus_stop_path': full_path,
                                'direct_length': direct_length,
                                'bus_stop_path_length': full_length,
                                'is_opposite_violation': is_opposite_violation
                            })
                            
                            if is_opposite_violation:
                                opposite_violations.append({
                                    'start': start_stop,
                                    'end': end_stop,
                                    'via_opposite': intermediate_bus_stops,
                                    'direct_path': direct_path,
                                    'bus_stop_path': full_path
                                })
                            
                            direct_path_opportunities.append({
                                'start': start_stop,
                                'end': end_stop,
                                'direct_length': direct_length,
                                'bus_stop_path_length': full_length,
                                'savings': full_length - direct_length
                            })
                            
                    except nx.NetworkXNoPath:
                        # No direct path available, this is acceptable
                        pass
                        
            except nx.NetworkXNoPath:
                # No path exists between these stops
                pass
    
    return {
        'unnecessary_bus_stop_paths': unnecessary_bus_stop_paths,
        'opposite_violations': opposite_violations,
        'direct_path_opportunities': direct_path_opportunities
    }


def print_direct_path_analysis(analysis: Dict, bus_stops: List, G: nx.MultiDiGraph):
    """Print a detailed analysis of direct path opportunities."""
    print("=" * 80)
    print("DIRECT PATH vs BUS STOP PATH ANALYSIS")
    print("=" * 80)
    
    total_bus_stops = len(bus_stops)
    print(f"Total bus stops: {total_bus_stops}")
    
    # Summary statistics
    unnecessary_count = len(analysis['unnecessary_bus_stop_paths'])
    opposite_violations_count = len(analysis['opposite_violations'])
    opportunities_count = len(analysis['direct_path_opportunities'])
    
    print(f"\nANALYSIS SUMMARY:")
    print(f"  • Unnecessary bus stop paths found: {unnecessary_count}")
    print(f"  • Opposite violations found: {opposite_violations_count}")
    print(f"  • Direct path opportunities: {opportunities_count}")
    
    # Opposite violations (most critical)
    if analysis['opposite_violations']:
        print(f"\n🚨 OPPOSITE VIOLATIONS (CRITICAL):")
        print(f"  Routes that unnecessarily pass through opposite stops:")
        for violation in analysis['opposite_violations']:
            print(f"    • {violation['start']} → {violation['end']}")
            print(f"      Via opposite: {violation['via_opposite']}")
            print(f"      Direct path: {violation['direct_path']}")
            print(f"      Bus stop path: {violation['bus_stop_path']}")
            print()
    
    # Direct path opportunities
    if analysis['direct_path_opportunities']:
        print(f"\n💡 DIRECT PATH OPPORTUNITIES:")
        print(f"  Routes that could be optimized:")
        
        # Sort by savings (highest first)
        sorted_opportunities = sorted(analysis['direct_path_opportunities'], 
                                    key=lambda x: x['savings'], reverse=True)
        
        for i, opp in enumerate(sorted_opportunities[:10]):  # Show top 10
            savings_pct = (opp['savings'] / opp['bus_stop_path_length']) * 100
            print(f"    {i+1}. {opp['start']} → {opp['end']}")
            print(f"       Direct: {opp['direct_length']:.1f}, Bus stop path: {opp['bus_stop_path_length']:.1f}")
            print(f"       Savings: {opp['savings']:.1f} ({savings_pct:.1f}%)")
    
    # Detailed unnecessary paths
    if analysis['unnecessary_bus_stop_paths']:
        print(f"\n📋 DETAILED UNNECESSARY PATHS:")
        for path_info in analysis['unnecessary_bus_stop_paths']:
            status = "🚨 OPPOSITE VIOLATION" if path_info['is_opposite_violation'] else "⚠️  SUBOPTIMAL"
            print(f"    {status}: {path_info['start']} → {path_info['end']}")
            print(f"      Via bus stops: {path_info['via_bus_stops']}")
            print(f"      Direct path: {path_info['direct_path']}")
            print(f"      Bus stop path: {path_info['bus_stop_path']}")
            print(f"      Direct length: {path_info['direct_length']:.1f}")
            print(f"      Bus stop path length: {path_info['bus_stop_path_length']:.1f}")


def suggest_fixes(analysis: Dict):
    """Suggest fixes for the identified issues."""
    print(f"\n🔧 SUGGESTED FIXES:")
    
    if analysis['opposite_violations']:
        print(f"\n1. CRITICAL: Fix opposite violations in route generation:")
        print(f"   • Modify route generation to prefer direct paths when they avoid opposite stops")
        print(f"   • Add penalty for routes that unnecessarily pass through opposite stops")
        print(f"   • Implement opposite-aware pathfinding in meta-graph construction")
    
    if analysis['direct_path_opportunities']:
        print(f"\n2. OPTIMIZATION: Improve route efficiency:")
        print(f"   • Modify shortest path algorithm to prefer direct paths when they're not much longer")
        print(f"   • Add weight penalty for unnecessary bus stop traversals")
        print(f"   • Consider direct paths in meta-graph edge calculation")
    
    if analysis['unnecessary_bus_stop_paths']:
        print(f"\n3. META-GRAPH: Update meta-graph construction:")
        print(f"   • When calculating meta-edges, check if direct paths exist")
        print(f"   • Prefer direct paths over bus-stop-intermediate paths when appropriate")
        print(f"   • Add configuration to control when to use direct vs bus-stop paths")


def main(graphml_path: str):
    """Main function to run the direct path analysis."""
    print(f"Analyzing direct paths vs bus stop paths for: {graphml_path}")
    
    # Load graph and get bus stops
    G = load_graph(graphml_path)
    bus_stops = get_bus_stops(G)
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Found {len(bus_stops)} bus stops")
    
    # Analyze direct paths
    analysis = find_direct_paths_vs_bus_stop_paths(G, bus_stops)
    
    # Print analysis
    print_direct_path_analysis(analysis, bus_stops, G)
    
    # Suggest fixes
    suggest_fixes(analysis)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    critical_issues = len(analysis['opposite_violations'])
    optimization_opportunities = len(analysis['direct_path_opportunities'])
    
    if critical_issues == 0 and optimization_opportunities == 0:
        print("✅ No direct path issues found!")
    else:
        print(f"❌ Found {critical_issues} critical opposite violations and {optimization_opportunities} optimization opportunities.")
    
    return analysis


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_direct_paths.py <graphml_file>")
        sys.exit(1)
    
    graphml_file = sys.argv[1]
    try:
        analysis = main(graphml_file)
        critical_issues = len(analysis['opposite_violations'])
        sys.exit(0 if critical_issues == 0 else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 