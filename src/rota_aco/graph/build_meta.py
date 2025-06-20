# src/rota_aco/graph/build_meta.py

import networkx as nx
import numpy as np
from typing import Any, List, Tuple, Dict, Set, Optional
from rota_aco.data.preprocess import pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access, stops_are_opposites


def validate_bus_stop_connectivity(G: nx.MultiDiGraph, bus_stops: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Validate bus stop connectivity to detect problematic stops.
    
    A bus stop is considered problematic if:
    1. It only has one bus_access edge (dead-end)
    2. Both bus_access edges lead to the same node (spur/loop)
    3. The only way to continue from the stop is to go back the same way
    
    Returns:
        Tuple of (valid_stops, problematic_stops)
    """
    valid_stops = []
    problematic_stops = []
    
    for stop in bus_stops:
        # Get all bus_access edges for this stop
        bus_access_edges = []
        
        # Check incoming edges
        for pred, _, data in G.in_edges(stop, data=True):
            if data.get('highway', '').strip().lower() == 'bus_access':
                bus_access_edges.append(pred)
        
        # Check outgoing edges
        for _, succ, data in G.out_edges(stop, data=True):
            if data.get('highway', '').strip().lower() == 'bus_access':
                bus_access_edges.append(succ)
        
        # Remove duplicates and the stop itself
        connected_nodes = list(set(bus_access_edges))
        if stop in connected_nodes:
            connected_nodes.remove(stop)
        
        # Check if this is a problematic stop
        is_problematic = False
        
        # Case 1: Only one connection (dead-end)
        if len(connected_nodes) <= 1:
            is_problematic = True
            print(f"[WARNING] Bus stop {stop} is a dead-end (only {len(connected_nodes)} connection(s))")
        
        # Case 2: Both edges lead to the same node (spur/loop)
        elif len(connected_nodes) == 2 and connected_nodes[0] == connected_nodes[1]:
            is_problematic = True
            print(f"[WARNING] Bus stop {stop} is a spur (both edges lead to same node: {connected_nodes[0]})")
        
        # Case 3: Check if the stop is only accessible as a spur (in and out same way)
        else:
            # For each connected node, check if there's a path to other nodes
            # that doesn't go back through this stop
            has_through_traffic = False
            
            for node in connected_nodes:
                # Check if this node can reach other parts of the network
                # without going back through the bus stop
                try:
                    # Get all nodes reachable from this connected node
                    reachable = set(nx.descendants(G, node))
                    # Remove the bus stop itself
                    reachable.discard(stop)
                    
                    # If we can reach other nodes, this is valid through-traffic
                    if len(reachable) > 0:
                        has_through_traffic = True
                        break
                        
                except nx.NetworkXError:
                    # If there's an error, assume it's problematic
                    pass
            
            if not has_through_traffic:
                is_problematic = True
                print(f"[WARNING] Bus stop {stop} has no through-traffic (only accessible as spur)")
        
        if is_problematic:
            problematic_stops.append(stop)
        else:
            valid_stops.append(stop)
    
    return valid_stops, problematic_stops


def prune_dead_ends(meta_G: nx.DiGraph, exit_node: Any = None) -> nx.DiGraph:
    """
    Remove dead-end nodes (degree 1) from the meta-graph, except the exit node.
    Iteratively prunes until no more dead-ends remain (except exit).
    """
    G = meta_G.copy()
    while True:
        to_remove = []
        for node in G.nodes():
            if node == exit_node:
                continue
            # Dead-end: only one neighbor (either in or out, but not both)
            if (G.in_degree[node] + G.out_degree[node]) <= 1:
                to_remove.append(node)
        if not to_remove:
            break
        G.remove_nodes_from(to_remove)
    return G


def detect_bounceback_path(path: List[Any], G: nx.MultiDiGraph, penalty_factor: float = 2.0) -> float:
    """
    Detect bounceback patterns in a path and return a penalty factor.
    
    Bounceback patterns include:
    1. Paths that go back and forth between the same areas
    2. Paths that access a bus stop and immediately return
    3. Paths with too many direction changes
    
    Args:
        path: List of nodes in the path
        G: Original graph
        penalty_factor: Multiplier to apply if bounceback is detected
    
    Returns:
        Penalty factor (1.0 = no penalty, >1.0 = penalty applied)
    """
    if len(path) < 4:
        return 1.0  # Too short to have meaningful bounceback
    
    # Calculate the overall direction of the path
    start_coords = (G.nodes[path[0]]['x'], G.nodes[path[0]]['y'])
    end_coords = (G.nodes[path[-1]]['x'], G.nodes[path[-1]]['y'])
    overall_direction = (end_coords[0] - start_coords[0], end_coords[1] - start_coords[1])
    overall_distance = np.sqrt(overall_direction[0]**2 + overall_direction[1]**2)
    
    if overall_distance < 1e-6:
        return penalty_factor  # Path starts and ends at same place
    
    # Check for local direction reversals
    direction_changes = 0
    total_path_length = 0
    
    for i in range(1, len(path) - 1):
        prev_coords = (G.nodes[path[i-1]]['x'], G.nodes[path[i-1]]['y'])
        curr_coords = (G.nodes[path[i]]['x'], G.nodes[path[i]]['y'])
        next_coords = (G.nodes[path[i+1]]['x'], G.nodes[path[i+1]]['y'])
        
        # Calculate local directions
        dir1 = (curr_coords[0] - prev_coords[0], curr_coords[1] - prev_coords[1])
        dir2 = (next_coords[0] - curr_coords[0], next_coords[1] - curr_coords[1])
        
        # Calculate dot product to detect direction changes
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        magnitudes = np.sqrt(dir1[0]**2 + dir1[1]**2) * np.sqrt(dir2[0]**2 + dir2[1]**2)
        
        if magnitudes > 1e-6:
            cos_angle = dot_product / magnitudes
            if cos_angle < -0.5:  # Angle > 120 degrees (significant direction change)
                direction_changes += 1
        
        # Add segment length
        segment_length = np.sqrt(dir2[0]**2 + dir2[1]**2)
        total_path_length += segment_length
    
    # Check for excessive direction changes relative to path length
    if total_path_length > 0:
        direction_change_ratio = direction_changes / (len(path) - 2)
        if direction_change_ratio > 0.3:  # More than 30% of segments have direction changes
            return penalty_factor
    
    # Check for paths that are much longer than the direct distance
    if overall_distance > 0:
        efficiency_ratio = total_path_length / overall_distance
        if efficiency_ratio > 3.0:  # Path is more than 3x longer than direct distance
            return penalty_factor
    
    return 1.0


def find_direct_path_avoiding_intermediates(
    G: nx.MultiDiGraph,
    start: Any,
    end: Any,
    problematic_stops: Set[Any],
    opposites_proximity: Dict[Any, List[Any]],
    opposites_access: Dict[Any, List[Any]],
    max_path_length_factor: float = 1.5,
    bounceback_penalty: float = 2.0
) -> Optional[Tuple[List[Any], float]]:
    """
    Find a direct path from start to end that avoids problematic intermediate bus stops.
    
    Args:
        G: Original graph
        start: Starting bus stop
        end: Ending bus stop
        problematic_stops: Set of bus stops to avoid as intermediates
        opposites_proximity: Dictionary of opposites by proximity
        opposites_access: Dictionary of opposites by access
        max_path_length_factor: Maximum allowed path length as factor of direct distance
        bounceback_penalty: Penalty factor for bounceback paths
    
    Returns:
        Tuple of (path, length) or None if no good path found
    """
    try:
        # First, try to find the shortest path
        shortest_path = nx.shortest_path(G, start, end, weight='length')
        shortest_length = nx.path_weight(G, shortest_path, weight='length')
        
        # Apply bounceback penalty to shortest path
        bounceback_factor = detect_bounceback_path(shortest_path, G, bounceback_penalty)
        adjusted_shortest_length = shortest_length * bounceback_factor
        
        # Check if the shortest path is already good (no problematic intermediates)
        problematic_intermediates = []
        for node in shortest_path[1:-1]:  # Exclude start and end
            if (G.nodes[node].get('bus_stop', '').strip().lower() == 'true' and
                node in problematic_stops):
                problematic_intermediates.append(node)
        
        if not problematic_intermediates and bounceback_factor == 1.0:
            # Shortest path is already good and has no bounceback
            return shortest_path, shortest_length
        
        # Calculate direct distance for reference
        start_coords = (G.nodes[start]['x'], G.nodes[start]['y'])
        end_coords = (G.nodes[end]['x'], G.nodes[end]['y'])
        direct_distance = np.sqrt((start_coords[0] - end_coords[0])**2 + 
                                (end_coords[1] - start_coords[1])**2)
        
        # Try to find alternative paths that avoid problematic stops
        best_path = None
        best_length = float('inf')
        best_adjusted_length = float('inf')
        
        # Create a temporary graph without problematic stops
        temp_G = G.copy()
        nodes_to_remove = [n for n in problematic_stops if n != start and n != end]
        temp_G.remove_nodes_from(nodes_to_remove)
        
        try:
            # Try to find path in the cleaned graph
            alt_path = nx.shortest_path(temp_G, start, end, weight='length')
            alt_length = nx.path_weight(temp_G, alt_path, weight='length')
            
            # Apply bounceback penalty
            alt_bounceback_factor = detect_bounceback_path(alt_path, G, bounceback_penalty)
            alt_adjusted_length = alt_length * alt_bounceback_factor
            
            # Check if this path is reasonable (not too much longer than direct distance)
            if alt_adjusted_length <= direct_distance * max_path_length_factor:
                best_path = alt_path
                best_length = alt_length
                best_adjusted_length = alt_adjusted_length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        # If we found a good alternative path, use it
        if best_path is not None:
            return best_path, best_length
        
        # If no good alternative found, check if the original path is acceptable
        # (maybe the problematic intermediates are not too bad)
        if adjusted_shortest_length <= direct_distance * max_path_length_factor:
            return shortest_path, shortest_length
        
        # No acceptable path found
        return None
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def build_meta_graph(
    G: nx.MultiDiGraph,
    precision: int = 6,
    extra_nodes: List[Any] = None,
    exit_node: Any = None,
    start_node: Any = None
) -> Tuple[
    nx.DiGraph,
    Dict[Tuple[Any, Any], dict],
    List[Any],
    dict,
    dict,
    dict,
    dict
]:
    """
    Constrói o meta-grafo a partir do grafo original G:
      - identifica opostos por proximidade e acesso
      - agrupa paradas por coordenadas arredondadas em representantes
      - adiciona nós extras (ex.: start, exit)
      - cria arestas entre representantes, permitindo passagens pela rua sem parar em paradas opostas

    Parâmetros especiais:
      - start_node e exit_node: se forem dead-ends, são mantidos no meta-grafo, mas só podem ser acessados uma vez (início/fim de rota)

    Retorna:
      - meta_G: grafo dirigido simplificado
      - meta_edges: mapa (u,v) -> {'length', 'path'}
      - unique_bus_stops: lista de representantes
      - opposites_proximity: dicionário de opostos por proximidade
      - opposites_access: dicionário de opostos por acesso
      - mapping: parada original -> representante
      - groups_by_rep: representante -> lista de paradas
    """
    if extra_nodes is None:
        extra_nodes = []

    # 1. Identifica paradas de ônibus e inclui extra_nodes
    bus_stops = [n for n, data in G.nodes(data=True)
                 if data.get('bus_stop', '').strip().lower() == 'true']
    for node in extra_nodes:
        if node not in bus_stops:
            bus_stops.append(node)

    # 1.5. Validate bus stop connectivity (NEW)
    print(f"Validating connectivity for {len(bus_stops)} bus stops...")
    valid_bus_stops, problematic_bus_stops = validate_bus_stop_connectivity(G, bus_stops)
    
    # Always keep start_node and exit_node, even if problematic
    special_stops = set()
    if start_node is not None:
        special_stops.add(start_node)
    if exit_node is not None:
        special_stops.add(exit_node)
    
    final_bus_stops = list(valid_bus_stops) + [stop for stop in problematic_bus_stops if stop in special_stops]
    bus_stops = list(set(final_bus_stops))
    
    if problematic_bus_stops:
        print(f"[WARNING] Found {len(problematic_bus_stops)} problematic bus stops:")
        for stop in problematic_bus_stops:
            if stop in special_stops:
                print(f"  • {stop} (kept as special start/exit)")
            else:
                print(f"  • {stop}")
        print(f"[INFO] Using {len(bus_stops)} bus stops for meta-graph construction (including special start/exit if needed)")
    else:
        print(f"[INFO] All {len(bus_stops)} bus stops have valid connectivity")

    # 2. Detecta opostos
    opposites_proximity = pre_process_opposites(bus_stops, G, precision)
    opposites_access = find_opposites_by_access(G, bus_stops)

    # Print all opposites (proximity and access)
    print("[VERBOSE] List of opposite bus stops (proximity):")
    for k, vlist in opposites_proximity.items():
        for v in vlist:
            print(f"  {k} <-> {v}")
    print("[VERBOSE] List of opposite bus stops (access):")
    for k, vlist in opposites_access.items():
        for v in vlist:
            print(f"  {k} <-> {v}")
    # Always highlight the specific pair if present
    special_pair = (12529762065, 12529762088)
    found = False
    for v in opposites_proximity.get(special_pair[0], []):
        if v == special_pair[1]:
            print(f"[VERBOSE] Special opposites: {special_pair[0]} <-> {special_pair[1]} (proximity)")
            found = True
    for v in opposites_access.get(special_pair[0], []):
        if v == special_pair[1]:
            print(f"[VERBOSE] Special opposites: {special_pair[0]} <-> {special_pair[1]} (access)")
            found = True
    if not found:
        print(f"[VERBOSE] Special opposites: {special_pair[0]} <-> {special_pair[1]} NOT FOUND in opposites")

    # 3. Agrupa paradas por coordenadas arredondadas
    groups: Dict[Tuple[float, float], List[Any]] = {}
    for stop in bus_stops:
        x = round(float(G.nodes[stop]['x']), precision)
        y = round(float(G.nodes[stop]['y']), precision)
        groups.setdefault((x, y), []).append(stop)

    # 4. Define representantes e mapeamento
    mapping: Dict[Any, Any] = {}
    unique_bus_stops: List[Any] = []
    groups_by_rep: Dict[Any, List[Any]] = {}
    for stops in groups.values():
        rep = stops[0]
        unique_bus_stops.append(rep)
        groups_by_rep[rep] = stops
        for s in stops:
            mapping[s] = rep
    for extra in extra_nodes:
        if extra not in mapping:
            unique_bus_stops.append(extra)
            groups_by_rep[extra] = [extra]
            mapping[extra] = extra

    # 5. Calcula arestas do meta-grafo usando paths que evitam intermediários problemáticos
    meta_edges: Dict[Tuple[Any, Any], dict] = {}
    
    # Create set of problematic stops for easier lookup
    problematic_stops_set = set(problematic_bus_stops)
    
    print(f"[INFO] Building meta-graph edges with {len(unique_bus_stops)} bus stops...")
    edges_created = 0
    edges_avoided_intermediates = 0
    
    for u in unique_bus_stops:
        for v in unique_bus_stops:
            if u == v:
                continue
            # ignora conexões diretas entre representantes opostos
            if stops_are_opposites(u, v, opposites_proximity, groups_by_rep):
                continue
            if v in opposites_access.get(u, []) or u in opposites_access.get(v, []):
                continue
            
            # Try to find direct path avoiding problematic intermediates
            result_uv = find_direct_path_avoiding_intermediates(
                G, u, v, problematic_stops_set, opposites_proximity, opposites_access
            )
            
            if result_uv is not None:
                path_uv, length_uv = result_uv
                meta_edges[(u, v)] = {'length': length_uv, 'time': length_uv, 'path': path_uv}
                edges_created += 1
                if len(path_uv) > 2:
                    problematic_in_path = [node for node in path_uv[1:-1] 
                                         if G.nodes[node].get('bus_stop', '').strip().lower() == 'true' 
                                         and node in problematic_stops_set]
                    if not problematic_in_path:
                        edges_avoided_intermediates += 1
                else:
                    edges_avoided_intermediates += 1
            
            # Try to add edge from v to u (bidirectional where possible)
            result_vu = find_direct_path_avoiding_intermediates(
                G, v, u, problematic_stops_set, opposites_proximity, opposites_access
            )
            
            if result_vu is not None:
                path_vu, length_vu = result_vu
                meta_edges[(v, u)] = {'length': length_vu, 'time': length_vu, 'path': path_vu}
                edges_created += 1
                if len(path_vu) > 2:
                    problematic_in_path = [node for node in path_vu[1:-1] 
                                         if G.nodes[node].get('bus_stop', '').strip().lower() == 'true' 
                                         and node in problematic_stops_set]
                    if not problematic_in_path:
                        edges_avoided_intermediates += 1
                else:
                    edges_avoided_intermediates += 1
    
    print(f"[INFO] Created {edges_created} meta-graph edges")
    print(f"[INFO] {edges_avoided_intermediates} edges avoid problematic intermediate bus stops")

    # 6. Monta o meta-grafo simplificado
    meta_G = nx.DiGraph()
    meta_G.add_nodes_from(unique_bus_stops)
    for (u, v), info in meta_edges.items():
        meta_G.add_edge(u, v, weight=info['length'])

    # Filter out bounceback edges
    meta_G, meta_edges = filter_bounceback_edges(meta_G, meta_edges, G)

    # Prune dead-ends except exit_node (if provided)
    if exit_node is not None and exit_node in meta_G:
        meta_G = prune_dead_ends(meta_G, exit_node=exit_node)
    else:
        meta_G = prune_dead_ends(meta_G)

    return (
        meta_G,
        meta_edges,
        list(meta_G.nodes()),  # update unique_bus_stops to pruned nodes
        opposites_proximity,
        opposites_access,
        mapping,
        groups_by_rep
    )


def resolver_TSP(meta_G: nx.DiGraph, start_node: Any) -> List[Any]:
    """Heurística do vizinho mais próximo no meta-grafo"""
    nodes = list(meta_G.nodes())
    route = [start_node]
    remaining = set(nodes) - {start_node}
    current = start_node
    while remaining:
        nearest, min_d = None, float('inf')
        for cand in remaining:
            try:
                d = nx.shortest_path_length(meta_G, current, cand, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if d < min_d:
                nearest, min_d = cand, d
        if nearest is None:
            break
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return route


def expand_meta_route(
    route: List[Any],
    meta_G: nx.DiGraph,
    meta_edges: Dict[Tuple[Any, Any], dict]
) -> List[Any]:
    """
    Densifica rota de representantes pelo meta_grafo e expande para rota completa:
      1) Substitui saltos inexistentes por subcaminhos de meta_G
      2) Usa meta_edges para traduzir cada salto em path original
    """
    # densificação via meta_G
    dense = [route[0]]
    for u, v in zip(route, route[1:]):
        if meta_G.has_edge(u, v):
            dense.append(v)
        else:
            try:
                sub = nx.shortest_path(meta_G, u, v, weight='weight')
                dense.extend(sub[1:])
            except nx.NetworkXNoPath:
                # Skip this segment if no path exists
                print(f"[WARNING] No path in meta_G between {u} and {v}. Skipping segment.")
                continue  # Skip this segment and continue with the rest
    
    # expansão para grafo original
    final: List[Any] = []
    for u, v in zip(dense, dense[1:]):
        if (u, v) in meta_edges:
            seg = meta_edges[(u, v)]['path']
        elif (v, u) in meta_edges:
            seg = list(reversed(meta_edges[(v, u)]['path']))
        else:
            print(f"[WARNING] No meta_edge between {u} and {v}. Skipping segment.")
            continue
        if final and final[-1] == seg[0]:
            final.extend(seg[1:])
        else:
            final.extend(seg)

    # remove duplicações consecutivas
    return [final[i] for i in range(len(final)) if i == 0 or final[i] != final[i-1]]


def filter_opposite_meta_route(
    route: List[Any],
    opposites_proximity: Dict[Any, List[Any]],
    opposites_access: Dict[Any, List[Any]]
) -> List[Any]:
    """
    Remove saltos entre representantes opostos de uma meta-rota.
    """
    if not route:
        return []
    filtered = [route[0]]
    for nxt in route[1:]:
        prev = filtered[-1]
        # se salto proibido, ignora nxt
        if nxt in opposites_proximity.get(prev, []) or nxt in opposites_access.get(prev, []):
            continue
        filtered.append(nxt)
    return filtered


def prune_meta_graph_edges(meta_G, meta_edges, factor=2.0):
    """
    Remove edges from meta_G and meta_edges if their length is much greater than the median outgoing edge length for their source node.
    """
    to_remove = []
    for u in meta_G.nodes:
        lengths = [meta_edges[(u, v)]['length'] for v in meta_G.successors(u) if (u, v) in meta_edges]
        if not lengths:
            continue
        median = np.median(lengths)
        for v in list(meta_G.successors(u)):
            if (u, v) in meta_edges and meta_edges[(u, v)]['length'] > factor * median:
                to_remove.append((u, v))
    for u, v in to_remove:
        if meta_G.has_edge(u, v):
            meta_G.remove_edge(u, v)
        if (u, v) in meta_edges:
            del meta_edges[(u, v)]
    return meta_G, meta_edges


def filter_bounceback_edges(meta_G: nx.DiGraph, meta_edges: Dict[Tuple[Any, Any], dict], G: nx.MultiDiGraph) -> Tuple[nx.DiGraph, Dict[Tuple[Any, Any], dict]]:
    """
    Filter out edges from meta-graph that would create bounceback patterns.
    
    Args:
        meta_G: Meta-graph
        meta_edges: Dictionary of meta-edges with their paths
        G: Original graph
    
    Returns:
        Filtered meta-graph and meta-edges
    """
    edges_to_remove = []
    
    for (u, v), edge_info in meta_edges.items():
        path = edge_info['path']
        
        # Check if this edge creates a bounceback pattern
        bounceback_factor = detect_bounceback_path(path, G, penalty_factor=2.0)
        
        # If bounceback is detected, mark for removal
        if bounceback_factor > 1.0:
            edges_to_remove.append((u, v))
            print(f"[DEBUG] Removing bounceback edge {u} -> {v} (bounceback factor: {bounceback_factor:.2f})")
    
    # Remove the bounceback edges
    for (u, v) in edges_to_remove:
        if meta_G.has_edge(u, v):
            meta_G.remove_edge(u, v)
        if (u, v) in meta_edges:
            del meta_edges[(u, v)]
    
    print(f"[INFO] Removed {len(edges_to_remove)} bounceback edges from meta-graph")
    return meta_G, meta_edges
