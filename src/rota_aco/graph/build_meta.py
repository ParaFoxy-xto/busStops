# src/rota_aco/graph/build_meta.py

from typing import Any, Dict, List, Optional, Tuple, Set
import networkx as nx
import numpy as np
from rota_aco.data.opposites import find_opposites_by_proximity, find_opposites_by_access, combine_opposites, load_manual_opposites

def _get_path_length(graph: nx.MultiDiGraph, path: List[Any]) -> float:
    try:
        return nx.path_weight(graph, path, weight='length')
    except (nx.NetworkXNoPath, KeyError):
        return float('inf')

def _is_path_sensible(path: List[Any], graph: nx.MultiDiGraph, max_inefficiency_ratio: float = 5.0) -> bool:
    if len(path) < 2: return True
    start_node, end_node = path[0], path[-1]
    try:
        start_coords = (graph.nodes[start_node]['y'], graph.nodes[start_node]['x'])
        end_coords = (graph.nodes[end_node]['y'], graph.nodes[end_node]['x'])
    except KeyError: return True
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [start_coords[0], start_coords[1], end_coords[0], end_coords[1]])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    direct_distance_meters = R * c
    if direct_distance_meters < 20.0: return True
    path_length = _get_path_length(graph, path)
    if path_length == float('inf'): return False
    return (path_length / direct_distance_meters) <= max_inefficiency_ratio

def build_meta_graph(
    graph: nx.MultiDiGraph,
    bus_stops: List[Any],
    start_node: Any,
    exit_node: Any,
    precision: int = 5,
    manual_opposites_path: Optional[str] = None,
    verbose: bool = False
    
) -> Tuple[nx.DiGraph, Dict, List, Dict, Dict, Dict]:
    
    # 1. Setup Inicial
    opposites_prox = find_opposites_by_proximity(bus_stops, graph, precision)
    opposites_acc = find_opposites_by_access(graph, bus_stops)
    opposites_manual = load_manual_opposites(manual_opposites_path)
    if verbose and opposites_manual:
        print(f"[INFO] Carregados {len(opposites_manual)} nós com regras de opostos manuais.")
    all_opposites = combine_opposites(opposites_prox, opposites_acc, opposites_manual)
    
    if verbose: print("[INFO] Limpando nós de início/fim do dicionário de opostos.")
    nodes_to_clear = {start_node, exit_node}
    if start_node in all_opposites: del all_opposites[start_node]
    if exit_node in all_opposites: del all_opposites[exit_node]
    for stop_key in list(all_opposites.keys()):
        all_opposites[stop_key] = [opp for opp in all_opposites[stop_key] if opp not in nodes_to_clear]
        if not all_opposites[stop_key]: del all_opposites[stop_key]

    groups: Dict[tuple, List[Any]] = {}
    for stop in bus_stops:
        try:
            node_data = graph.nodes[stop]
            key = (round(float(node_data['y']), precision), round(float(node_data['x']), precision))
            groups.setdefault(key, []).append(stop)
        except KeyError: continue
        
    mapping: Dict[Any, Any] = {}
    initial_representatives: List[Any] = []
    groups_by_rep: Dict[Any, List[Any]] = {}
    for stops in groups.values():
        rep = stops[0]
        initial_representatives.append(rep)
        groups_by_rep[rep] = stops
        for s in stops: mapping[s] = rep
    if start_node and start_node not in initial_representatives: initial_representatives.append(start_node)
    if exit_node and exit_node not in initial_representatives: initial_representatives.append(exit_node)
    initial_representatives = sorted(list(set(initial_representatives)))

    # 2. Poda de Viabilidade
    print("[INFO] Verificando a viabilidade dos representantes...")
    try: all_paths = dict(nx.all_pairs_dijkstra_path(graph, weight='length'))
    except Exception as e:
        print(f"[ERRO] Falha ao calcular caminhos: {e}. O grafo pode estar desconectado.")
        return nx.DiGraph(), {}, [], {}, {}, {}
    
    viable_representatives = []
    for rep in initial_representatives:
        is_reachable = start_node in all_paths and rep in all_paths[start_node]
        can_reach_exit = rep in all_paths and exit_node in all_paths[rep]
        if rep in {start_node, exit_node} or (is_reachable and can_reach_exit):
            viable_representatives.append(rep)
    
    representatives = sorted(list(set(viable_representatives)))
    representatives_set = set(representatives)
    print(f"[INFO] {len(representatives)} representantes viáveis mantidos para o meta-grafo.")

    # 3. Construção das Arestas do Meta-Grafo com Filtros Corretos
    print("[INFO] Construindo arestas do meta-grafo...")
    meta_edges: Dict[Tuple[Any, Any], dict] = {}

    for u in representatives:
        if u not in all_paths: continue
        for v in representatives:
            if u == v or v not in all_paths[u]: continue
            
            path = all_paths[u][v]
            
            # --- FILTROS DE VALIDAÇÃO DA META-ARESTA ---

            # Filtro 1: Aresta direta entre opostos é estritamente proibida.
            if v in all_opposites.get(u, []):
                continue
            
            # Filtro 2 (Refinado): Evitar atalhos que pulam paradas, mas permitir "saltos"
            # para escapar de gargalos topológicos.
            opposites_of_u = set(all_opposites.get(u, []))
            has_intermediate_rep = False
            for intermediate_node in path[1:-1]:
                if intermediate_node in representatives_set:
                    # Permite pular um nó intermediário APENAS se ele for oposto ao ponto de partida.
                    if intermediate_node not in opposites_of_u:
                        has_intermediate_rep = True
                        break
            if has_intermediate_rep:
                continue
                
            # Filtro 3: O caminho subjacente não pode ser um "U-turn" ou retorno ineficiente.
            if not _is_path_sensible(path, graph):
                continue
            
            # Se a aresta passou por todos os filtros, ela é válida.
            path_length = _get_path_length(graph, path)
            if path_length != float('inf'):
                meta_edges[(u, v)] = {'length': path_length, 'time': path_length, 'path': path}

    # 4. Montagem Final do Grafo
    meta_graph = nx.DiGraph()
    meta_graph.add_nodes_from(representatives)
    for n in meta_graph.nodes:
        if n in graph.nodes: meta_graph.nodes[n].update(graph.nodes[n])
    
    for (u, v), data in meta_edges.items():
        meta_graph.add_edge(u, v, weight=data['time'])

    print(f"[INFO] Meta-grafo construído com {meta_graph.number_of_nodes()} nós e {meta_graph.number_of_edges()} arestas.")
    return meta_graph, meta_edges, representatives, all_opposites, mapping, groups_by_rep


def expand_meta_route(
    meta_route: List[Any],
    meta_graph: nx.DiGraph,
    meta_edges: Dict[Tuple[Any, Any], dict]
) -> List[Any]:
    if not meta_route: return []
    dense_route = [meta_route[0]]
    for u, v in zip(meta_route, meta_route[1:]):
        if meta_graph.has_edge(u, v): dense_route.append(v)
        else:
            try: dense_route.extend(nx.shortest_path(meta_graph, u, v, weight='weight')[1:])
            except (nx.NetworkXNoPath, nx.NodeNotFound): continue
    full_path: List[Any] = []
    if not dense_route: return []
    full_path.append(dense_route[0])
    for u, v in zip(dense_route, dense_route[1:]):
        edge_path = meta_edges.get((u, v), {}).get('path')
        if not edge_path: continue
        if full_path[-1] == edge_path[0]: full_path.extend(edge_path[1:])
        else: full_path.extend(edge_path)
    return full_path