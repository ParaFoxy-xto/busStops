# src/rota_aco/graph/build_meta.py

import networkx as nx
from typing import Any, List, Tuple, Dict
from rota_aco.data.preprocess import pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access, stops_are_opposites


def build_meta_graph(
    G: nx.MultiDiGraph,
    precision: int = 6,
    extra_nodes: List[Any] = None
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

    Retorna:
      - meta_G: grafo dirigido simplificado
      - meta_edges: mapa (u,v) -> {'length', 'path'}
      - unique_bus_stops: lista de representantes
      - opposites_proximity: dicionário de opostos por proximidade
      - opposites_access: dicionário de opostos por acesso
      - mapping: parada original -> representante
      - groups_by_rep: representante -> lista de paradas
    """
    extra_nodes = extra_nodes or []

    # 1. Identifica paradas de ônibus e inclui extra_nodes
    bus_stops = [n for n, data in G.nodes(data=True)
                 if data.get('bus_stop', '').strip().lower() == 'true']
    for node in extra_nodes:
        if node not in bus_stops:
            bus_stops.append(node)

    # 2. Detecta opostos
    opposites_proximity = pre_process_opposites(bus_stops, G, precision)
    opposites_access = find_opposites_by_access(G, bus_stops)

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

    # 5. Calcula arestas do meta-grafo (sem filtrar intermediários)
    meta_edges: Dict[Tuple[Any, Any], dict] = {}
    for u in unique_bus_stops:
        for v in unique_bus_stops:
            if u == v:
                continue
            # ignora conexões diretas entre representantes opostos
            if stops_are_opposites(u, v, opposites_proximity, groups_by_rep):
                continue
            if v in opposites_access.get(u, []) or u in opposites_access.get(v, []):
                continue
            try:
                path_uv = nx.shortest_path(G, u, v, weight='length')
                length_uv = nx.path_weight(G, path_uv, weight='length')
                meta_edges[(u, v)] = {'length': length_uv, 'path': path_uv}
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # 6. Monta o meta-grafo simplificado
    meta_G = nx.DiGraph()
    meta_G.add_nodes_from(unique_bus_stops)
    for (u, v), info in meta_edges.items():
        meta_G.add_edge(u, v, weight=info['length'])

    return (
        meta_G,
        meta_edges,
        unique_bus_stops,
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
            sub = nx.shortest_path(meta_G, u, v, weight='weight')
            dense.extend(sub[1:])

    # expansão para grafo original
    final: List[Any] = []
    for u, v in zip(dense, dense[1:]):
        if (u, v) in meta_edges:
            seg = meta_edges[(u, v)]['path']
        else:
            seg = list(reversed(meta_edges[(v, u)]['path']))
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
