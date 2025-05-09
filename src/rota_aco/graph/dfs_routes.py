# src/rota_aco/graph/dfs_routes.py

import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
from typing import Any, List, Dict
from rota_aco.data.opposites import stops_are_opposites


def prune_meta_graph(meta_G: nx.DiGraph, k: int) -> nx.DiGraph:
    pruned = nx.DiGraph()
    pruned.add_nodes_from(meta_G.nodes(data=True))
    for u in meta_G.nodes():
        nbrs = sorted(
            meta_G[u].items(),
            key=lambda item: item[1].get('weight', float('inf'))
        )
        for v, data in nbrs[:k]:
            pruned.add_edge(u, v, **data)
    return pruned


def find_k_shortest_paths(
    meta_G: nx.DiGraph,
    sources: List[Any],
    targets: List[Any],
    k: int = 3
) -> List[List[Any]]:
    """
    Gera até k caminhos simples de menor peso para cada par (u,v).
    Sai silenciosamente se não houver caminho.
    """
    paths: List[List[Any]] = []
    for u in sources:
        for v in targets:
            if u == v:
                continue
            try:
                gen = shortest_simple_paths(meta_G, u, v, weight='weight')
                for i, path in enumerate(gen):
                    if i >= k:
                        break
                    paths.append(path)
            except nx.NetworkXNoPath:
                # sem caminho entre u e v, ignora
                continue
    return paths


def filter_paths_remove_opposites(
    paths: List[List[Any]],
    opposites_proximity: Dict[Any, List[Any]],
    opposites_access: Dict[Any, List[Any]],
) -> List[List[Any]]:
    """
    Remove qualquer caminho que contenha nós opostos consecutivos.
    """
    filtered: List[List[Any]] = []
    for path in paths:
        valid = True
        for a, b in zip(path, path[1:]):
            if stops_are_opposites(a, b, opposites_proximity, opposites_access):
                valid = False
                break
        if valid:
            filtered.append(path)
    return filtered
