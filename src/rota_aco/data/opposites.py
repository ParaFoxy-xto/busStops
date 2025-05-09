import networkx as nx
from typing import Any, Dict, List


def is_bus_access_edge(data: dict) -> bool:
    """
    Detecta se a aresta contém 'bus_access' em qualquer atributo,
    seja no campo 'highway' ou em qualquer outro valor do dicionário.
    """
    if str(data.get("highway", "")).strip().lower() == "bus_access":
        return True
    return any(
        str(v).strip().lower() == "bus_access"
        for v in data.values()
    )


def find_opposites_by_access(
    G: nx.Graph,
    bus_stops: List[Any]
) -> Dict[Any, List[Any]]:
    """
    Identifica pares de paradas opostas via duas arestas 'bus_access'.
    Para cada parada A, verifica todas as edges A->X ou X->A com is_bus_access_edge,
    e se X não for parada, explora X->Y (ou Y->X) para capturar Y em bus_stops.
    """
    opposites: Dict[Any, List[Any]] = {stop: [] for stop in bus_stops}
    for A in bus_stops:
        # arestas de saída
        for _, X, data in G.out_edges(A, data=True):
            if is_bus_access_edge(data):
                if X in bus_stops and X != A:
                    opposites[A].append(X)
                else:
                    for _, Y, data2 in G.out_edges(X, data=True):
                        if (
                            is_bus_access_edge(data2)
                            and Y in bus_stops
                            and Y != A
                        ):
                            opposites[A].append(Y)
        # arestas de entrada
        for X, _, data in G.in_edges(A, data=True):
            if is_bus_access_edge(data):
                if X in bus_stops and X != A:
                    opposites[A].append(X)
                else:
                    for Y, _, data2 in G.in_edges(X, data=True):
                        if (
                            is_bus_access_edge(data2)
                            and Y in bus_stops
                            and Y != A
                        ):
                            opposites[A].append(Y)
        # remove duplicatas
        opposites[A] = list(set(opposites[A]))
    return opposites


def stops_are_opposites(
    rep_u: Any,
    rep_v: Any,
    opposites_dict: Dict[Any, List[Any]],
    groups_by_rep: Dict[Any, List[Any]]
) -> bool:
    """
    Retorna True se qualquer parada no grupo de rep_u
    tiver como oposta (em opposites_dict) alguma parada no grupo de rep_v.
    """
    for s in groups_by_rep.get(rep_u, []):
        for opp in opposites_dict.get(s, []):
            if opp in groups_by_rep.get(rep_v, []):
                return True
    return False
