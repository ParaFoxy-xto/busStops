# graph/utils.py

import math
from typing import Any, List, Tuple
import networkx as nx


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em metros entre dois pontos geográficos usando a fórmula de Haversine.
    """
    R = 6371000  # raio médio da Terra em metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_bus_stop(
    G: nx.MultiDiGraph,
    bus_stops: List[Any],
    coords: Tuple[float, float]
) -> Any:
    """
    Encontra a parada de ônibus mais próxima das coordenadas fornecidas.

    Args:
        G: grafo com atributos 'x' e 'y' nos nós.
        bus_stops: lista de nós que são paradas de ônibus.
        coords: tupla (latitude, longitude).

    Returns:
        O nó de parada de ônibus mais próximo.
    """
    lat0, lon0 = coords
    best_node = None
    min_dist = float('inf')
    for stop in bus_stops:
        data = G.nodes[stop]
        lat = float(data.get('y', 0))
        lon = float(data.get('x', 0))
        d = haversine_distance(lat0, lon0, lat, lon)
        if d < min_dist:
            min_dist = d
            best_node = stop
    return best_node

# Exemplo de uso:
# start_coords = (-15.7755448, -47.8739389)
# exit_coords = (-15.760521, -47.8741429)
# start_node = find_nearest_bus_stop(G, bus_stops, start_coords)
# exit_node = find_nearest_bus_stop(G, bus_stops, exit_coords)
# src/rota_aco/graph/utils.py

def find_edges_between_opposites(
    G, 
    opposites_proximity: dict, 
    opposites_access: dict
) -> list[tuple]:
    """
    Retorna a lista de arestas (u, v) em G tais que u e v sejam considerados opostos,
    seja por proximidade (opposites_proximity) ou por acesso (opposites_access).
    """
    bad = []
    for u, v in G.edges():
        # Checa se (u,v) ou (v,u) aparecem como opostos
        if v in opposites_proximity.get(u, []) or u in opposites_proximity.get(v, []):
            bad.append((u, v, "proximity"))
        elif v in opposites_access.get(u, []) or u in opposites_access.get(v, []):
            bad.append((u, v, "access"))
    return bad
