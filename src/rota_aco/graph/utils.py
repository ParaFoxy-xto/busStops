# src/rota_aco/graph/utils.py

"""
Funções de utilidade para operações em grafos e cálculos geoespaciais.
"""

import math
from typing import Any, List, Tuple
import networkx as nx

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em metros entre dois pontos geográficos usando a fórmula de Haversine.

    Args:
        lat1, lon1: Coordenadas do primeiro ponto.
        lat2, lon2: Coordenadas do segundo ponto.

    Returns:
        A distância em metros.
    """
    R = 6371000  # Raio médio da Terra em metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def find_nearest_node(
    graph: nx.MultiDiGraph,
    target_coords: Tuple[float, float],
    node_list: List[Any]
) -> Any:
    """
    Encontra o nó em uma lista que está mais próximo de um dado par de coordenadas.

    Args:
        graph: O grafo contendo os dados dos nós (com atributos 'x' e 'y').
        target_coords: A tupla de coordenadas (latitude, longitude) de referência.
        node_list: A lista de nós candidatos para a busca.

    Returns:
        O ID do nó mais próximo.
    """
    target_lat, target_lon = target_coords
    best_node = None
    min_dist = float('inf')

    for node in node_list:
        try:
            node_data = graph.nodes[node]
            lat = float(node_data['y'])
            lon = float(node_data['x'])
            
            dist = haversine_distance(target_lat, target_lon, lat, lon)
            
            if dist < min_dist:
                min_dist = dist
                best_node = node
        except (KeyError, TypeError):
            # Ignora nós que não possuem coordenadas válidas.
            continue
            
    return best_node