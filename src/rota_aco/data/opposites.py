# src/rota_aco/data/opposites.py

import json
import os
from typing import Any, Dict, List, Optional
"""
Funções para detectar pares de paradas de ônibus opostas.

Uma parada é considerada oposta a outra por duas lógicas principais:
1. Proximidade: Duas paradas estão no mesmo local geográfico (ruas de mão dupla).
2. Acesso: Duas paradas estão conectadas por uma via de acesso de ônibus,
   indicando que uma serve o fluxo de ida e a outra, o de volta.
"""

from typing import Any, Dict, List, Set

import networkx as nx

def find_opposites_by_proximity(
    bus_stops: List[Any],
    graph: nx.MultiDiGraph,
    precision: int = 5
) -> Dict[Any, List[Any]]:
    """
    Identifica paradas opostas com base na proximidade geoespacial.

    Agrupa as paradas por coordenadas (arredondadas para a precisão definida).
    Se mais de uma parada cair no mesmo grupo, elas são consideradas opostas.

    Args:
        bus_stops: Lista de nós que são paradas de ônibus.
        graph: O grafo completo com dados de coordenadas nos nós.
        precision: O número de casas decimais para arredondar as coordenadas.

    Returns:
        Um dicionário mapeando cada parada para uma lista de suas opostas por proximidade.
    """
    groups: Dict[tuple, List[Any]] = {}
    for stop_node in bus_stops:
        try:
            node_data = graph.nodes[stop_node]
            # Cria uma chave de localização arredondando as coordenadas
            location_key = (
                round(float(node_data['y']), precision),
                round(float(node_data['x']), precision)
            )
            groups.setdefault(location_key, []).append(stop_node)
        except (KeyError, TypeError):
            # Ignora paradas sem coordenadas válidas
            continue

    opposites: Dict[Any, List[Any]] = {}
    for location_key, stops_in_group in groups.items():
        # Se um grupo tem mais de uma parada, elas são opostas entre si
        if len(stops_in_group) > 1:
            for stop in stops_in_group:
                # O oposto de 'stop' são todas as outras paradas no mesmo grupo
                opposites[stop] = [other for other in stops_in_group if other != stop]

    return opposites

def _is_bus_access_edge(data: dict) -> bool:
    """Verifica se uma aresta é uma via de acesso para ônibus."""
    # A verificação primária é pelo atributo 'highway'
    if str(data.get("highway", "")).strip().lower() == "bus_access":
        return True
    # Uma verificação secundária, menos comum, em outros valores
    return any(str(v).strip().lower() == "bus_access" for v in data.values())

def find_opposites_by_access(
    graph: nx.MultiDiGraph,
    bus_stops: List[Any]
) -> Dict[Any, List[Any]]:
    """
    Identifica pares de paradas opostas conectadas por uma via de acesso.

    O padrão típico é: Parada_A <--> Nó_de_Acesso <--> Parada_B.
    Esta função encontra esses padrões.

    Args:
        graph: O grafo completo da rede de ruas.
        bus_stops: Lista de nós que são paradas de ônibus.

    Returns:
        Um dicionário mapeando cada parada para uma lista de suas opostas por acesso.
    """
    opposites: Dict[Any, List[Any]] = {stop: [] for stop in bus_stops}
    bus_stops_set: Set[Any] = set(bus_stops)

    # Itera sobre todas as arestas do grafo que são de acesso de ônibus
    for u, v, data in graph.edges(data=True):
        if not _is_bus_access_edge(data):
            continue

        # 'u' é uma parada e 'v' é um nó de acesso (não é uma parada)
        if u in bus_stops_set and v not in bus_stops_set:
            # Procura por outras paradas conectadas a este nó de acesso 'v'
            for neighbor in graph.neighbors(v):
                # Se o vizinho é uma parada diferente da original, elas são opostas
                if neighbor in bus_stops_set and neighbor != u:
                    opposites[u].append(neighbor)
                    opposites[neighbor].append(u)

    # Remove duplicatas que possam ter sido adicionadas
    for stop in opposites:
        opposites[stop] = sorted(list(set(opposites[stop])))

    return opposites

# Em src/rota_aco/data/opposites.py

def combine_opposites(
    prox_map: Dict[Any, List[Any]], 
    access_map: Dict[Any, List[Any]],
    manual_map: Dict[Any, List[Any]]  # Adicionado
) -> Dict[Any, List[Any]]:
    """Combina múltiplos dicionários de opostos em um único dicionário."""
    combined = {}
    
    # Itera sobre todos os dicionários de entrada
    for source_map in (prox_map, access_map, manual_map):
        for stop, opp_list in source_map.items():
            combined.setdefault(stop, []).extend(opp_list)
            
    # Remove duplicatas de cada lista de opostos
    for stop in combined:
        combined[stop] = sorted(list(set(combined[stop])))
        
    return combined



def load_manual_opposites(file_path: Optional[str]) -> Dict[Any, List[Any]]:
    """
    Carrega definições de opostos de um arquivo JSON.
    O formato esperado é uma lista de objetos, cada um com "group1" e "group2".
    """
    if not file_path or not os.path.exists(file_path):
        return {}

    manual_opposites: Dict[Any, List[Any]] = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for rule in data:
            group1 = rule.get("group1", [])
            group2 = rule.get("group2", [])

            if not group1 or not group2:
                continue

            # Garante a simetria: todos em G1 são opostos a todos em G2
            for node1 in group1:
                manual_opposites.setdefault(node1, []).extend(group2)
            
            # E vice-versa
            for node2 in group2:
                manual_opposites.setdefault(node2, []).extend(group1)

        # Limpa duplicatas que possam ter sido adicionadas
        for key in manual_opposites:
            manual_opposites[key] = sorted(list(set(manual_opposites[key])))
            
        return manual_opposites
    except (json.JSONDecodeError, IOError) as e:
        print(f"[AVISO] Não foi possível ler ou processar o arquivo de opostos manuais: {e}")
        return {}