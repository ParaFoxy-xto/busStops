"""
Funções utilitárias para os algoritmos ACO.

Este módulo contém funções auxiliares utilizadas pelos algoritmos
ACS-TIME e ACS-VEHICLE.
"""

import networkx as nx
import random
from typing import List, Any, Dict, Tuple, Set, Optional


def route_has_opposite_violation(route: List[Any], opposites: Dict[Any, List[Any]]) -> bool:
    """
    Verifica se uma rota tem violações de paradas opostas.
    
    Args:
        route: Lista de nós da rota
        opposites: Dicionário de paradas opostas
        
    Returns:
        bool: True se há violação, False caso contrário
    """
    if not route or not opposites:
        return False
    
    visited_stops = set(route)
    
    for stop in route:
        if stop in opposites:
            opposite_stops = opposites[stop]
            for opp_stop in opposite_stops:
                if opp_stop in visited_stops:
                    return True
    
    return False


def filter_candidates(graph: nx.DiGraph, 
                     current_node: Any, 
                     forbidden_nodes: Set[Any], 
                     visited_in_route: Set[Any]) -> List[Any]:
    """
    Filtra candidatos válidos para o próximo nó.
    
    Args:
        graph: Grafo de navegação
        current_node: Nó atual
        forbidden_nodes: Nós proibidos
        visited_in_route: Nós já visitados na rota atual
        
    Returns:
        List[Any]: Lista de candidatos válidos
    """
    if current_node not in graph:
        return []
    
    candidates = []
    for neighbor in graph.successors(current_node):
        if neighbor not in forbidden_nodes and neighbor not in visited_in_route:
            candidates.append(neighbor)
    
    return candidates


def select_next_node(current_node: Any,
                    candidates: List[Any],
                    pheromones: Dict[Tuple[Any, Any], float],
                    meta_edges: Dict[Tuple[Any, Any], dict],
                    alpha: float,
                    beta: float) -> Optional[Any]:
    """
    Seleciona o próximo nó baseado em feromônios e heurística.
    
    Args:
        current_node: Nó atual
        candidates: Lista de candidatos
        pheromones: Tabela de feromônios
        meta_edges: Arestas do meta-grafo com informações
        alpha: Peso do feromônio
        beta: Peso da heurística
        
    Returns:
        Any: Próximo nó selecionado ou None
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Calcular probabilidades
    probabilities = []
    total_weight = 0.0
    
    for candidate in candidates:
        edge = (current_node, candidate)
        
        # Feromônio
        pheromone = pheromones.get(edge, 1.0)
        
        # Heurística (inverso da distância/tempo)
        edge_info = meta_edges.get(edge, {})
        distance = edge_info.get('time', edge_info.get('distance', 1.0))
        heuristic = 1.0 / max(distance, 0.1)
        
        # Peso combinado
        weight = (pheromone ** alpha) * (heuristic ** beta)
        probabilities.append(weight)
        total_weight += weight
    
    if total_weight == 0:
        return random.choice(candidates)
    
    # Normalizar probabilidades
    probabilities = [p / total_weight for p in probabilities]
    
    # Seleção por roleta
    r = random.random()
    cumulative = 0.0
    
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return candidates[i]
    
    return candidates[-1]  # Fallback


def close_route_to_exit(route: List[Any],
                       graph: nx.DiGraph,
                       exit_node: Any,
                       meta_edges: Dict[Tuple[Any, Any], dict],
                       opposites: Dict[Any, List[Any]]) -> Tuple[Optional[List[Any]], float]:
    """
    Tenta fechar uma rota até o nó de saída.
    
    Args:
        route: Rota parcial atual
        graph: Grafo de navegação
        exit_node: Nó de saída
        meta_edges: Arestas do meta-grafo
        opposites: Paradas opostas
        
    Returns:
        Tuple[List[Any], float]: Rota fechada e custo, ou (None, 0.0) se falhar
    """
    if not route:
        return None, 0.0
    
    current_node = route[-1]
    
    # Se já está no nó de saída
    if current_node == exit_node:
        return route.copy(), 0.0
    
    # Tentar caminho direto
    if graph.has_edge(current_node, exit_node):
        closed_route = route + [exit_node]
        
        # Verificar violações de opostos
        if not route_has_opposite_violation(closed_route, opposites):
            edge_cost = meta_edges.get((current_node, exit_node), {}).get('time', 0.0)
            return closed_route, edge_cost
    
    # Tentar caminho através de vizinhos (busca simples)
    for neighbor in graph.successors(current_node):
        if neighbor == exit_node:
            continue
        
        if graph.has_edge(neighbor, exit_node):
            potential_route = route + [neighbor, exit_node]
            
            if not route_has_opposite_violation(potential_route, opposites):
                cost1 = meta_edges.get((current_node, neighbor), {}).get('time', 0.0)
                cost2 = meta_edges.get((neighbor, exit_node), {}).get('time', 0.0)
                return potential_route, cost1 + cost2
    
    return None, 0.0


def calculate_route_cost(route: List[Any], 
                        meta_edges: Dict[Tuple[Any, Any], dict]) -> float:
    """
    Calcula o custo total de uma rota.
    
    Args:
        route: Lista de nós da rota
        meta_edges: Arestas com informações de custo
        
    Returns:
        float: Custo total da rota
    """
    if len(route) < 2:
        return 0.0
    
    total_cost = 0.0
    for i in range(len(route) - 1):
        edge = (route[i], route[i + 1])
        edge_info = meta_edges.get(edge, {})
        cost = edge_info.get('time', edge_info.get('distance', 0.0))
        total_cost += cost
    
    return total_cost


def validate_route_constraints(route: List[Any],
                              capacity: int,
                              opposites: Dict[Any, List[Any]],
                              demand_per_stop: int = 10) -> Dict[str, Any]:
    """
    Valida se uma rota atende às restrições do problema.
    
    Args:
        route: Lista de nós da rota
        capacity: Capacidade máxima do veículo
        opposites: Paradas opostas
        demand_per_stop: Demanda por parada
        
    Returns:
        Dict: Informações sobre violações e validade
    """
    result = {
        'is_valid': True,
        'capacity_violations': 0,
        'opposite_violations': 0,
        'total_demand': 0,
        'violations': []
    }
    
    if not route:
        return result
    
    # Verificar capacidade
    total_demand = len([stop for stop in route if stop != route[0] and stop != route[-1]]) * demand_per_stop
    result['total_demand'] = total_demand
    
    if total_demand > capacity:
        result['capacity_violations'] = 1
        result['is_valid'] = False
        result['violations'].append(f"Capacidade excedida: {total_demand} > {capacity}")
    
    # Verificar opostos
    if route_has_opposite_violation(route, opposites):
        result['opposite_violations'] = 1
        result['is_valid'] = False
        result['violations'].append("Violação de paradas opostas")
    
    return result