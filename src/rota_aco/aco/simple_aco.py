# src/rota_aco/aco/simple_aco.py

"""
Implementação simples de ACO (Ant Colony Optimization) para comparação.

Esta versão simplificada foca apenas na otimização de rotas básicas,
sem as complexidades do sistema multi-colônia.
"""

import networkx as nx
import random
import math
from typing import List, Any, Dict, Tuple, Optional


class SimpleACO:
    """
    Implementação básica do algoritmo ACO para otimização de rotas.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops_to_visit: List[Any],
        start_node: Any,
        exit_node: Any,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q_param: float = 1.0
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops_to_visit = set(stops_to_visit)
        self.start_node = start_node
        self.exit_node = exit_node
        
        # Parâmetros do ACO
        self.alpha = alpha  # Importância do feromônio
        self.beta = beta    # Importância da heurística
        self.rho = rho      # Taxa de evaporação
        self.q_param = q_param  # Fator de reforço
        
        # Inicializar feromônios
        self.pheromones = {edge: 1.0 for edge in meta_edges.keys()}
        
        # Histórico para análise
        self.history = []
    
    def run(self, n_ants: int = 10, n_iterations: int = 50, verbose: bool = False) -> Tuple[List[Any], float, Dict]:
        """
        Executa o algoritmo ACO simples.
        
        Returns:
            Tuple contendo: (melhor_rota, melhor_distancia, estatisticas)
        """
        best_route = []
        best_distance = float('inf')
        
        for iteration in range(n_iterations):
            # Construir soluções para todas as formigas
            routes = []
            distances = []
            
            for ant in range(n_ants):
                route = self._construct_route()
                if route:
                    distance = self._calculate_route_distance(route)
                    routes.append(route)
                    distances.append(distance)
            
            # Encontrar melhor solução da iteração
            if distances:
                min_idx = distances.index(min(distances))
                iteration_best_route = routes[min_idx]
                iteration_best_distance = distances[min_idx]
                
                # Atualizar melhor global
                if iteration_best_distance < best_distance:
                    best_route = iteration_best_route.copy()
                    best_distance = iteration_best_distance
                    
                    if verbose:
                        print(f"Iteração {iteration + 1}: Nova melhor distância = {best_distance:.2f}")
            
            # Atualizar feromônios
            self._update_pheromones(routes, distances)
            
            # Registrar histórico
            self.history.append({
                'iteration': iteration,
                'best_distance': best_distance,
                'iteration_best': iteration_best_distance if distances else float('inf'),
                'avg_distance': sum(distances) / len(distances) if distances else float('inf')
            })
        
        # Calcular estatísticas finais
        coverage = self._calculate_coverage(best_route)
        stats = {
            'total_distance': best_distance,
            'coverage': coverage,
            'route_length': len(best_route),
            'iterations': n_iterations,
            'ants_per_iteration': n_ants
        }
        
        return best_route, best_distance, stats
    
    def _construct_route(self) -> List[Any]:
        """
        Constrói uma rota usando a regra de transição probabilística do ACO.
        Garante que todos os pontos obrigatórios sejam visitados.
        """
        route = [self.start_node]
        current_node = self.start_node
        visited_stops = set()
        max_steps = 100  # Evitar loops infinitos
        
        for _ in range(max_steps):
            # Se visitamos todos os pontos, ir para o nó de saída
            if len(visited_stops) >= len(self.stops_to_visit):
                # Tentar ir diretamente para o nó de saída
                if self.exit_node in self.graph.successors(current_node):
                    route.append(self.exit_node)
                    break
                else:
                    # Se não conseguir ir diretamente, continuar navegando
                    possible_nodes = list(self.graph.successors(current_node))
                    if not possible_nodes:
                        break
                    
                    # Priorizar nós que levam ao exit_node
                    exit_neighbors = []
                    for node in possible_nodes:
                        if self.exit_node in self.graph.successors(node):
                            exit_neighbors.append(node)
                    
                    if exit_neighbors:
                        next_node = exit_neighbors[0]  # Escolher o primeiro que leva ao exit
                    else:
                        probabilities = self._calculate_transition_probabilities(current_node, possible_nodes)
                        next_node = self._select_next_node(possible_nodes, probabilities)
                    
                    route.append(next_node)
                    current_node = next_node
                    continue
            
            # Obter próximos nós possíveis
            possible_nodes = list(self.graph.successors(current_node))
            if not possible_nodes:
                break
            
            # Priorizar pontos não visitados
            unvisited_stops = [node for node in possible_nodes if node in self.stops_to_visit and node not in visited_stops]
            
            if unvisited_stops:
                # Se há pontos não visitados acessíveis, priorizar eles
                probabilities = self._calculate_transition_probabilities(current_node, unvisited_stops)
                next_node = self._select_next_node(unvisited_stops, probabilities)
            else:
                # Senão, usar todos os nós possíveis
                probabilities = self._calculate_transition_probabilities(current_node, possible_nodes)
                next_node = self._select_next_node(possible_nodes, probabilities)
            
            route.append(next_node)
            
            # Atualizar estado
            if next_node in self.stops_to_visit:
                visited_stops.add(next_node)
            
            current_node = next_node
            
            # Se chegamos ao nó de saída, parar
            if current_node == self.exit_node:
                break
        
        # Se não conseguiu visitar todos os pontos, penalizar a rota
        if len(visited_stops) < len(self.stops_to_visit):
            # Adicionar penalidade alta para rotas incompletas
            return []
        
        return route
    
    def _calculate_transition_probabilities(self, current_node: Any, possible_nodes: List[Any]) -> List[float]:
        """
        Calcula as probabilidades de transição para os nós possíveis.
        """
        probabilities = []
        
        for node in possible_nodes:
            edge = (current_node, node)
            
            # Feromônio
            pheromone = self.pheromones.get(edge, 1.0)
            
            # Heurística (inverso da distância/tempo)
            edge_data = self.meta_edges.get(edge, {})
            distance = edge_data.get('time', 1.0)
            heuristic = 1.0 / (distance + 0.1)  # Evitar divisão por zero
            
            # Fórmula do ACO
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(probability)
        
        # Normalizar probabilidades
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Se todas as probabilidades são zero, usar distribuição uniforme
            probabilities = [1.0 / len(possible_nodes)] * len(possible_nodes)
        
        return probabilities
    
    def _select_next_node(self, possible_nodes: List[Any], probabilities: List[float]) -> Any:
        """
        Seleciona o próximo nó baseado nas probabilidades calculadas.
        """
        r = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return possible_nodes[i]
        
        # Fallback: retornar último nó
        return possible_nodes[-1]
    
    def _calculate_route_distance(self, route: List[Any]) -> float:
        """
        Calcula a distância total de uma rota.
        """
        total_distance = 0.0
        
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            edge_data = self.meta_edges.get(edge, {})
            distance = edge_data.get('time', 0.0)
            total_distance += distance
        
        return total_distance
    
    def _calculate_coverage(self, route: List[Any]) -> float:
        """
        Calcula a cobertura dos pontos de interesse.
        """
        if not self.stops_to_visit:
            return 1.0
        
        visited_stops = set(node for node in route if node in self.stops_to_visit)
        return len(visited_stops) / len(self.stops_to_visit)
    
    def _update_pheromones(self, routes: List[List[Any]], distances: List[float]):
        """
        Atualiza os níveis de feromônio nas arestas.
        """
        # Evaporação
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
        
        # Reforço baseado na qualidade das soluções
        for route, distance in zip(routes, distances):
            if distance > 0:
                delta_pheromone = self.q_param / distance
                
                for i in range(len(route) - 1):
                    edge = (route[i], route[i + 1])
                    if edge in self.pheromones:
                        self.pheromones[edge] += delta_pheromone
    
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas da execução.
        """
        if not self.history:
            return {}
        
        best_distances = [h['best_distance'] for h in self.history]
        iteration_bests = [h['iteration_best'] for h in self.history if h['iteration_best'] != float('inf')]
        
        return {
            'final_best_distance': min(best_distances) if best_distances else float('inf'),
            'convergence_iteration': next((i for i, h in enumerate(self.history) 
                                         if h['best_distance'] == min(best_distances)), -1),
            'avg_iteration_best': sum(iteration_bests) / len(iteration_bests) if iteration_bests else float('inf'),
            'improvement_iterations': len([h for h in self.history if h['iteration_best'] < float('inf')])
        }