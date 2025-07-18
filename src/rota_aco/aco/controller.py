# src/rota_aco/aco/controller.py

"""
Controlador do sistema multi-colônia de formigas (ACS).

Este módulo orquestra a interação entre diferentes colônias de formigas
(ACSTime e ACSVehicle), gerencia a tabela de feromônios compartilhada e
dirige o processo de otimização geral.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple, Optional

from rota_aco.aco.acs_time import ACSTime
from rota_aco.aco.acs_vehicle import ACSVehicle

class ACSController:
    """
    Orquestra múltiplas colônias de formigas para encontrar uma solução de roteamento
    balanceada, avaliando as propostas de cada colônia em relação a uma função
    de qualidade global.
    """
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops_to_visit: List[Any],
        start_node: Any,
        exit_node: Any,
        opposites: Dict[Any, List[Any]],
        aco_params: Dict[str, Any],
        problem_params: Dict[str, Any],
    ):
        # Parâmetros do problema
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops_to_visit = stops_to_visit
        self.start_node = start_node
        self.exit_node = exit_node
        self.opposites = opposites
        self.capacity = problem_params.get('capacity', 70)
        self.max_route_attempts = problem_params.get('max_route_attempts', 10)
        self.max_route_length = problem_params.get('max_route_length', 100)

        # Parâmetros do algoritmo ACO
        self.alpha = aco_params.get('alpha', 1.0)
        self.beta = aco_params.get('beta', 2.0)
        self.rho = aco_params.get('rho', 0.1)  # Taxa de evaporação
        self.q_param = aco_params.get('Q', 1.0) # Fator de reforço de feromônio
        
        # Tabela de feromônios compartilhada
        self.pheromones: Dict[Tuple[Any, Any], float] = {
            edge: 1.0 for edge in meta_edges.keys()
        }
        
        # Histórico para análise de convergência
        self.history: List[Dict[str, Any]] = []

    def run(
        self,
        n_ants: int,
        n_iterations: int,
        quality_weights: Dict[str, float],
        verbose: bool = False
    ) -> Tuple[List[List[Any]], float, int, float]:
        """
        Executa o loop de otimização principal do ACS multi-colônia.
        """
        w_c = quality_weights.get('w_c', 10.0)
        w_r = quality_weights.get('w_r', 1.0)
        w_d = quality_weights.get('w_d', 0.5)

        common_args = {
            'graph': self.graph, 'meta_edges': self.meta_edges, 'stops_to_visit': self.stops_to_visit,
            'start_node': self.start_node, 'exit_node': self.exit_node, 'opposites': self.opposites,
            'pheromones': self.pheromones, 'alpha': self.alpha, 'beta': self.beta, 'capacity': self.capacity,
            'max_route_attempts': self.max_route_attempts, 'verbose': verbose
        }
        
        acs_time = ACSTime(**common_args)
        acs_vehicle = ACSVehicle(**common_args, max_route_length=self.max_route_length)

        best_solution_so_far: List[List[Any]] = []
        best_quality_so_far: float = -float('inf')
        no_improvement_streak = 0
        
        for iteration_num in range(n_iterations): # Mudei para iteration_num para clareza
            time_solution, _, _, _ = acs_time.iterate(n_ants)
            vehicle_solution, _, _, _ = acs_vehicle.iterate(n_ants)
            
            quality_time = self._calculate_solution_quality(time_solution, w_c, w_r, w_d)
            quality_vehicle = self._calculate_solution_quality(vehicle_solution, w_c, w_r, w_d)
            
            current_best_solution = time_solution if quality_time >= quality_vehicle else vehicle_solution
            current_best_quality = max(quality_time, quality_vehicle)
            
            if current_best_quality > best_quality_so_far:
                best_solution_so_far = current_best_solution
                best_quality_so_far = current_best_quality
                no_improvement_streak = 0
                if verbose:
                    print(f"[CONTROLLER Iter {iteration_num+1}] Nova melhor solução global! Qualidade: {best_quality_so_far:.4f}")
            else:
                no_improvement_streak += 1

            if verbose:
                print(f"[CONTROLLER Iter {iteration_num+1}] Melhor global: {best_quality_so_far:.4f}")

            self._update_pheromones(best_solution_so_far, best_quality_so_far)

            # --- CORREÇÃO APLICADA AQUI ---
            # Grava o histórico detalhado da iteração com dados de ambas as colônias
            self._record_history(
                iteration_num=iteration_num,
                time_solution=time_solution,
                vehicle_solution=vehicle_solution,
                chosen_solution_quality=current_best_quality,
                best_overall_quality=best_quality_so_far,
            )

            if no_improvement_streak >= 5 and iteration_num > (n_iterations / 2):
                if verbose: print("\n[CONTROLLER] Estagnação detectada. Encerrando otimização.")
                break

        final_dist, final_count, final_coverage = self._get_solution_metrics(best_solution_so_far)
        return best_solution_so_far, final_dist, final_count, final_coverage

    def _calculate_solution_quality(self, routes: List[List[Any]], w_c: float, w_r: float, w_d: float) -> float:
        if not routes: return -float('inf')
        dist, count, coverage = self._get_solution_metrics(routes)
        normalized_dist_penalty = dist / (1 + dist)
        
        # Penalidade por número de rotas com sweet spot em 3 rotas
        optimal_routes = 3
        route_penalty = w_r * abs(count - optimal_routes)
        
        quality = (w_c * coverage) - route_penalty - (w_d * normalized_dist_penalty)
        return quality

    def _update_pheromones(self, routes: List[List[Any]], quality: float):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
        if not routes or quality <= 0: return
        reinforcement_amount = self.q_param * quality
        for route in routes:
            for u, v in zip(route, route[1:]):
                if (u, v) in self.pheromones:
                    self.pheromones[(u, v)] += reinforcement_amount

    def _get_solution_metrics(self, routes: List[List[Any]]) -> Tuple[float, int, float]:
        if not routes: return 0.0, 0, 0.0
        total_dist = sum(self.meta_edges.get((u, v), {}).get('time', 0) for r in routes for u, v in zip(r, r[1:]))
        num_routes = len(routes)
        covered_stops = {s for r in routes for s in r if s in self.stops_to_visit}
        coverage = len(covered_stops) / max(1, len(self.stops_to_visit)) if self.stops_to_visit else 1.0
        return total_dist, num_routes, coverage

    
    def _record_history(
        self,
        iteration_num: int,
        time_solution: List[List[Any]],
        vehicle_solution: List[List[Any]],
        chosen_solution_quality: float,
        best_overall_quality: float,
    ):
        """Salva as métricas da iteração atual para ambas as colônias."""
        time_dist, time_count, time_coverage = self._get_solution_metrics(time_solution)
        vehicle_dist, vehicle_count, vehicle_coverage = self._get_solution_metrics(vehicle_solution)

        self.history.append({
            'iteration': iteration_num,
            'time_metrics': {'dist': time_dist, 'count': time_count, 'coverage': time_coverage},
            'vehicle_metrics': {'dist': vehicle_dist, 'count': vehicle_count, 'coverage': vehicle_coverage},
            'chosen_quality': chosen_solution_quality,
            'best_quality_so_far': best_overall_quality
        })