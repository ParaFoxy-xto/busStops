# src/rota_aco/aco/acs_vehicle.py

"""
Implementação da colônia de formigas ACSVehicle.

Esta colônia foca na minimização do número de veículos (rotas) necessários
para cobrir todas as paradas.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple, Set, Optional

from rota_aco.aco.utils import (
    route_has_opposite_violation,
    filter_candidates,
    select_next_node,
    close_route_to_exit,
)

class ACSVehicle:
    """
    Uma colônia de formigas cujo objetivo é encontrar um conjunto de rotas
    que utilize o menor número de veículos (rotas), cobrindo o máximo de paradas.
    """
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops_to_visit: List[Any],
        start_node: Any,
        exit_node: Any,
        opposites: Dict[Any, List[Any]],
        pheromones: Dict[Tuple[Any, Any], float],
        alpha: float,
        beta: float,
        capacity: int,
        max_route_attempts: int,
        max_route_length: int,
        verbose: bool = False
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops_to_visit = set(stops_to_visit)
        self.start_node = start_node
        self.exit_node = exit_node
        self.opposites = opposites
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.max_route_attempts = max_route_attempts
        self.max_route_length = max_route_length
        self.verbose = verbose
        self.DEMAND_PER_STOP = 10

    def iterate(self, n_ants: int) -> Tuple[List[List[Any]], float, int, float]:
        """
        Executa uma iteração completa da otimização com um número de formigas.
        """
        best_solution_routes: List[List[Any]] = []
        best_solution_count: int = float('inf')
        best_solution_time: float = float('inf')

        for ant_id in range(n_ants):
            if self.verbose:
                print(f"\n--- [ACSVehicle ANT {ant_id + 1}] ---")
            
            demand = {stop: self.DEMAND_PER_STOP for stop in self.stops_to_visit}
            routes, total_time = self._build_solution_for_ant(demand, ant_id)
            
            # A heurística desta colônia é: menor número de rotas, depois menor tempo.
            is_better_solution = (
                not best_solution_routes or
                len(routes) < best_solution_count or
                (len(routes) == best_solution_count and total_time < best_solution_time)
            )

            if is_better_solution:
                best_solution_routes = routes
                best_solution_count = len(routes)
                best_solution_time = total_time
        
        final_coverage = self._calculate_coverage({stop: 10 for stop in self.stops_to_visit} if not best_solution_routes else demand)

        return best_solution_routes, best_solution_time, best_solution_count, final_coverage

    def _build_solution_for_ant(self, demand: Dict[Any, int], ant_id: int) -> Tuple[List[List[Any]], float]:
        """
        Uma única formiga constrói um conjunto de rotas para atender a demanda.
        """
        solution_routes: List[List[Any]] = []
        total_time: float = 0.0
        stops_served_by_ant: Set[Any] = set()

        for attempt in range(self.max_route_attempts):
            if sum(demand.values()) == 0:
                break
            
                        # Tenta construir uma nova rota para atender à demanda restante.
            new_route = self._build_single_route(stops_served_by_ant, ant_id, len(solution_routes) + 1)
            
            if not new_route:
                if self.verbose:
                    print(f"  > [Ant {ant_id+1}] Tentativa {attempt+1}: Nenhuma rota válida pôde ser construída. Encerrando.")
                break

            newly_served_stops = {stop for stop in new_route if stop in demand and demand[stop] > 0}

            if newly_served_stops:
                solution_routes.append(new_route)
                stops_served_by_ant.update(newly_served_stops)
                total_time = self._update_demand_and_time(demand, new_route, total_time)
                if self.verbose:
                    print(f"  > [Ant {ant_id+1}] Rota {len(solution_routes)} aceita: {new_route}")
            elif self.verbose:
                print(f"  > [Ant {ant_id+1}] Tentativa {attempt+1}: Rota construída mas não atendeu nova demanda. Descartando.")
        
        return solution_routes, total_time

    # Em: src/rota_aco/aco/acs_vehicle.py

    def _build_single_route(self, stops_served_by_ant: Set[Any], ant_id: int, route_num: int) -> Optional[List[Any]]:
        """
        Constrói uma única rota, respeitando capacidade, comprimento e restrições.
        (Versão com lógica de construção flexibilizada)
        """
        if self.verbose:
            print(f"    -> [Ant {ant_id+1}, Rota {route_num}] Iniciando construção a partir de {self.start_node}")

        route = [self.start_node]
        visited_in_route = {self.start_node}
        remaining_capacity = self.capacity

        # O `while` aqui inclui a condição de comprimento máximo da rota
        while remaining_capacity > 0 and len(route) < self.max_route_length:
            current_node = route[-1]
            if current_node == self.exit_node:
                if self.verbose: print(f"      - Chegou ao nó de saída. Finalizando construção da rota parcial.")
                break

            if self.verbose: print(f"\n      - Posição atual: {current_node}, Capacidade: {remaining_capacity}, Comprimento: {len(route)}")

            # --- LÓGICA DE PROIBIÇÃO SIMPLIFICADA ---
            # Proíbe apenas nós já visitados nesta rota e opostos de paradas já servidas EM OUTRAS ROTAS.
            forbidden_nodes = set(visited_in_route)
            for served_stop in stops_served_by_ant:
                forbidden_nodes.update(self.opposites.get(served_stop, []))

            if self.verbose:
                print(f"      - Nós já visitados nesta rota: {visited_in_route}")
                print(f"      - Opostos de paradas JÁ SERVIDAS (outras rotas): {[self.opposites.get(n, []) for n in stops_served_by_ant]}")
                print(f"      - Lista de proibidos (simplificada): {forbidden_nodes}")

            # O filter_candidates agora usa a lista de proibidos mais simples
            candidates = filter_candidates(self.graph, current_node, forbidden_nodes, visited_in_route)
            
            if self.verbose: print(f"      - Candidatos válidos (pós-filtro): {candidates}")

            if not candidates:
                if self.verbose: print(f"      - [!!!] Nenhum candidato válido encontrado. Encerrando rota parcial.")
                break

            next_node = select_next_node(current_node, candidates, self.pheromones, self.meta_edges, self.alpha, self.beta)
            if self.verbose: print(f"      - Próximo nó selecionado: {next_node}")
            
            if not next_node: break

            route.append(next_node)
            visited_in_route.add(next_node)
            if self.verbose: print(f"      - Adicionando '{next_node}' à rota. Rota atual: {route}")

            if next_node in self.stops_to_visit:
                remaining_capacity -= self.DEMAND_PER_STOP
                if self.verbose: print(f"      - '{next_node}' é uma parada. Nova capacidade: {remaining_capacity}")
        
        # A validação final na rota completa agora é a ÚNICA que verifica os opostos da rota atual.
        if self.verbose:
            print(f"    -> Rota parcial construída: {route}. Tentando fechar para {self.exit_node}.")
            
        closed_route, _ = close_route_to_exit(route, self.graph, self.exit_node, self.meta_edges, self.opposites)
        
        # A verificação de opostos após fechar a rota continua sendo a guarda principal.
        # A função `close_route_to_exit` já faz uma verificação, mas uma dupla checagem aqui não faz mal.
        if not closed_route or route_has_opposite_violation(closed_route, self.opposites):
            if self.verbose: print(f"      - [FALHA] Não foi possível fechar a rota ou a rota final {closed_route} tem violação de opostos.")
            return None

        if self.verbose: print(f"    -> Rota final válida construída: {closed_route}")
        return closed_route

    def _update_demand_and_time(self, demand: Dict[Any, int], route: List[Any], total_time: float) -> float:
        """
        Atualiza a demanda e o tempo total com base em uma nova rota.
        """
        for stop in route:
            if stop in demand and demand[stop] > 0:
                demand[stop] = 0
                for opp_stop in self.opposites.get(stop, []):
                    if opp_stop in demand:
                        demand[opp_stop] = 0
        
        for u, v in zip(route, route[1:]):
            total_time += self.meta_edges.get((u, v), {}).get('time', 0)
        return total_time

    def _calculate_coverage(self, final_demand: Dict[Any, int]) -> float:
        """
        Calcula a porcentagem de paradas cobertas.
        """
        total_stops = len(self.stops_to_visit)
        if total_stops == 0:
            return 100.0
        
        uncovered_count = sum(1 for demand_val in final_demand.values() if demand_val > 0)
        covered_count = total_stops - uncovered_count
        
        return 100.0 * (covered_count / total_stops)