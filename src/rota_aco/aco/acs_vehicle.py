# src/rota_aco/aco/acs_vehicle.py
"""
Colônia ACS focada na minimização do número de rotas necessárias.
Cada formiga constrói um conjunto de rotas no meta‐grafo até cobrir todas as paradas.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple
from rota_aco.graph.build_meta import resolver_TSP

class ACSVehicle:
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops: List[Any],
        start_node: Any,
        pheromone_matrix: Dict[Tuple[Any, Any], float],
        evaporation: float = 0.1,
        Q: float = 1.0
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops = stops
        self.start_node = start_node
        self.tau = pheromone_matrix   # tau_vehicle compartilhado
        self.rho = evaporation
        self.Q = Q

    def iterate(self, n_ants: int) -> Tuple[List[List[Any]], int]:
        """
        Executa n_formigas iterações no meta‐grafo:
          - Cada formiga faz TSP heurístico repetidas vezes, removendo paradas cobertas
          - Conta quantas rotas foram necessárias (número de veículos)
        Retorna:
          best_meta_routes: lista de meta‐rotas (cada rota é lista de nós do meta‐grafo)
          best_count: número mínimo de rotas encontrado
        """
        best_meta_routes: List[List[Any]] = []
        best_count: int = float('inf')

        for _ in range(n_ants):
            remaining = set(self.stops)
            routes = []
            # gera rotas até cobrir todas as paradas
            while remaining:
                meta_route = resolver_TSP(self.graph, self.start_node)
                routes.append(meta_route)
                covered = set(meta_route) & remaining
                remaining -= covered
            count = len(routes)
            if count < best_count:
                best_count = count
                best_meta_routes = routes

        # evaporação global
        for edge in list(self.tau.keys()):
            self.tau[edge] *= (1 - self.rho)

        # reforço elitista: deposita feromônio em cada aresta usada
        # inversamente proporcional ao número de rotas
        for route in best_meta_routes:
            for u, v in zip(route, route[1:]):
                self.tau[(u, v)] = self.tau.get((u, v), 0) + self.Q / best_count

        return best_meta_routes, best_count

    def get_pheromone(self) -> Dict[Tuple[Any, Any], float]:
        return self.tau
