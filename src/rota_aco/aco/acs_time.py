# src/rota_aco/aco/acs_time.py
"""
Colônia ACS focada na minimização da distância total.
Cada formiga faz um TSP simples no meta-grafo cobrindo todas as paradas.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple
from rota_aco.graph.build_meta import resolver_TSP

class ACSTime:
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops: List[Any],
        start_node: Any,
        pheromone_matrix: Dict[Tuple[Any, Any], float],
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.1,
        Q: float = 1.0
    ):
        self.graph = graph
        self.meta_edges = meta_edges      # dicionário (u, v) -> {'length', 'path'}
        self.stops = stops
        self.start_node = start_node
        self.tau = pheromone_matrix       # matriz global compartilhada
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation
        self.Q = Q

    def iterate(self, n_ants: int) -> Tuple[List[Any], float]:
        """
        Executa n_formigas iterações de TSP heurístico no meta-grafo.
        Retorna:
          best_meta_route: lista de nós do meta-grafo
          best_dist: distância total (soma de custos em meta_edges)
        """
        best_meta_route, best_dist = [], float('inf')

        for _ in range(n_ants):
            route = resolver_TSP(self.graph, self.start_node)
            # soma de custos a partir de meta_edges (usa sempre 'length')
            dist = sum(
                self.meta_edges[(u, v)]['length']
                for u, v in zip(route, route[1:])
                if (u, v) in self.meta_edges
            )
            if dist < best_dist:
                best_dist = dist
                best_meta_route = route

        # evaporação global
        for edge in list(self.tau.keys()):
            self.tau[edge] *= (1 - self.rho)

        # reforço elitista sobre a melhor rota
        for u, v in zip(best_meta_route, best_meta_route[1:]):
            self.tau[(u, v)] = self.tau.get((u, v), 0.0) + (self.Q / best_dist)

        return best_meta_route, best_dist

    def get_pheromone(self) -> Dict[Tuple[Any, Any], float]:
        return self.tau
