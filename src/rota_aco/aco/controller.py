# src/rota_aco/aco/controller.py
"""
Controlador que orquestra as colônias ACSTime e ACSVehicle,
mantendo uma matriz global de feromônios compartilhada.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple
from rota_aco.aco.acs_time import ACSTime
from rota_aco.aco.acs_vehicle import ACSVehicle

class ACSController:
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops: List[Any],
        start_node: Any,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.1,
        Q: float = 1.0
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops = stops
        self.start_node = start_node
        self.rho_global = evaporation

        # inicia tau_global apenas para arestas válidas em meta_edges
        self.tau_global: Dict[Tuple[Any, Any], float] = {
            edge: 1.0 for edge in meta_edges.keys()
        }

        self.acs_time = ACSTime(
            graph,
            meta_edges,
            stops,
            start_node,
            self.tau_global,
            alpha,
            beta,
            evaporation,
            Q
        )
        self.acs_vehicle = ACSVehicle(
            graph,
            meta_edges,
            stops,
            start_node,
            self.tau_global,
            evaporation,
            Q
        )

    def run(
        self,
        n_iterations: int,
        ants_time: int,
        ants_vehicle: int,
        lam: float = 0.5
    ) -> Tuple[List[List[Any]], float, int]:
        best_routes: List[List[Any]] = []
        best_dist: float = float('inf')
        best_count: int = float('inf')

        for _ in range(n_iterations):
            # 1) ACS-TIME
            meta_route, dist_time = self.acs_time.iterate(ants_time)
            tau_time = self.acs_time.get_pheromone()

            # 2) ACS-VEHICLE
            meta_routes, count_vehicle = self.acs_vehicle.iterate(ants_vehicle)
            tau_vehicle = self.acs_vehicle.get_pheromone()

            # 3) Mescla global de feromônios
            for edge in self.tau_global:
                t_time = tau_time.get(edge, 0.0)
                t_veh  = tau_vehicle.get(edge, 0.0)
                self.tau_global[edge] = (lam * t_time + (1 - lam) * t_veh) * (1 - self.rho_global)


            # 4) Atualiza melhor solução
            if (count_vehicle < best_count) or (count_vehicle == best_count and dist_time < best_dist):
                best_routes = meta_routes if isinstance(meta_routes, list) else [meta_route]
                best_count = count_vehicle
                best_dist = dist_time

        return best_routes, best_dist, best_count
