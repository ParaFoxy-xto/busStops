# src/rota_aco/aco/controller.py
"""
Controlador que orquestra as colônias ACSTime e ACSVehicle,
mantendo uma matriz global de feromônios compartilhada.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple, Optional
from rota_aco.aco.acs_time import ACSTime
from rota_aco.aco.acs_vehicle import ACSVehicle
import csv
import statistics

def route_has_opposite_violation(route: List[Any], opposites: Dict[Any, List[Any]]) -> bool:
    """Check if a route violates the opposite stops constraint."""
    for i, stop in enumerate(route):
        if stop in opposites:
            for opposite in opposites[stop]:
                if opposite in route[i+1:]:
                    return True
    return False


def route_serves_new_stops(route: List[Any], stops: List[Any], already_served_stops: set) -> bool:
    """Check if a route serves any new stops that haven't been served before."""
    route_stops = set(stop for stop in route if stop in stops)
    return len(route_stops - already_served_stops) > 0


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
        Q: float = 1.0,
        opposites: Optional[Dict[Any, List[Any]]] = None,
        exit_node: Any = None
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops = stops
        self.start_node = start_node
        self.rho_global = evaporation
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        if opposites is None:
            opposites = {}
        self.opposites = opposites
        self.exit_node = exit_node
        # inicia tau_global apenas para arestas válidas em meta_edges
        self.tau_global: Dict[Tuple[Any, Any], float] = {
            edge: 1.0 for edge in meta_edges.keys()
        }

    def run(
        self,
        n_iterations: int,
        n_ants: int,
        lam: float = 0.5,
        verbose=False
    ) -> Tuple[List[List[Any]], float, int, List[str], float, List[Any]]:
        import random
        best_routes: List[List[Any]] = []
        best_count: float = float('inf')
        best_total_time: float = float('inf')
        best_route_directions: List[str] = []
        best_coverage: float = 0.0
        best_uncovered: List[Any] = []
        best_solution = None

        v = 1
        max_vehicles = max(1, len(self.stops) // 3)
        tau_global = {k: 1.0 for k in self.meta_edges}
        # --- Detailed convergence logging ---
        convergence_log = []
        while v <= max_vehicles:
            # --- ACS-VEI: Try to maximize coverage with v routes ---
            acs_vei = ACSVehicle(
                self.graph,
                self.meta_edges,
                self.stops,
                self.start_node,
                tau_global,
                self.rho_global,
                self.Q,
                self.opposites,
                self.exit_node
            )
            vei_routes, vei_count = acs_vei.iterate(n_ants)
            vei_covered = set()
            for route in vei_routes:
                vei_covered.update(stop for stop in route if stop in self.stops)
            vei_coverage = 100.0 * len(vei_covered) / max(1, len(self.stops))
            vei_uncovered = sorted(set(self.stops) - vei_covered)
            vei_total_time = 0.0
            for route in vei_routes:
                for u, v_ in zip(route, route[1:]):
                    if (u, v_) in self.meta_edges:
                        vei_total_time += self.meta_edges[(u, v_)]['time']
            # Log ACSVehicle phase
            pheromone_vals_vei = list(tau_global.values())
            convergence_log.append({
                'v': v,
                'phase': 'ACSVehicle',
                'coverage': vei_coverage,
                'total_time': vei_total_time,
                'count': vei_count,
                'uncovered_stops': len(vei_uncovered),
                'reason_for_switch': 'coverage<99.9' if vei_coverage < 99.9 else 'coverage>=99.9',
                'delta_time': '',
                'delta_coverage': '',
                'improvement_found': '',
                'pheromone_mean': statistics.mean(pheromone_vals_vei) if pheromone_vals_vei else 0.0,
                'pheromone_min': min(pheromone_vals_vei) if pheromone_vals_vei else 0.0,
                'pheromone_max': max(pheromone_vals_vei) if pheromone_vals_vei else 0.0
            })
            if vei_coverage < 99.9:
                v += 1
                continue
            # --- ACS-TIME: For v routes, minimize total time, starting from VEI solution ---
            acs_time = ACSTime(
                self.graph,
                self.meta_edges,
                self.stops,
                self.start_node,
                tau_global,
                self.alpha,
                self.beta,
                self.rho_global,
                self.Q,
                self.opposites,
                self.exit_node
            )
            time_routes, time_count = acs_time.iterate(n_ants, initial_routes=vei_routes)
            time_covered = set()
            total_time = 0.0
            for route in time_routes:
                time_covered.update(stop for stop in route if stop in self.stops)
                for u, v_ in zip(route, route[1:]):
                    if (u, v_) in self.meta_edges:
                        total_time += self.meta_edges[(u, v_)]['time']
            time_coverage = 100.0 * len(time_covered) / max(1, len(self.stops))
            time_uncovered = sorted(set(self.stops) - time_covered)
            # Calculate improvements
            delta_time = total_time - vei_total_time
            delta_coverage = time_coverage - vei_coverage
            improvement_found = (delta_time < 0 or delta_coverage > 0)
            # Log ACSTime phase
            pheromone_vals_time = list(tau_global.values())
            convergence_log.append({
                'v': v,
                'phase': 'ACSTime',
                'coverage': time_coverage,
                'total_time': total_time,
                'count': time_count,
                'uncovered_stops': len(time_uncovered),
                'reason_for_switch': 'ACSTime phase',
                'delta_time': delta_time,
                'delta_coverage': delta_coverage,
                'improvement_found': improvement_found,
                'pheromone_mean': statistics.mean(pheromone_vals_time) if pheromone_vals_time else 0.0,
                'pheromone_min': min(pheromone_vals_time) if pheromone_vals_time else 0.0,
                'pheromone_max': max(pheromone_vals_time) if pheromone_vals_time else 0.0
            })
            # Only update best if this is a full, coherent solution
            if time_coverage > best_coverage or (time_coverage == best_coverage and total_time < best_total_time):
                best_routes = time_routes
                best_count = time_count
                best_total_time = total_time
                best_coverage = time_coverage
                best_uncovered = time_uncovered
                best_solution = (best_routes, best_total_time, best_count, best_coverage, best_uncovered)
                # Update global pheromone matrix based on best solution
            for route in best_routes:
                    for u, v_ in zip(route, route[1:]):
                        if (u, v_) in tau_global:
                            tau_global[(u, v_)] += self.Q / max(1, best_count)
            v += 1
        # --- Save detailed convergence log to CSV ---
        try:
            with open('output/controller_convergence_detailed.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'v', 'phase', 'coverage', 'total_time', 'count', 'uncovered_stops',
                    'reason_for_switch', 'delta_time', 'delta_coverage', 'improvement_found',
                    'pheromone_mean', 'pheromone_min', 'pheromone_max'])
                writer.writeheader()
                for row in convergence_log:
                    writer.writerow(row)
        except Exception as e:
            if verbose:
                print(f"[WARNING] Could not write detailed convergence log: {e}")
        if best_solution is None:
            return [], float('inf'), 0, [], 0.0, []
        best_routes, best_total_time, best_count, best_coverage, best_uncovered = best_solution
        best_route_directions = ["FORWARD"] * best_count
        return best_routes, best_total_time, best_count, best_route_directions, best_coverage, best_uncovered
