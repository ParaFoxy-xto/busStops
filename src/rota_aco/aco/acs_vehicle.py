# src/rota_aco/aco/acs_vehicle.py
"""
Colônia ACS focada na minimização do número de rotas necessárias.
Cada formiga constrói um conjunto de rotas no meta‐grafo até cobrir todas as paradas.
"""
import networkx as nx
from typing import List, Any, Dict, Tuple, Set, Optional
from rota_aco.graph.build_meta import resolver_TSP

# Helper validation function (to be reused in acs_time.py)
def route_has_opposite_violation(route: List[Any], opposites: Dict[Any, List[Any]]) -> bool:
    visited = set(route)
    for stop in route:
        for opp in opposites.get(stop, []):
            if opp in visited:
                return True
    return False

class ACSVehicle:
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops: List[Any],
        start_node: Any,
        pheromone_matrix: Dict[Tuple[Any, Any], float],
        evaporation: float = 0.1,
        Q: float = 1.0,
        opposites: Optional[Dict[Any, List[Any]]] = None,
        exit_node: Any = None
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops = stops
        self.start_node = start_node
        self.tau = pheromone_matrix
        self.rho = evaporation
        self.Q = Q
        if opposites is None:
            opposites = dict()
        self.opposites = opposites
        self.exit_node = exit_node

    def iterate(self, n_ants: int) -> Tuple[List[List[Any]], int]:
        import random
        best_routes: List[List[Any]] = []
        best_count: float = float('inf')
        best_total_dist: float = float('inf')

        for _ in range(n_ants):
            demand = {stop: 10 for stop in self.stops}
            routes = []
            total_dist = 0.0
            
            # Maintain visited set across all routes for this ant
            ant_visited = set([self.start_node])
            if self.exit_node is not None:
                ant_visited.add(self.exit_node)
            
            # Forward routes (start -> exit)
            route_attempts = 0
            max_route_attempts = 50  # Prevent infinite loops
            while sum(demand.values()) > 0 and route_attempts < max_route_attempts:
                route_attempts += 1
                route = [self.start_node]
                visited = set([self.start_node])  # Local visited for this route
                forbidden = set(self.opposites.get(self.start_node, []))
                capacity = 70
                current = self.start_node
                route_length = 0
                max_route_length = 100  # Prevent infinite route construction
                
                while route_length < max_route_length:
                    route_length += 1
                    # If at exit node, end route
                    if self.exit_node is not None and current == self.exit_node:
                        break
                    # If bus is full or no more valid candidates, try to go to exit if possible
                    exit_reachable = self.exit_node is not None and nx.has_path(self.graph, current, self.exit_node)
                    # Only consider candidates with remaining demand and not visited by this ant
                    candidates = [n for n in self.graph.successors(current)
                                  if n not in visited and n not in forbidden and n not in ant_visited and demand.get(n, 0) > 0]
                    # Always allow exit node if reachable
                    if self.exit_node is not None and self.exit_node in self.graph.successors(current):
                        candidates.append(self.exit_node)
                    # Remove bounceback candidates (nodes already in current route)
                    candidates = [n for n in candidates if n not in route]
                    if (capacity == 0 or not candidates) and exit_reachable:
                        try:
                            path_to_exit = nx.shortest_path(self.graph, current, self.exit_node, weight='weight')
                            for n in path_to_exit[1:]:
                                if n in forbidden:
                                    raise Exception('Opposite encountered')
                                route.append(n)
                                visited.add(n)
                                ant_visited.add(n)  # Add to ant's global visited
                                forbidden.update(self.opposites.get(n, []))
                            current = self.exit_node
                        except Exception:
                            break
                        break
                    if not candidates:
                        break
                    # Ant/pheromone logic: increase exit node probability if bus is nearly full
                    probs = []
                    for n in candidates:
                        tau = self.tau.get((current, n), 1e-6)
                        eta = 1.0 / (self.meta_edges.get((current, n), {'length': 1.0})['length'])
                        eta = eta ** 2.0  # Apply beta parameter (default 2.0)
                        if n == self.exit_node and capacity <= 20:
                            tau *= 2.0
                            eta *= 2.0
                        probs.append((n, tau * eta))
                    total_prob = sum(p for _, p in probs)
                    if total_prob == 0:
                        next_node = candidates[0]
                    else:
                        r = random.random()
                        cum = 0.0
                        for n, p in probs:
                            cum += p / total_prob
                            if r <= cum:
                                next_node = n
                                break
                        else:
                            next_node = candidates[-1]
                    route.append(next_node)
                    visited.add(next_node)
                    ant_visited.add(next_node)  # Add to ant's global visited
                    forbidden.update(self.opposites.get(next_node, []))
                    current = next_node
                    if self.exit_node is not None and current == self.exit_node:
                        break
                # Ensure route ends at exit node
                if self.exit_node is not None and (not route or route[-1] != self.exit_node):
                    try:
                        path_to_exit = nx.shortest_path(self.graph, route[-1], self.exit_node, weight='weight')
                        for n in path_to_exit[1:]:
                            if n in forbidden:
                                raise Exception('Opposite encountered')
                            route.append(n)
                            visited.add(n)
                            ant_visited.add(n)  # Add to ant's global visited
                            forbidden.update(self.opposites.get(n, []))
                    except Exception:
                        continue
                if self.exit_node is not None and (not route or route[-1] != self.exit_node):
                    continue
                # Enforce opposites constraint: skip route if it violates opposites
                if route_has_opposite_violation(route, self.opposites):
                    print("[DEBUG] Skipping route due to opposites violation:", route)
                    continue
                # Check if this route actually serves any bus stops (not just start→exit)
                stops_in_route = [stop for stop in route if stop in self.stops]
                if len(stops_in_route) == 0:
                    print("[DEBUG] Skipping route: no bus stops served:", route)
                    continue  # Skip routes that don't serve any bus stops
                # Check if this route serves any new demand (demand that was > 0 before this route)
                if len(stops_in_route) > 0:
                    routes.append(route)
                    # Add distance for this route (only meta-edges)
                    route_dist = 0.0
                    for u, v in zip(route, route[1:]):
                        if (u, v) in self.meta_edges:
                            route_dist += self.meta_edges[(u, v)]['length']
                    total_dist += route_dist
                    # Consume demand for stops in this route (only after route is accepted)
                    capacity = 70
                    for stop in route:
                        if stop in demand and demand[stop] > 0 and capacity > 0:
                            picked = min(10, demand[stop], capacity)
                            demand[stop] -= picked
                            capacity -= picked
                    # Debug: show route details
                    print(f"[DEBUG] Route accepted: {route}")
                    print(f"[DEBUG] Stops in route with demand: {[stop for stop in route if stop in demand and demand[stop] > 0]}")
                    print(f"[DEBUG] Route serves stops with demand: {len([stop for stop in route if stop in demand and demand[stop] > 0]) > 0}")
                # After route construction, print remaining demand
                print("[DEBUG] Remaining demand after route:", {k: v for k, v in demand.items() if v > 0})
            
            # Reverse routes (exit -> start) to cover remaining demand
            if sum(demand.values()) > 0 and self.exit_node is not None:
                route_attempts = 0
                while sum(demand.values()) > 0 and route_attempts < max_route_attempts:
                    route_attempts += 1
                    route = [self.exit_node]
                    visited = set([self.exit_node])  # Local visited for this route
                    forbidden = set(self.opposites.get(self.exit_node, []))
                    capacity = 70
                    current = self.exit_node
                    route_length = 0
                    
                    while route_length < max_route_length:
                        route_length += 1
                        # Pick up passengers at current stop
                        if current in demand and demand[current] > 0 and capacity > 0:
                            picked = min(10, demand[current], capacity)
                            demand[current] -= picked
                            capacity -= picked
                        # If at start node, end route
                        if current == self.start_node:
                            break
                        # If bus is full or no more valid candidates, try to go to start if possible
                        start_reachable = nx.has_path(self.graph, current, self.start_node)
                        # Only consider candidates with remaining demand and not visited by this ant
                        candidates = [n for n in self.graph.predecessors(current)
                                      if n not in visited and n not in forbidden and n not in ant_visited and demand.get(n, 0) > 0]
                        # Always allow start node if reachable
                        if self.start_node in self.graph.predecessors(current):
                            candidates.append(self.start_node)
                        # Remove bounceback candidates (nodes already in current route)
                        candidates = [n for n in candidates if n not in route]
                        if (capacity == 0 or not candidates) and start_reachable:
                            try:
                                path_to_start = nx.shortest_path(self.graph, current, self.start_node, weight='weight')
                                for n in path_to_start[1:]:
                                    if n in forbidden:
                                        raise Exception('Opposite encountered')
                                    route.append(n)
                                    visited.add(n)
                                    ant_visited.add(n)  # Add to ant's global visited
                                    forbidden.update(self.opposites.get(n, []))
                                current = self.start_node
                            except Exception:
                                break
                            break
                        if not candidates:
                            break
                        # Ant/pheromone logic for reverse direction
                        probs = []
                        for n in candidates:
                            # For reverse routes, use the reverse edge for pheromone
                            tau = self.tau.get((n, current), 1e-6)
                            eta = 1.0 / (self.meta_edges.get((n, current), {'length': 1.0})['length'])
                            eta = eta ** 2.0  # Apply beta parameter
                            if n == self.start_node and capacity <= 20:
                                tau *= 2.0
                                eta *= 2.0
                            probs.append((n, tau * eta))
                        total_prob = sum(p for _, p in probs)
                        if total_prob == 0:
                            next_node = candidates[0]
                        else:
                            r = random.random()
                            cum = 0.0
                            for n, p in probs:
                                cum += p / total_prob
                                if r <= cum:
                                    next_node = n
                                    break
                            else:
                                next_node = candidates[-1]
                        route.append(next_node)
                        visited.add(next_node)
                        ant_visited.add(next_node)  # Add to ant's global visited
                        forbidden.update(self.opposites.get(next_node, []))
                        current = next_node
                        if current == self.start_node:
                            break
                    # Ensure route ends at start node
                    if not route or route[-1] != self.start_node:
                        try:
                            path_to_start = nx.shortest_path(self.graph, route[-1], self.start_node, weight='weight')
                            for n in path_to_start[1:]:
                                if n in forbidden:
                                    raise Exception('Opposite encountered')
                                route.append(n)
                                visited.add(n)
                                ant_visited.add(n)  # Add to ant's global visited
                                forbidden.update(self.opposites.get(n, []))
                        except Exception:
                            continue
                    if not route or route[-1] != self.start_node:
                        continue
                    # Enforce opposites constraint: skip route if it violates opposites
                    if route_has_opposite_violation(route, self.opposites):
                        continue
                    
                    # Check if this route actually serves any bus stops (not just exit→start)
                    stops_in_route = [stop for stop in route if stop in self.stops]
                    if len(stops_in_route) == 0:
                        continue  # Skip routes that don't serve any bus stops
                    
                    # Check if this route serves any new demand
                    if len(stops_in_route) > 0:
                        routes.append(route)
                        # Add distance for this route (only meta-edges)
                        route_dist = 0.0
                        for u, v in zip(route, route[1:]):
                            if (u, v) in self.meta_edges:
                                route_dist += self.meta_edges[(u, v)]['length']
                        total_dist += route_dist
                # After the outer route loop, print number of routes built
                print(f"[DEBUG] Number of routes built in this ant: {len(routes)}")
            
            # Only accept solutions that cover all demand
            if any(d > 0 for d in demand.values()):
                continue
            count = len(routes)
            if (count < best_count) or (count == best_count and total_dist < best_total_dist):
                best_count = count
                best_routes = routes
                best_total_dist = total_dist

        # evaporação global
        for edge in list(self.tau.keys()):
            self.tau[edge] *= (1 - self.rho)

        # reforço elitista: mais feromônio nas arestas usadas (only meta-edges)
        for route in best_routes:
            for u, v in zip(route, route[1:]):
                if (u, v) in self.meta_edges:
                    self.tau[(u, v)] = self.tau.get((u, v), 0.0) + (self.Q / best_count)

        if best_count == float('inf'):
            return [], 0
        return best_routes, int(best_count)

    def get_pheromone(self) -> Dict[Tuple[Any, Any], float]:
        return self.tau
