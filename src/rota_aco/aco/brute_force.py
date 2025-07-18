# src/rota_aco/aco/brute_force.py

"""
Implementação de força bruta para otimização de rotas.

Esta implementação serve como baseline para comparação com algoritmos
heurísticos como ACO, testando todas as combinações possíveis de rotas.
"""

import networkx as nx
import itertools
from typing import List, Any, Dict, Tuple, Optional
import time


class BruteForceOptimizer:
    """
    Otimizador de rotas por força bruta.
    
    Testa todas as permutações possíveis dos pontos de interesse
    para encontrar a rota ótima.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops_to_visit: List[Any],
        start_node: Any,
        exit_node: Any,
        max_permutations: int = 10000  # Limite para evitar explosão combinatória
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops_to_visit = list(stops_to_visit)
        self.start_node = start_node
        self.exit_node = exit_node
        self.max_permutations = max_permutations
        
        # Estatísticas
        self.routes_tested = 0
        self.valid_routes = 0
        self.execution_time = 0.0
    
    def run(self, verbose: bool = False) -> Tuple[List[Any], float, Dict]:
        """
        Executa a otimização por força bruta.
        
        Returns:
            Tuple contendo: (melhor_rota, melhor_distancia, estatisticas)
        """
        start_time = time.time()
        
        best_route = []
        best_distance = float('inf')
        
        # Gerar todas as permutações dos pontos de interesse
        if len(self.stops_to_visit) > 10:
            if verbose:
                print(f"Aviso: {len(self.stops_to_visit)} pontos podem gerar muitas permutações.")
                print(f"Limitando a {self.max_permutations} permutações.")
        
        permutations = itertools.permutations(self.stops_to_visit)
        
        # Testar cada permutação
        for i, perm in enumerate(permutations):
            if i >= self.max_permutations:
                if verbose:
                    print(f"Limite de {self.max_permutations} permutações atingido.")
                break
            
            self.routes_tested += 1
            
            # Construir rota completa
            route = self._build_complete_route(list(perm))
            
            if route:
                self.valid_routes += 1
                distance = self._calculate_route_distance(route)
                
                if distance < best_distance:
                    best_route = route.copy()
                    best_distance = distance
                    
                    if verbose and self.routes_tested % 1000 == 0:
                        print(f"Testadas {self.routes_tested} rotas. Melhor: {best_distance:.2f}")
        
        self.execution_time = time.time() - start_time
        
        # Calcular estatísticas finais
        coverage = self._calculate_coverage(best_route)
        stats = {
            'total_distance': best_distance,
            'coverage': coverage,
            'route_length': len(best_route),
            'routes_tested': self.routes_tested,
            'valid_routes': self.valid_routes,
            'execution_time': self.execution_time,
            'success_rate': self.valid_routes / max(1, self.routes_tested)
        }
        
        if verbose:
            print(f"\nForça bruta concluída:")
            print(f"- Rotas testadas: {self.routes_tested}")
            print(f"- Rotas válidas: {self.valid_routes}")
            print(f"- Tempo de execução: {self.execution_time:.2f}s")
            print(f"- Melhor distância: {best_distance:.2f}")
        
        return best_route, best_distance, stats
    
    def _build_complete_route(self, stop_sequence: List[Any]) -> Optional[List[Any]]:
        """
        Constrói uma rota completa conectando os pontos na sequência dada.
        """
        route = [self.start_node]
        current_node = self.start_node
        
        # Conectar cada ponto na sequência
        for target_stop in stop_sequence:
            path = self._find_shortest_path(current_node, target_stop)
            if not path:
                return None  # Não é possível conectar
            
            # Adicionar caminho (excluindo o nó atual para evitar duplicação)
            route.extend(path[1:])
            current_node = target_stop
        
        # Conectar ao nó de saída
        final_path = self._find_shortest_path(current_node, self.exit_node)
        if not final_path:
            return None
        
        route.extend(final_path[1:])
        
        return route
    
    def _find_shortest_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Encontra o caminho mais curto entre dois nós usando Dijkstra.
        """
        try:
            # Verificar se os nós existem no grafo
            if start not in self.graph.nodes() or end not in self.graph.nodes():
                return None
            
            # Implementação alternativa mais robusta usando dijkstra_path
            try:
                path = nx.dijkstra_path(
                    self.graph, 
                    start, 
                    end, 
                    weight=lambda u, v, d: self.meta_edges.get((u, v), {}).get('time', 1.0)
                )
                return path
            except:
                # Fallback para shortest_path simples
                try:
                    path = nx.shortest_path(self.graph, start, end)
                    return path
                except:
                    return None
                    
        except Exception as e:
            # Tratamento mais robusto de erros
            return None
    
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
    
    def run_limited(self, max_time_seconds: float = 60.0, verbose: bool = False) -> Tuple[List[Any], float, Dict]:
        """
        Executa força bruta com limite de tempo.
        
        Args:
            max_time_seconds: Tempo máximo de execução em segundos
            verbose: Se deve imprimir informações de progresso
            
        Returns:
            Tuple contendo: (melhor_rota, melhor_distancia, estatisticas)
        """
        start_time = time.time()
        
        best_route = []
        best_distance = float('inf')
        
        permutations = itertools.permutations(self.stops_to_visit)
        
        for perm in permutations:
            # Verificar limite de tempo
            if time.time() - start_time > max_time_seconds:
                if verbose:
                    print(f"Limite de tempo ({max_time_seconds}s) atingido.")
                break
            
            self.routes_tested += 1
            
            route = self._build_complete_route(list(perm))
            
            if route:
                self.valid_routes += 1
                distance = self._calculate_route_distance(route)
                
                if distance < best_distance:
                    best_route = route.copy()
                    best_distance = distance
                    
                    if verbose and self.routes_tested % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Tempo: {elapsed:.1f}s, Testadas: {self.routes_tested}, Melhor: {best_distance:.2f}")
        
        self.execution_time = time.time() - start_time
        
        # Calcular estatísticas finais
        coverage = self._calculate_coverage(best_route)
        stats = {
            'total_distance': best_distance,
            'coverage': coverage,
            'route_length': len(best_route),
            'routes_tested': self.routes_tested,
            'valid_routes': self.valid_routes,
            'execution_time': self.execution_time,
            'success_rate': self.valid_routes / max(1, self.routes_tested),
            'time_limited': True
        }
        
        return best_route, best_distance, stats


class GreedyOptimizer:
    """
    Implementação gulosa simples como alternativa mais rápida à força bruta.
    
    Sempre escolhe o próximo ponto mais próximo que ainda não foi visitado.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        meta_edges: Dict[Tuple[Any, Any], dict],
        stops_to_visit: List[Any],
        start_node: Any,
        exit_node: Any
    ):
        self.graph = graph
        self.meta_edges = meta_edges
        self.stops_to_visit = set(stops_to_visit)
        self.start_node = start_node
        self.exit_node = exit_node
    
    def run(self, verbose: bool = False) -> Tuple[List[Any], float, Dict]:
        """
        Executa a otimização gulosa.
        """
        start_time = time.time()
        
        route = [self.start_node]
        current_node = self.start_node
        remaining_stops = self.stops_to_visit.copy()
        
        # Visitar pontos usando estratégia gulosa
        while remaining_stops:
            # Encontrar o ponto mais próximo
            closest_stop = None
            min_distance = float('inf')
            
            for stop in remaining_stops:
                try:
                    distance = nx.shortest_path_length(
                        self.graph,
                        current_node,
                        stop,
                        weight=lambda u, v, d: self.meta_edges.get((u, v), {}).get('time', 1.0)
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_stop = stop
                except nx.NetworkXNoPath:
                    continue
            
            if closest_stop is None:
                break  # Não é possível alcançar nenhum ponto restante
            
            # Adicionar caminho para o ponto mais próximo
            try:
                path = nx.shortest_path(
                    self.graph,
                    current_node,
                    closest_stop,
                    weight=lambda u, v, d: self.meta_edges.get((u, v), {}).get('time', 1.0)
                )
                route.extend(path[1:])  # Excluir nó atual
                current_node = closest_stop
                remaining_stops.remove(closest_stop)
            except nx.NetworkXNoPath:
                break
        
        # Conectar ao nó de saída
        try:
            final_path = nx.shortest_path(
                self.graph,
                current_node,
                self.exit_node,
                weight=lambda u, v, d: self.meta_edges.get((u, v), {}).get('time', 1.0)
            )
            route.extend(final_path[1:])
        except nx.NetworkXNoPath:
            pass  # Não é possível chegar ao nó de saída
        
        execution_time = time.time() - start_time
        total_distance = self._calculate_route_distance(route)
        coverage = self._calculate_coverage(route)
        
        stats = {
            'total_distance': total_distance,
            'coverage': coverage,
            'route_length': len(route),
            'execution_time': execution_time,
            'algorithm': 'greedy'
        }
        
        if verbose:
            print(f"Algoritmo guloso concluído:")
            print(f"- Tempo de execução: {execution_time:.4f}s")
            print(f"- Distância total: {total_distance:.2f}")
            print(f"- Cobertura: {coverage:.2%}")
        
        return route, total_distance, stats
    
    def _calculate_route_distance(self, route: List[Any]) -> float:
        """Calcula a distância total de uma rota."""
        total_distance = 0.0
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            edge_data = self.meta_edges.get(edge, {})
            distance = edge_data.get('time', 0.0)
            total_distance += distance
        return total_distance
    
    def _calculate_coverage(self, route: List[Any]) -> float:
        """Calcula a cobertura dos pontos de interesse."""
        if not self.stops_to_visit:
            return 1.0
        visited_stops = set(node for node in route if node in self.stops_to_visit)
        return len(visited_stops) / len(self.stops_to_visit)