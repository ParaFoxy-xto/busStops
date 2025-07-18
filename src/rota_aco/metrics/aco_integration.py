"""
Integração do sistema de métricas com algoritmos ACO existentes.

Este módulo fornece adaptadores e wrappers para integrar a coleta de dados
com o código ACO existente de forma não-intrusiva.
"""

import time
from typing import List, Dict, Any, Tuple, Optional, Callable
import networkx as nx

from .data_collector import DataCollector
from .data_models import Route, Solution, IterationData
from .config import MetricsConfig


class ACOMetricsIntegrator:
    """
    Integrador principal para coleta de métricas em algoritmos ACO.
    
    Esta classe atua como uma ponte entre o sistema de métricas e os
    algoritmos ACO existentes, permitindo coleta de dados sem modificar
    o código original.
    """
    
    def __init__(self, config: MetricsConfig = None):
        """
        Inicializa o integrador.
        
        Args:
            config: Configuração do sistema de métricas
        """
        self.config = config or MetricsConfig()
        self.collector = DataCollector(self.config)
        self.current_execution_id: Optional[str] = None
        
        # Cache para conversão de dados
        self._route_cache: Dict[str, Route] = {}
        self._solution_cache: Dict[str, Solution] = {}
    
    def wrap_controller_execution(self,
                                 controller,
                                 algorithm_params: Dict[str, Any],
                                 problem_params: Dict[str, Any],
                                 quality_weights: Dict[str, Any]) -> Callable:
        """
        Cria um wrapper para execução do ACSController com coleta de métricas.
        
        Args:
            controller: Instância do ACSController
            algorithm_params: Parâmetros do algoritmo ACO
            problem_params: Parâmetros do problema
            quality_weights: Pesos da função de qualidade
            
        Returns:
            Callable: Função wrapper para execução
        """
        def wrapped_run(n_ants: int, n_iterations: int, verbose: bool = False):
            # Preparar configuração para coleta
            config = {
                'algorithm_params': algorithm_params,
                'problem_params': problem_params,
                'quality_weights': quality_weights,
                'n_ants': n_ants,
                'n_iterations': n_iterations
            }
            
            # Determinar tipo de algoritmo baseado nos pesos
            algorithm_type = self._determine_algorithm_type(quality_weights)
            
            # Iniciar coleta de dados
            self.current_execution_id = self.collector.start_execution(
                algorithm_type=algorithm_type,
                config=config
            )
            
            start_time = time.time()
            
            try:
                # Interceptar o método run original
                original_run = controller.run
                controller.run = self._create_instrumented_run(
                    original_run, controller, n_ants, n_iterations, verbose
                )
                
                # Executar algoritmo original
                result = original_run(n_ants, n_iterations, quality_weights, verbose)
                
                # Processar resultado
                execution_time = time.time() - start_time
                solution = self._convert_result_to_solution(result, controller)
                
                # Registrar solução final
                self.collector.record_final_solution(
                    solution=solution,
                    execution_time=execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Registrar falha
                execution_time = time.time() - start_time
                self.collector.record_final_solution(
                    solution=None,
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e)
                )
                raise
            
            finally:
                # Finalizar coleta
                self.collector.finish_execution()
                self.current_execution_id = None
        
        return wrapped_run
    
    def _create_instrumented_run(self, 
                                original_run: Callable,
                                controller,
                                n_ants: int,
                                n_iterations: int,
                                verbose: bool) -> Callable:
        """
        Cria versão instrumentada do método run que coleta dados de iterações.
        
        Args:
            original_run: Método run original
            controller: Instância do controller
            n_ants: Número de formigas
            n_iterations: Número de iterações
            verbose: Se deve imprimir logs
            
        Returns:
            Callable: Método run instrumentado
        """
        def instrumented_run(n_ants_param, n_iterations_param, quality_weights, verbose_param):
            # Interceptar histórico do controller
            original_record_history = controller._record_history
            controller._record_history = self._create_history_interceptor(
                original_record_history, controller
            )
            
            try:
                return original_run(n_ants_param, n_iterations_param, quality_weights, verbose_param)
            finally:
                # Restaurar método original
                controller._record_history = original_record_history
        
        return instrumented_run
    
    def _create_history_interceptor(self, 
                                   original_record_history: Callable,
                                   controller) -> Callable:
        """
        Cria interceptador para o método _record_history do controller.
        
        Args:
            original_record_history: Método original _record_history
            controller: Instância do controller
            
        Returns:
            Callable: Interceptador instrumentado
        """
        def history_interceptor(iteration_num: int,
                               time_solution: List[List[Any]],
                               vehicle_solution: List[List[Any]],
                               chosen_solution_quality: float,
                               best_overall_quality: float):
            
            # Chamar método original
            original_record_history(
                iteration_num, time_solution, vehicle_solution,
                chosen_solution_quality, best_overall_quality
            )
            
            # Coletar dados para métricas
            try:
                self._record_iteration_metrics(
                    controller, iteration_num, time_solution, vehicle_solution,
                    chosen_solution_quality, best_overall_quality
                )
            except Exception as e:
                # Log error but don't interrupt execution
                if hasattr(controller, 'verbose') and controller.verbose:
                    print(f"Warning: Failed to record iteration metrics: {e}")
        
        return history_interceptor
    
    def _record_iteration_metrics(self,
                                 controller,
                                 iteration_num: int,
                                 time_solution: List[List[Any]],
                                 vehicle_solution: List[List[Any]],
                                 chosen_solution_quality: float,
                                 best_overall_quality: float):
        """
        Registra métricas de uma iteração específica.
        
        Args:
            controller: Instância do controller
            iteration_num: Número da iteração
            time_solution: Solução da colônia ACS-TIME
            vehicle_solution: Solução da colônia ACS-VEHICLE
            chosen_solution_quality: Qualidade da solução escolhida
            best_overall_quality: Melhor qualidade até agora
        """
        # Converter soluções para nosso formato
        time_sol = self._convert_routes_to_solution(time_solution, controller, "ACS-TIME")
        vehicle_sol = self._convert_routes_to_solution(vehicle_solution, controller, "ACS-VEHICLE")
        
        # Determinar qual solução foi escolhida
        time_quality = controller._calculate_solution_quality(
            time_solution, 
            controller.history[-1]['time_metrics']['dist'] if controller.history else 0,
            controller.history[-1]['time_metrics']['count'] if controller.history else 0,
            controller.history[-1]['time_metrics']['coverage'] if controller.history else 0
        ) if controller.history else chosen_solution_quality
        
        best_solution = time_sol if time_quality >= chosen_solution_quality else vehicle_sol
        
        # Calcular variância da população (simplificado)
        population_variance = abs(time_quality - chosen_solution_quality) if time_quality != chosen_solution_quality else 0.0
        
        # Registrar dados da iteração
        self.collector.record_iteration(
            iteration=iteration_num,
            best_fitness=best_overall_quality,
            avg_fitness=(time_quality + chosen_solution_quality) / 2,
            population_variance=population_variance,
            best_solution=best_solution,
            additional_metrics={
                'time_solution_quality': time_quality,
                'vehicle_solution_quality': chosen_solution_quality,
                'chosen_solution_quality': chosen_solution_quality,
                'time_solution_routes': len(time_solution),
                'vehicle_solution_routes': len(vehicle_solution)
            }
        )
    
    def _convert_routes_to_solution(self, 
                                   routes: List[List[Any]], 
                                   controller,
                                   algorithm_type: str) -> Solution:
        """
        Converte rotas do formato ACO para Solution.
        
        Args:
            routes: Lista de rotas no formato ACO
            controller: Instância do controller
            algorithm_type: Tipo do algoritmo
            
        Returns:
            Solution: Solução convertida
        """
        if not routes:
            return Solution(
                routes=[],
                total_vehicles=0,
                total_distance=0.0,
                total_passengers_served=0,
                fitness_time=float('inf'),
                fitness_vehicle=float('inf'),
                is_feasible=False
            )
        
        # Converter cada rota
        solution_routes = []
        total_distance = 0.0
        total_passengers = 0
        
        for route in routes:
            converted_route = self._convert_single_route(route, controller)
            solution_routes.append(converted_route)
            total_distance += converted_route.total_distance
            total_passengers += converted_route.total_passengers
        
        # Calcular fitness
        fitness_time = total_distance
        fitness_vehicle = len(routes) * 1000 + total_distance  # Penaliza mais veículos
        
        return Solution(
            routes=solution_routes,
            total_vehicles=len(routes),
            total_distance=total_distance,
            total_passengers_served=total_passengers,
            fitness_time=fitness_time,
            fitness_vehicle=fitness_vehicle,
            is_feasible=all(route.is_valid for route in solution_routes)
        )
    
    def _convert_single_route(self, route: List[Any], controller) -> Route:
        """
        Converte uma rota individual do formato ACO para Route.
        
        Args:
            route: Rota no formato ACO
            controller: Instância do controller
            
        Returns:
            Route: Rota convertida
        """
        # Cache key para evitar reconversões
        route_key = str(route)
        if route_key in self._route_cache:
            return self._route_cache[route_key]
        
        if len(route) < 2:
            # Rota inválida
            converted_route = Route(
                stops=route,
                distances=[],
                passenger_load=[],
                total_distance=0.0,
                total_passengers=0,
                is_valid=False,
                capacity_violations=1,
                opposite_stops_violations=0
            )
        else:
            # Calcular distâncias entre paradas consecutivas
            distances = []
            for i in range(len(route) - 1):
                edge_data = controller.meta_edges.get((route[i], route[i+1]), {})
                distance = edge_data.get('time', 0.0)  # Usar 'time' como distância
                distances.append(distance)
            
            # Simular carga de passageiros (10 por parada)
            passenger_load = []
            total_passengers = 0
            for i in range(len(route) - 1):
                if route[i] in controller.stops_to_visit:
                    load = 10  # Demanda fixa por parada
                    passenger_load.append(load)
                    total_passengers += load
                else:
                    passenger_load.append(0)
            
            # Verificar violações
            capacity_violations = 1 if total_passengers > controller.capacity else 0
            
            # Verificar violações de paradas opostas
            opposite_violations = 0
            for stop in route:
                if stop in controller.opposites:
                    for opposite in controller.opposites[stop]:
                        if opposite in route:
                            opposite_violations += 1
                            break
            
            converted_route = Route(
                stops=route,
                distances=distances,
                passenger_load=passenger_load,
                total_distance=sum(distances),
                total_passengers=total_passengers,
                is_valid=(capacity_violations == 0 and opposite_violations == 0),
                capacity_violations=capacity_violations,
                opposite_stops_violations=opposite_violations
            )
        
        # Cache da rota convertida
        self._route_cache[route_key] = converted_route
        return converted_route
    
    def _convert_result_to_solution(self, 
                                   result: Tuple,
                                   controller) -> Solution:
        """
        Converte resultado final do ACO para Solution.
        
        Args:
            result: Resultado do método run do controller
            controller: Instância do controller
            
        Returns:
            Solution: Solução final convertida
        """
        if not result or len(result) < 4:
            return Solution(
                routes=[],
                total_vehicles=0,
                total_distance=0.0,
                total_passengers_served=0,
                fitness_time=float('inf'),
                fitness_vehicle=float('inf'),
                is_feasible=False
            )
        
        best_routes, total_dist, num_routes, coverage = result[:4]
        
        return self._convert_routes_to_solution(best_routes, controller, "FINAL")
    
    def _determine_algorithm_type(self, quality_weights: Dict[str, Any]) -> str:
        """
        Determina o tipo de algoritmo baseado nos pesos da função de qualidade.
        
        Args:
            quality_weights: Pesos da função de qualidade
            
        Returns:
            str: Tipo do algoritmo ('ACS-TIME' ou 'ACS-VEHICLE')
        """
        w_r = quality_weights.get('w_r', 1.0)  # Peso do número de rotas
        w_d = quality_weights.get('w_d', 0.5)  # Peso da distância
        
        # Se o peso das rotas é maior, foca em minimizar veículos
        if w_r > w_d:
            return "ACS-VEHICLE"
        else:
            return "ACS-TIME"
    
    def get_collected_data(self) -> List:
        """
        Retorna todos os dados coletados.
        
        Returns:
            List: Lista de ExecutionData coletados
        """
        return self.collector.execution_history
    
    def save_all_data(self):
        """Salva todos os dados coletados em disco."""
        for execution in self.collector.execution_history:
            self.collector.save_execution_data(execution)
    
    def clear_cache(self):
        """Limpa cache de conversões."""
        self._route_cache.clear()
        self._solution_cache.clear()


def create_metrics_enabled_controller(controller_class,
                                     graph: nx.DiGraph,
                                     meta_edges: Dict,
                                     stops_to_visit: List[Any],
                                     start_node: Any,
                                     exit_node: Any,
                                     opposites: Dict,
                                     aco_params: Dict[str, Any],
                                     problem_params: Dict[str, Any],
                                     metrics_config: MetricsConfig = None):
    """
    Cria um controller ACO com coleta de métricas habilitada.
    
    Args:
        controller_class: Classe do controller (ACSController)
        graph: Grafo do problema
        meta_edges: Arestas do meta-grafo
        stops_to_visit: Paradas a serem visitadas
        start_node: Nó de início
        exit_node: Nó de saída
        opposites: Mapeamento de paradas opostas
        aco_params: Parâmetros do ACO
        problem_params: Parâmetros do problema
        metrics_config: Configuração das métricas
        
    Returns:
        Tuple: (controller_instance, metrics_integrator)
    """
    # Criar controller original
    controller = controller_class(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node,
        opposites=opposites,
        aco_params=aco_params,
        problem_params=problem_params
    )
    
    # Criar integrador de métricas
    integrator = ACOMetricsIntegrator(metrics_config)
    
    return controller, integrator


# Função de conveniência para uso direto
def run_aco_with_metrics(controller_class,
                        graph: nx.DiGraph,
                        meta_edges: Dict,
                        stops_to_visit: List[Any],
                        start_node: Any,
                        exit_node: Any,
                        opposites: Dict,
                        aco_params: Dict[str, Any],
                        problem_params: Dict[str, Any],
                        quality_weights: Dict[str, Any],
                        n_ants: int,
                        n_iterations: int,
                        verbose: bool = False,
                        metrics_config: MetricsConfig = None):
    """
    Executa algoritmo ACO com coleta de métricas completa.
    
    Args:
        controller_class: Classe do controller
        graph: Grafo do problema
        meta_edges: Arestas do meta-grafo
        stops_to_visit: Paradas a serem visitadas
        start_node: Nó de início
        exit_node: Nó de saída
        opposites: Mapeamento de paradas opostas
        aco_params: Parâmetros do ACO
        problem_params: Parâmetros do problema
        quality_weights: Pesos da função de qualidade
        n_ants: Número de formigas
        n_iterations: Número de iterações
        verbose: Se deve imprimir logs
        metrics_config: Configuração das métricas
        
    Returns:
        Tuple: (result, execution_data)
    """
    # Criar controller com métricas
    controller, integrator = create_metrics_enabled_controller(
        controller_class, graph, meta_edges, stops_to_visit,
        start_node, exit_node, opposites, aco_params, problem_params,
        metrics_config
    )
    
    # Criar wrapper para execução
    wrapped_run = integrator.wrap_controller_execution(
        controller, aco_params, problem_params, quality_weights
    )
    
    # Executar com coleta de métricas
    result = wrapped_run(n_ants, n_iterations, verbose)
    
    # Retornar resultado e dados coletados
    execution_data = integrator.get_collected_data()[-1] if integrator.get_collected_data() else None
    
    return result, execution_data