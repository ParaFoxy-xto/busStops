"""
Coletor de dados para execuções ACO.

Este módulo implementa a coleta não-intrusiva de dados durante execuções
dos algoritmos ACO, capturando informações de iterações, soluções e métricas
para posterior análise.
"""

import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import uuid

from .data_models import (
    ExecutionData, IterationData, Route, Solution,
    RouteQualityMetrics, ConvergenceMetrics
)
from .config import MetricsConfig
from .exceptions import DataCollectionError, FileOperationError


class DataCollector:
    """
    Coleta dados de execuções ACO de forma não-intrusiva.
    
    Esta classe atua como um interceptador que captura dados durante
    a execução dos algoritmos ACO sem interferir em sua lógica.
    """
    
    def __init__(self, config: MetricsConfig = None):
        """
        Inicializa o coletor de dados.
        
        Args:
            config: Configuração do sistema de métricas
        """
        self.config = config or MetricsConfig()
        self.current_execution: Optional[ExecutionData] = None
        self.execution_history: List[ExecutionData] = []
        
        # Callbacks para eventos específicos
        self.iteration_callbacks: List[Callable] = []
        self.solution_callbacks: List[Callable] = []
        
        # Ensure output directories exist
        self.config._ensure_directories()
    
    def start_execution(self, 
                       algorithm_type: str,
                       config: Dict[str, Any],
                       execution_id: str = None) -> str:
        """
        Inicia a coleta de dados para uma nova execução.
        
        Args:
            algorithm_type: Tipo do algoritmo ('ACS-TIME' ou 'ACS-VEHICLE')
            config: Configuração utilizada na execução
            execution_id: ID único da execução (gerado automaticamente se None)
            
        Returns:
            str: ID da execução iniciada
            
        Raises:
            DataCollectionError: Se já existe uma execução em andamento
        """
        if self.current_execution is not None:
            raise DataCollectionError(
                "Já existe uma execução em andamento",
                execution_id=self.current_execution.execution_id,
                details="Finalize a execução atual antes de iniciar uma nova"
            )
        
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        self.current_execution = ExecutionData(
            execution_id=execution_id,
            algorithm_type=algorithm_type,
            config=config.copy(),
            timestamp=datetime.now()
        )
        
        return execution_id
    
    def record_iteration(self,
                        iteration: int,
                        best_fitness: float,
                        avg_fitness: float,
                        population_variance: float,
                        best_solution: Solution,
                        additional_metrics: Dict[str, Any] = None):
        """
        Registra dados de uma iteração específica.
        
        Args:
            iteration: Número da iteração
            best_fitness: Melhor fitness da iteração
            avg_fitness: Fitness médio da população
            population_variance: Variância da população
            best_solution: Melhor solução encontrada na iteração
            additional_metrics: Métricas adicionais específicas do algoritmo
        """
        if self.current_execution is None:
            raise DataCollectionError(
                "Nenhuma execução em andamento",
                details="Chame start_execution() antes de registrar iterações"
            )
        
        iteration_data = IterationData(
            iteration=iteration,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            population_variance=population_variance,
            best_solution=best_solution,
            additional_metrics=additional_metrics or {}
        )
        
        self.current_execution.iterations_data.append(iteration_data)
        
        # Trigger callbacks
        for callback in self.iteration_callbacks:
            try:
                callback(iteration_data)
            except Exception as e:
                # Log error but don't stop execution
                print(f"Warning: Iteration callback failed: {e}")
    
    def record_final_solution(self, 
                             solution: Solution,
                             execution_time: float,
                             success: bool = True,
                             error_message: str = None):
        """
        Registra a solução final da execução.
        
        Args:
            solution: Solução final encontrada
            execution_time: Tempo total de execução em segundos
            success: Se a execução foi bem-sucedida
            error_message: Mensagem de erro (se success=False)
        """
        if self.current_execution is None:
            raise DataCollectionError(
                "Nenhuma execução em andamento",
                details="Chame start_execution() antes de registrar a solução final"
            )
        
        self.current_execution.final_solution = solution
        self.current_execution.routes = solution.routes if solution else []
        self.current_execution.execution_time = execution_time
        self.current_execution.success = success
        self.current_execution.error_message = error_message
        
        # Trigger callbacks
        for callback in self.solution_callbacks:
            try:
                callback(solution)
            except Exception as e:
                print(f"Warning: Solution callback failed: {e}")
    
    def finish_execution(self, save_to_disk: bool = True) -> ExecutionData:
        """
        Finaliza a execução atual e opcionalmente salva os dados.
        
        Args:
            save_to_disk: Se deve salvar os dados em disco
            
        Returns:
            ExecutionData: Dados da execução finalizada
            
        Raises:
            DataCollectionError: Se não há execução em andamento
        """
        if self.current_execution is None:
            raise DataCollectionError(
                "Nenhuma execução em andamento",
                details="Chame start_execution() antes de finalizar"
            )
        
        execution_data = self.current_execution
        self.execution_history.append(execution_data)
        self.current_execution = None
        
        if save_to_disk:
            self.save_execution_data(execution_data)
        
        return execution_data
    
    def save_execution_data(self, execution_data: ExecutionData):
        """
        Salva dados de execução em disco.
        
        Args:
            execution_data: Dados da execução a serem salvos
            
        Raises:
            FileOperationError: Se houver erro na operação de arquivo
        """
        try:
            # Save as JSON (human readable)
            json_path = os.path.join(
                self.config.get_execution_data_path(),
                f"{execution_data.execution_id}.json"
            )
            
            # Convert to serializable format
            data_dict = self._execution_to_dict(execution_data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)
            
            # Save as pickle (for exact object reconstruction)
            pickle_path = os.path.join(
                self.config.get_execution_data_path(),
                f"{execution_data.execution_id}.pkl"
            )
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(execution_data, f)
                
        except Exception as e:
            raise FileOperationError(
                f"Erro ao salvar dados de execução",
                file_path=json_path,
                operation="write",
                details=str(e)
            )
    
    def load_execution_data(self, execution_id: str) -> ExecutionData:
        """
        Carrega dados de execução do disco.
        
        Args:
            execution_id: ID da execução a ser carregada
            
        Returns:
            ExecutionData: Dados da execução carregada
            
        Raises:
            FileOperationError: Se houver erro na operação de arquivo
        """
        pickle_path = os.path.join(
            self.config.get_execution_data_path(),
            f"{execution_id}.pkl"
        )
        
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            # Try JSON fallback
            json_path = os.path.join(
                self.config.get_execution_data_path(),
                f"{execution_id}.json"
            )
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                return self._dict_to_execution(data_dict)
            except FileNotFoundError:
                raise FileOperationError(
                    f"Dados de execução não encontrados",
                    file_path=pickle_path,
                    operation="read",
                    details=f"Tentou carregar {execution_id} mas arquivo não existe"
                )
        except Exception as e:
            raise FileOperationError(
                f"Erro ao carregar dados de execução",
                file_path=pickle_path,
                operation="read",
                details=str(e)
            )
    
    def list_executions(self) -> List[str]:
        """
        Lista IDs de todas as execuções salvas em disco.
        
        Returns:
            List[str]: Lista de IDs de execução
        """
        execution_dir = self.config.get_execution_data_path()
        
        if not os.path.exists(execution_dir):
            return []
        
        execution_ids = []
        for filename in os.listdir(execution_dir):
            if filename.endswith('.pkl'):
                execution_ids.append(filename[:-4])  # Remove .pkl extension
            elif filename.endswith('.json') and not filename[:-5] + '.pkl' in os.listdir(execution_dir):
                execution_ids.append(filename[:-5])  # Remove .json extension if no .pkl exists
        
        return sorted(execution_ids)
    
    def load_multiple_executions(self, execution_ids: List[str] = None) -> List[ExecutionData]:
        """
        Carrega múltiplas execuções do disco.
        
        Args:
            execution_ids: Lista de IDs específicos (None para carregar todas)
            
        Returns:
            List[ExecutionData]: Lista de dados de execução
        """
        if execution_ids is None:
            execution_ids = self.list_executions()
        
        executions = []
        for execution_id in execution_ids:
            try:
                execution_data = self.load_execution_data(execution_id)
                executions.append(execution_data)
            except FileOperationError as e:
                print(f"Warning: Failed to load execution {execution_id}: {e}")
        
        return executions
    
    def add_iteration_callback(self, callback: Callable[[IterationData], None]):
        """
        Adiciona callback para ser executado a cada iteração.
        
        Args:
            callback: Função a ser chamada com dados da iteração
        """
        self.iteration_callbacks.append(callback)
    
    def add_solution_callback(self, callback: Callable[[Solution], None]):
        """
        Adiciona callback para ser executado quando uma solução é registrada.
        
        Args:
            callback: Função a ser chamada com a solução
        """
        self.solution_callbacks.append(callback)
    
    def clear_callbacks(self):
        """Remove todos os callbacks registrados."""
        self.iteration_callbacks.clear()
        self.solution_callbacks.clear()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo das execuções coletadas.
        
        Returns:
            Dict: Resumo com estatísticas das execuções
        """
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for e in self.execution_history if e.success)
        
        if total_executions == 0:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'success_rate': 0.0,
                'algorithms': {},
                'avg_execution_time': 0.0
            }
        
        # Group by algorithm type
        algorithms = {}
        total_time = 0.0
        
        for execution in self.execution_history:
            algo_type = execution.algorithm_type
            if algo_type not in algorithms:
                algorithms[algo_type] = {'count': 0, 'successful': 0}
            
            algorithms[algo_type]['count'] += 1
            if execution.success:
                algorithms[algo_type]['successful'] += 1
            
            total_time += execution.execution_time
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions,
            'algorithms': algorithms,
            'avg_execution_time': total_time / total_executions
        }
    
    def _execution_to_dict(self, execution_data: ExecutionData) -> Dict[str, Any]:
        """Converte ExecutionData para dicionário serializável."""
        return {
            'execution_id': execution_data.execution_id,
            'algorithm_type': execution_data.algorithm_type,
            'config': execution_data.config,
            'execution_time': execution_data.execution_time,
            'timestamp': execution_data.timestamp.isoformat(),
            'success': execution_data.success,
            'error_message': execution_data.error_message,
            'routes': [self._route_to_dict(route) for route in execution_data.routes],
            'iterations_data': [self._iteration_to_dict(iter_data) for iter_data in execution_data.iterations_data],
            'final_solution': self._solution_to_dict(execution_data.final_solution) if execution_data.final_solution else None
        }
    
    def _route_to_dict(self, route: Route) -> Dict[str, Any]:
        """Converte Route para dicionário."""
        return {
            'stops': route.stops,
            'distances': route.distances,
            'passenger_load': route.passenger_load,
            'total_distance': route.total_distance,
            'total_passengers': route.total_passengers,
            'is_valid': route.is_valid,
            'capacity_violations': route.capacity_violations,
            'opposite_stops_violations': route.opposite_stops_violations
        }
    
    def _solution_to_dict(self, solution: Solution) -> Dict[str, Any]:
        """Converte Solution para dicionário."""
        return {
            'routes': [self._route_to_dict(route) for route in solution.routes],
            'total_vehicles': solution.total_vehicles,
            'total_distance': solution.total_distance,
            'total_passengers_served': solution.total_passengers_served,
            'fitness_time': solution.fitness_time,
            'fitness_vehicle': solution.fitness_vehicle,
            'is_feasible': solution.is_feasible,
            'generation_time': solution.generation_time
        }
    
    def _iteration_to_dict(self, iteration_data: IterationData) -> Dict[str, Any]:
        """Converte IterationData para dicionário."""
        return {
            'iteration': iteration_data.iteration,
            'best_fitness': iteration_data.best_fitness,
            'avg_fitness': iteration_data.avg_fitness,
            'population_variance': iteration_data.population_variance,
            'timestamp': iteration_data.timestamp.isoformat(),
            'best_solution': self._solution_to_dict(iteration_data.best_solution),
            'additional_metrics': iteration_data.additional_metrics
        }
    
    def _dict_to_execution(self, data_dict: Dict[str, Any]) -> ExecutionData:
        """Converte dicionário para ExecutionData."""
        # This is a simplified version - in practice, you'd want full reconstruction
        # For now, we'll rely on pickle for exact reconstruction
        raise NotImplementedError("JSON to ExecutionData conversion not fully implemented. Use pickle files.")


class ACODataCollectorWrapper:
    """
    Wrapper para integração fácil com algoritmos ACO existentes.
    
    Esta classe fornece métodos convenientes para integrar a coleta
    de dados com o código ACO existente sem modificações extensas.
    """
    
    def __init__(self, collector: DataCollector):
        """
        Inicializa o wrapper.
        
        Args:
            collector: Instância do DataCollector
        """
        self.collector = collector
        self.execution_id: Optional[str] = None
    
    def wrap_controller_run(self, 
                           controller_run_method: Callable,
                           algorithm_type: str,
                           config: Dict[str, Any]):
        """
        Wrapper para o método run do ACSController.
        
        Args:
            controller_run_method: Método run original do controller
            algorithm_type: Tipo do algoritmo
            config: Configuração da execução
            
        Returns:
            Resultado original do método run
        """
        import time
        
        # Start data collection
        self.execution_id = self.collector.start_execution(algorithm_type, config)
        start_time = time.time()
        
        try:
            # Execute original method
            result = controller_run_method()
            
            # Record success
            execution_time = time.time() - start_time
            
            # Create solution from result (this would need adaptation based on actual return format)
            if isinstance(result, tuple) and len(result) >= 4:
                routes, total_dist, num_routes, coverage = result[:4]
                
                # Convert to our data structures
                solution_routes = []
                for route in routes:
                    # This conversion would need to be adapted based on actual route format
                    solution_route = Route(
                        stops=route,
                        distances=[0.0] * (len(route) - 1),  # Would need actual distances
                        passenger_load=[10] * (len(route) - 1),  # Would need actual loads
                        total_distance=total_dist / len(routes) if routes else 0.0,
                        total_passengers=len(route) * 10,  # Simplified
                        is_valid=True,  # Would need actual validation
                        capacity_violations=0,
                        opposite_stops_violations=0
                    )
                    solution_routes.append(solution_route)
                
                solution = Solution(
                    routes=solution_routes,
                    total_vehicles=num_routes,
                    total_distance=total_dist,
                    total_passengers_served=sum(r.total_passengers for r in solution_routes),
                    fitness_time=total_dist,
                    fitness_vehicle=num_routes,
                    is_feasible=True
                )
                
                self.collector.record_final_solution(solution, execution_time, True)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self.collector.record_final_solution(None, execution_time, False, str(e))
            raise
        
        finally:
            # Finish execution
            self.collector.finish_execution()
            self.execution_id = None