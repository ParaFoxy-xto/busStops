"""
Testes para integração do sistema de métricas com algoritmos ACO.

Este módulo testa a funcionalidade de integração não-intrusiva
com o código ACO existente.
"""

import pytest
import tempfile
from unittest.mock import Mock, MagicMock
import networkx as nx

from src.rota_aco.metrics.aco_integration import (
    ACOMetricsIntegrator,
    create_metrics_enabled_controller,
    run_aco_with_metrics
)
from src.rota_aco.metrics.config import MetricsConfig
from src.rota_aco.metrics.data_models import Route, Solution


class TestACOMetricsIntegrator:
    """Testes para a classe ACOMetricsIntegrator."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MetricsConfig(base_output_dir=self.temp_dir)
        self.integrator = ACOMetricsIntegrator(self.config)
    
    def teardown_method(self):
        """Cleanup após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_controller(self):
        """Cria um mock controller para testes."""
        controller = Mock()
        controller.meta_edges = {
            (1, 2): {'time': 100.0},
            (2, 3): {'time': 150.0},
            (3, 4): {'time': 200.0}
        }
        controller.stops_to_visit = [2, 3]
        controller.capacity = 70
        controller.opposites = {2: [3], 3: [2]}
        controller.history = []
        
        # Mock methods
        controller._calculate_solution_quality = Mock(return_value=100.0)
        controller._get_solution_metrics = Mock(return_value=(250.0, 1, 0.95))
        controller._record_history = Mock()
        
        return controller
    
    def test_determine_algorithm_type(self):
        """Testa determinação do tipo de algoritmo."""
        # Peso das rotas maior -> ACS-VEHICLE
        weights1 = {'w_r': 2.0, 'w_d': 0.5}
        assert self.integrator._determine_algorithm_type(weights1) == "ACS-VEHICLE"
        
        # Peso da distância maior -> ACS-TIME
        weights2 = {'w_r': 0.5, 'w_d': 2.0}
        assert self.integrator._determine_algorithm_type(weights2) == "ACS-TIME"
        
        # Pesos iguais -> ACS-TIME (padrão)
        weights3 = {'w_r': 1.0, 'w_d': 1.0}
        assert self.integrator._determine_algorithm_type(weights3) == "ACS-TIME"
    
    def test_convert_single_route_valid(self):
        """Testa conversão de rota válida."""
        controller = self.create_mock_controller()
        route = [1, 2, 3]
        
        converted = self.integrator._convert_single_route(route, controller)
        
        assert isinstance(converted, Route)
        assert converted.stops == [1, 2, 3]
        assert len(converted.distances) == 2
        assert converted.distances[0] == 100.0  # (1,2)
        assert converted.distances[1] == 150.0  # (2,3)
        assert converted.total_distance == 250.0
        assert converted.total_passengers == 10  # Apenas parada 2 conta (parada 3 também)
        assert converted.capacity_violations == 0  # 10 < 70
        assert converted.opposite_stops_violations == 2  # 2 e 3 são opostos (conta ambos)
        assert not converted.is_valid  # Devido à violação de opostos
    
    def test_convert_single_route_invalid(self):
        """Testa conversão de rota inválida."""
        controller = self.create_mock_controller()
        route = [1]  # Rota muito curta
        
        converted = self.integrator._convert_single_route(route, controller)
        
        assert isinstance(converted, Route)
        assert converted.stops == [1]
        assert converted.distances == []
        assert converted.total_distance == 0.0
        assert not converted.is_valid
        assert converted.capacity_violations == 1
    
    def test_convert_single_route_caching(self):
        """Testa cache de conversão de rotas."""
        controller = self.create_mock_controller()
        route = [1, 2, 3]
        
        # Primeira conversão
        converted1 = self.integrator._convert_single_route(route, controller)
        assert len(self.integrator._route_cache) == 1
        
        # Segunda conversão (deve usar cache)
        converted2 = self.integrator._convert_single_route(route, controller)
        assert converted1 is converted2  # Mesmo objeto do cache
    
    def test_convert_routes_to_solution_empty(self):
        """Testa conversão de lista vazia de rotas."""
        controller = self.create_mock_controller()
        routes = []
        
        solution = self.integrator._convert_routes_to_solution(routes, controller, "ACS-TIME")
        
        assert isinstance(solution, Solution)
        assert solution.routes == []
        assert solution.total_vehicles == 0
        assert solution.total_distance == 0.0
        assert solution.fitness_time == float('inf')
        assert not solution.is_feasible
    
    def test_convert_routes_to_solution_valid(self):
        """Testa conversão de rotas válidas para solução."""
        controller = self.create_mock_controller()
        routes = [[1, 2, 3], [1, 4]]
        
        solution = self.integrator._convert_routes_to_solution(routes, controller, "ACS-TIME")
        
        assert isinstance(solution, Solution)
        assert len(solution.routes) == 2
        assert solution.total_vehicles == 2
        assert solution.total_distance > 0
        assert solution.fitness_time == solution.total_distance
        assert solution.fitness_vehicle > solution.fitness_time  # Penaliza mais veículos
    
    def test_wrap_controller_execution_success(self):
        """Testa wrapper de execução bem-sucedida."""
        controller = self.create_mock_controller()
        
        # Mock do método run
        def mock_run(n_ants, n_iterations, quality_weights, verbose):
            # Simular algumas iterações
            for i in range(n_iterations):
                controller._record_history(
                    iteration_num=i,
                    time_solution=[[1, 2, 3]],
                    vehicle_solution=[[1, 2], [3, 4]],
                    chosen_solution_quality=100.0 - i,
                    best_overall_quality=100.0 - i
                )
            return [[1, 2, 3]], 250.0, 1, 0.95
        
        controller.run = mock_run
        
        # Parâmetros
        algorithm_params = {'alpha': 1.0, 'beta': 2.0}
        problem_params = {'capacity': 70}
        quality_weights = {'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5}
        
        # Criar wrapper
        wrapped_run = self.integrator.wrap_controller_execution(
            controller, algorithm_params, problem_params, quality_weights
        )
        
        # Executar
        result = wrapped_run(n_ants=10, n_iterations=3, verbose=False)
        
        # Verificar resultado
        assert result == ([[1, 2, 3]], 250.0, 1, 0.95)
        
        # Verificar coleta de dados
        collected_data = self.integrator.get_collected_data()
        assert len(collected_data) == 1
        
        execution = collected_data[0]
        assert execution.success is True
        assert execution.algorithm_type == "ACS-VEHICLE"  # Baseado nos pesos
        assert execution.final_solution is not None
        assert execution.execution_time > 0
    
    def test_wrap_controller_execution_failure(self):
        """Testa wrapper com execução que falha."""
        controller = self.create_mock_controller()
        
        # Mock que falha
        def mock_run(n_ants, n_iterations, quality_weights, verbose):
            raise Exception("Algorithm failed to converge")
        
        controller.run = mock_run
        
        # Parâmetros
        algorithm_params = {'alpha': 1.0}
        problem_params = {'capacity': 70}
        quality_weights = {'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5}
        
        # Criar wrapper
        wrapped_run = self.integrator.wrap_controller_execution(
            controller, algorithm_params, problem_params, quality_weights
        )
        
        # Executar (deve re-lançar exceção)
        with pytest.raises(Exception, match="Algorithm failed to converge"):
            wrapped_run(n_ants=10, n_iterations=3, verbose=False)
        
        # Verificar que falha foi registrada
        collected_data = self.integrator.get_collected_data()
        assert len(collected_data) == 1
        
        execution = collected_data[0]
        assert execution.success is False
        assert execution.error_message == "Algorithm failed to converge"
    
    def test_history_interceptor(self):
        """Testa interceptação do histórico do controller."""
        controller = self.create_mock_controller()
        
        # Iniciar execução
        self.integrator.collector.start_execution("ACS-TIME", {})
        
        # Criar interceptador
        original_record_history = Mock()
        history_interceptor = self.integrator._create_history_interceptor(
            original_record_history, controller
        )
        
        # Executar interceptador
        history_interceptor(
            iteration_num=0,
            time_solution=[[1, 2, 3]],
            vehicle_solution=[[1, 2], [3, 4]],
            chosen_solution_quality=95.0,
            best_overall_quality=95.0
        )
        
        # Verificar que método original foi chamado
        original_record_history.assert_called_once_with(
            0, [[1, 2, 3]], [[1, 2], [3, 4]], 95.0, 95.0
        )
        
        # Verificar que dados foram coletados
        assert len(self.integrator.collector.current_execution.iterations_data) == 1
        
        iteration_data = self.integrator.collector.current_execution.iterations_data[0]
        assert iteration_data.iteration == 0
        assert iteration_data.best_fitness == 95.0
        assert 'time_solution_quality' in iteration_data.additional_metrics
    
    def test_clear_cache(self):
        """Testa limpeza do cache."""
        controller = self.create_mock_controller()
        
        # Adicionar algo ao cache
        self.integrator._convert_single_route([1, 2, 3], controller)
        assert len(self.integrator._route_cache) > 0
        
        # Limpar cache
        self.integrator.clear_cache()
        assert len(self.integrator._route_cache) == 0
        assert len(self.integrator._solution_cache) == 0


class TestIntegrationFunctions:
    """Testes para funções de integração."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MetricsConfig(base_output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_metrics_enabled_controller(self):
        """Testa criação de controller com métricas habilitadas."""
        # Mock controller class
        MockControllerClass = Mock()
        mock_instance = Mock()
        MockControllerClass.return_value = mock_instance
        
        # Parâmetros
        graph = nx.DiGraph()
        meta_edges = {}
        stops_to_visit = []
        start_node = 1
        exit_node = 2
        opposites = {}
        aco_params = {'alpha': 1.0}
        problem_params = {'capacity': 70}
        
        # Criar controller com métricas
        controller, integrator = create_metrics_enabled_controller(
            MockControllerClass, graph, meta_edges, stops_to_visit,
            start_node, exit_node, opposites, aco_params, problem_params,
            self.config
        )
        
        # Verificar criação
        assert controller is mock_instance
        assert isinstance(integrator, ACOMetricsIntegrator)
        
        # Verificar que controller foi criado com parâmetros corretos
        MockControllerClass.assert_called_once_with(
            graph=graph,
            meta_edges=meta_edges,
            stops_to_visit=stops_to_visit,
            start_node=start_node,
            exit_node=exit_node,
            opposites=opposites,
            aco_params=aco_params,
            problem_params=problem_params
        )
    
    def test_run_aco_with_metrics(self):
        """Testa execução completa com métricas."""
        # Mock controller class
        MockControllerClass = Mock()
        mock_controller = Mock()
        MockControllerClass.return_value = mock_controller
        
        # Mock do método run
        def mock_run(n_ants, n_iterations, quality_weights, verbose):
            return [[1, 2, 3]], 250.0, 1, 0.95
        
        mock_controller.run = mock_run
        mock_controller.meta_edges = {(1, 2): {'time': 100}, (2, 3): {'time': 150}}
        mock_controller.stops_to_visit = [2, 3]
        mock_controller.capacity = 70
        mock_controller.opposites = {}
        mock_controller.history = []
        mock_controller._record_history = Mock()
        
        # Parâmetros
        graph = nx.DiGraph()
        meta_edges = {}
        stops_to_visit = []
        start_node = 1
        exit_node = 2
        opposites = {}
        aco_params = {'alpha': 1.0}
        problem_params = {'capacity': 70}
        quality_weights = {'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5}
        
        # Executar com métricas
        result, execution_data = run_aco_with_metrics(
            MockControllerClass, graph, meta_edges, stops_to_visit,
            start_node, exit_node, opposites, aco_params, problem_params,
            quality_weights, n_ants=10, n_iterations=5, verbose=False,
            metrics_config=self.config
        )
        
        # Verificar resultado
        assert result == ([[1, 2, 3]], 250.0, 1, 0.95)
        assert execution_data is not None
        assert execution_data.success is True
        assert execution_data.algorithm_type == "ACS-VEHICLE"


if __name__ == "__main__":
    pytest.main([__file__])