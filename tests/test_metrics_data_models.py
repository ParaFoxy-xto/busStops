"""
Testes unitários para os modelos de dados do sistema de métricas.
"""

import pytest
from datetime import datetime
from src.rota_aco.metrics.data_models import (
    Route, Solution, IterationData, ExecutionData,
    RouteQualityMetrics, ConvergenceMetrics, ComparativeMetrics,
    DomainMetrics, ExecutionSummary, MetricsReport
)


class TestRoute:
    """Testes para a classe Route"""
    
    def test_valid_route_creation(self):
        """Testa criação de rota válida"""
        route = Route(
            stops=[1, 2, 3, 4],
            distances=[10.0, 15.0, 8.0],
            passenger_load=[20, 35, 15],
            total_distance=33.0,
            total_passengers=70,
            is_valid=True
        )
        assert len(route.stops) == 4
        assert len(route.distances) == 3
        assert len(route.passenger_load) == 3
        assert route.total_distance == 33.0
        assert route.is_valid
    
    def test_invalid_route_stops_mismatch(self):
        """Testa erro quando número de paradas não confere com passenger_load"""
        with pytest.raises(ValueError, match="Número de paradas deve ser igual"):
            Route(
                stops=[1, 2, 3],  # 3 paradas
                distances=[10.0, 15.0],
                passenger_load=[20, 35, 15, 10],  # 4 segmentos
                total_distance=25.0,
                total_passengers=80,
                is_valid=False
            )
    
    def test_invalid_route_distances_mismatch(self):
        """Testa erro quando número de distâncias não confere"""
        with pytest.raises(ValueError, match="Número de distâncias deve ser igual"):
            Route(
                stops=[1, 2, 3, 4],  # 4 paradas
                distances=[10.0, 15.0, 8.0, 5.0],  # 4 distâncias (deveria ser 3)
                passenger_load=[20, 35, 15],
                total_distance=38.0,
                total_passengers=70,
                is_valid=True
            )


class TestSolution:
    """Testes para a classe Solution"""
    
    def test_valid_solution_creation(self):
        """Testa criação de solução válida"""
        routes = [
            Route([1, 2, 3], [10.0, 15.0], [20, 30], 25.0, 50, True),
            Route([4, 5, 6], [8.0, 12.0], [15, 25], 20.0, 40, True)
        ]
        
        solution = Solution(
            routes=routes,
            total_vehicles=2,
            total_distance=45.0,
            total_passengers_served=90,
            fitness_time=45.0,
            fitness_vehicle=2.0,
            is_feasible=True
        )
        
        assert len(solution.routes) == 2
        assert solution.total_vehicles == 2
        assert solution.is_feasible
    
    def test_invalid_solution_vehicle_count(self):
        """Testa erro quando número de veículos não confere com rotas"""
        routes = [
            Route([1, 2, 3], [10.0, 15.0], [20, 30], 25.0, 50, True)
        ]
        
        with pytest.raises(ValueError, match="Número de veículos deve ser igual"):
            Solution(
                routes=routes,
                total_vehicles=3,  # Deveria ser 1
                total_distance=25.0,
                total_passengers_served=50,
                fitness_time=25.0,
                fitness_vehicle=3.0,
                is_feasible=True
            )


class TestIterationData:
    """Testes para a classe IterationData"""
    
    def test_iteration_data_creation(self):
        """Testa criação de dados de iteração"""
        solution = Solution(
            routes=[Route([1, 2], [10.0], [20], 10.0, 20, True)],
            total_vehicles=1,
            total_distance=10.0,
            total_passengers_served=20,
            fitness_time=10.0,
            fitness_vehicle=1.0,
            is_feasible=True
        )
        
        iteration_data = IterationData(
            iteration=1,
            best_fitness=10.0,
            avg_fitness=15.0,
            population_variance=2.5,
            best_solution=solution
        )
        
        assert iteration_data.iteration == 1
        assert iteration_data.best_fitness == 10.0
        assert iteration_data.timestamp is not None


class TestExecutionData:
    """Testes para a classe ExecutionData"""
    
    def test_valid_execution_data(self):
        """Testa criação de dados de execução válidos"""
        routes = [Route([1, 2], [10.0], [20], 10.0, 20, True)]
        solution = Solution(routes, 1, 10.0, 20, 10.0, 1.0, True)
        iterations = [IterationData(1, 10.0, 15.0, 2.5, solution)]
        
        execution_data = ExecutionData(
            execution_id="test_001",
            algorithm_type="ACS-TIME",
            config={"ants": 10, "iterations": 100},
            routes=routes,
            iterations_data=iterations,
            execution_time=5.2,
            timestamp=datetime.now(),
            final_solution=solution
        )
        
        assert execution_data.algorithm_type == "ACS-TIME"
        assert execution_data.execution_id == "test_001"
    
    def test_invalid_algorithm_type(self):
        """Testa erro com tipo de algoritmo inválido"""
        routes = [Route([1, 2], [10.0], [20], 10.0, 20, True)]
        solution = Solution(routes, 1, 10.0, 20, 10.0, 1.0, True)
        
        with pytest.raises(ValueError, match="algorithm_type deve ser"):
            ExecutionData(
                execution_id="test_001",
                algorithm_type="INVALID_TYPE",
                config={},
                routes=routes,
                iterations_data=[],
                execution_time=5.2,
                timestamp=datetime.now(),
                final_solution=solution
            )


class TestRouteQualityMetrics:
    """Testes para métricas de qualidade de rotas"""
    
    def test_valid_quality_metrics(self):
        """Testa criação de métricas de qualidade válidas"""
        metrics = RouteQualityMetrics(
            valid_routes_percentage=85.5,
            demand_coverage_percentage=92.3,
            vehicle_utilization_efficiency=0.78,
            capacity_violations=2,
            opposite_stops_violations=1,
            total_routes_analyzed=10
        )
        
        assert metrics.valid_routes_percentage == 85.5
        assert metrics.demand_coverage_percentage == 92.3
    
    def test_invalid_percentage_values(self):
        """Testa erro com valores de porcentagem inválidos"""
        with pytest.raises(ValueError, match="valid_routes_percentage deve estar entre"):
            RouteQualityMetrics(
                valid_routes_percentage=150.0,  # Inválido
                demand_coverage_percentage=92.3,
                vehicle_utilization_efficiency=0.78,
                capacity_violations=2,
                opposite_stops_violations=1,
                total_routes_analyzed=10
            )


class TestConvergenceMetrics:
    """Testes para métricas de convergência"""
    
    def test_valid_convergence_metrics(self):
        """Testa criação de métricas de convergência válidas"""
        metrics = ConvergenceMetrics(
            convergence_point=45,
            final_stability=0.95,
            improvement_rate=0.02,
            plateau_detection=True,
            convergence_speed=0.85,
            total_iterations=100,
            best_fitness_evolution=[100.0, 95.0, 90.0],
            avg_fitness_evolution=[110.0, 105.0, 100.0],
            variance_evolution=[5.0, 3.0, 2.0]
        )
        
        assert metrics.convergence_point == 45
        assert metrics.total_iterations == 100
        assert metrics.plateau_detection is True
    
    def test_invalid_convergence_point(self):
        """Testa erro com ponto de convergência inválido"""
        with pytest.raises(ValueError, match="convergence_point deve estar entre"):
            ConvergenceMetrics(
                convergence_point=150,  # Maior que total_iterations
                final_stability=0.95,
                improvement_rate=0.02,
                plateau_detection=True,
                convergence_speed=0.85,
                total_iterations=100,
                best_fitness_evolution=[],
                avg_fitness_evolution=[],
                variance_evolution=[]
            )


class TestComparativeMetrics:
    """Testes para métricas comparativas"""
    
    def test_valid_comparative_metrics(self):
        """Testa criação de métricas comparativas válidas"""
        metrics = ComparativeMetrics(
            mean_fitness=85.5,
            median_fitness=87.2,
            std_dev_fitness=5.3,
            mean_execution_time=12.5,
            success_rate=0.9,
            total_executions=10,
            successful_executions=9,
            fitness_evaluations_count=50000
        )
        
        assert metrics.success_rate == 0.9
        assert metrics.total_executions == 10
        assert metrics.successful_executions == 9
    
    def test_invalid_success_rate(self):
        """Testa erro com taxa de sucesso inválida"""
        with pytest.raises(ValueError, match="success_rate deve estar entre"):
            ComparativeMetrics(
                mean_fitness=85.5,
                median_fitness=87.2,
                std_dev_fitness=5.3,
                mean_execution_time=12.5,
                success_rate=1.5,  # Inválido
                total_executions=10,
                successful_executions=9,
                fitness_evaluations_count=50000
            )


class TestDomainMetrics:
    """Testes para métricas específicas do domínio"""
    
    def test_valid_domain_metrics(self):
        """Testa criação de métricas de domínio válidas"""
        metrics = DomainMetrics(
            estimated_travel_time=45.5,
            average_transfers=1.2,
            geographic_coverage=25.8,
            load_balancing_index=0.85,
            energy_efficiency=2.3,
            accessibility_index=0.15
        )
        
        assert metrics.estimated_travel_time == 45.5
        assert metrics.load_balancing_index == 0.85
    
    def test_invalid_load_balancing_index(self):
        """Testa erro com índice de balanceamento inválido"""
        with pytest.raises(ValueError, match="load_balancing_index deve estar entre"):
            DomainMetrics(
                estimated_travel_time=45.5,
                average_transfers=1.2,
                geographic_coverage=25.8,
                load_balancing_index=1.5,  # Inválido
                energy_efficiency=2.3,
                accessibility_index=0.15
            )


class TestExecutionSummary:
    """Testes para resumo de execuções"""
    
    def test_execution_summary_auto_calculation(self):
        """Testa cálculo automático da taxa de sucesso"""
        summary = ExecutionSummary(
            total_executions=10,
            successful_executions=8,
            success_rate=0.7,  # Será corrigido automaticamente para 0.8
            avg_execution_time=15.2,
            avg_iterations_to_convergence=75.5,
            best_overall_fitness=82.3,
            worst_overall_fitness=95.7
        )
        
        assert summary.success_rate == 0.8  # Corrigido automaticamente
        assert summary.total_executions == 10
        assert summary.successful_executions == 8