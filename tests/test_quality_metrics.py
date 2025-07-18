"""
Testes para o sistema de métricas de qualidade de rotas.

Este módulo testa todas as funcionalidades do RouteQualityEvaluator,
incluindo cálculo de métricas, validação de rotas e casos extremos.
"""

import pytest
from src.rota_aco.metrics.quality_metrics import RouteQualityEvaluator
from src.rota_aco.metrics.data_models import Route, Solution, RouteQualityMetrics
from src.rota_aco.metrics.exceptions import MetricsCalculationError


class TestRouteQualityEvaluator:
    """Testes para a classe RouteQualityEvaluator."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.evaluator = RouteQualityEvaluator(capacity_limit=70)
        self.evaluator_with_graph = RouteQualityEvaluator(
            capacity_limit=70,
            graph_data={'total_stops': 20}
        )
    
    def create_valid_route(self) -> Route:
        """Cria uma rota válida para testes."""
        return Route(
            stops=[1, 3, 5, 7],
            distances=[10.0, 15.0, 12.0],
            passenger_load=[30, 45, 25],
            total_distance=37.0,
            total_passengers=100,
            is_valid=True
        )
    
    def create_invalid_route(self) -> Route:
        """Cria uma rota inválida para testes."""
        return Route(
            stops=[2, 4, 6, 8],
            distances=[8.0, 20.0, 10.0],
            passenger_load=[80, 90, 75],  # Violações de capacidade
            total_distance=38.0,
            total_passengers=245,
            is_valid=False,
            capacity_violations=3
        )
    
    def create_opposite_stops_route(self) -> Route:
        """Cria uma rota com violações de paradas opostas."""
        return Route(
            stops=[1, 2, 3, 4],  # 1-2 e 3-4 são pares opostos
            distances=[5.0, 8.0, 6.0],
            passenger_load=[20, 30, 25],
            total_distance=19.0,
            total_passengers=75,
            is_valid=False,
            opposite_stops_violations=2
        )
    
    def test_evaluate_empty_routes(self):
        """Testa avaliação com lista vazia de rotas."""
        metrics = self.evaluator.evaluate_routes([])
        
        assert metrics.valid_routes_percentage == 0.0
        assert metrics.demand_coverage_percentage == 0.0
        assert metrics.vehicle_utilization_efficiency == 0.0
        assert metrics.capacity_violations == 0
        assert metrics.opposite_stops_violations == 0
        assert metrics.average_route_length == 0.0
        assert metrics.route_length_variance == 0.0
        assert metrics.load_balancing_index == 0.0
    
    def test_evaluate_single_valid_route(self):
        """Testa avaliação com uma única rota válida."""
        route = self.create_valid_route()
        metrics = self.evaluator.evaluate_routes([route])
        
        assert metrics.valid_routes_percentage == 100.0
        assert metrics.capacity_violations == 0
        assert metrics.average_route_length == 37.0
        assert metrics.route_length_variance == 0.0  # Apenas uma rota
        assert metrics.vehicle_utilization_efficiency > 0
    
    def test_evaluate_single_invalid_route(self):
        """Testa avaliação com uma única rota inválida."""
        route = self.create_invalid_route()
        metrics = self.evaluator.evaluate_routes([route])
        
        assert metrics.valid_routes_percentage == 0.0
        assert metrics.capacity_violations == 3  # 80, 90, 75 > 70
        assert metrics.average_route_length == 38.0
    
    def test_evaluate_mixed_routes(self):
        """Testa avaliação com mix de rotas válidas e inválidas."""
        valid_route = self.create_valid_route()
        invalid_route = self.create_invalid_route()
        routes = [valid_route, invalid_route]
        
        metrics = self.evaluator.evaluate_routes(routes)
        
        assert metrics.valid_routes_percentage == 50.0  # 1 de 2 válidas
        assert metrics.capacity_violations == 3
        assert abs(metrics.average_route_length - 37.5) < 0.1  # (37+38)/2
        assert metrics.route_length_variance > 0  # Há variação
    
    def test_demand_coverage_with_graph_data(self):
        """Testa cálculo de cobertura com dados do grafo."""
        route = Route(
            stops=[1, 2, 3, 4, 5],  # 5 paradas de 20 total
            distances=[1.0, 1.0, 1.0, 1.0],
            passenger_load=[10, 10, 10, 10],
            total_distance=4.0,
            total_passengers=40,
            is_valid=True
        )
        
        metrics = self.evaluator_with_graph.evaluate_routes([route])
        assert metrics.demand_coverage_percentage == 25.0  # 5/20 * 100
    
    def test_demand_coverage_without_graph_data(self):
        """Testa cálculo de cobertura sem dados do grafo."""
        route = self.create_valid_route()
        metrics = self.evaluator.evaluate_routes([route])
        
        # Sem dados do grafo, assume 100% de cobertura das paradas visitadas
        assert metrics.demand_coverage_percentage == 100.0
    
    def test_vehicle_utilization_efficiency(self):
        """Testa cálculo de eficiência de utilização."""
        route = Route(
            stops=[1, 2, 3],
            distances=[10.0, 10.0],
            passenger_load=[35, 70],  # 50% e 100% de utilização
            total_distance=20.0,
            total_passengers=105,
            is_valid=True
        )
        
        metrics = self.evaluator.evaluate_routes([route])
        expected_efficiency = (50.0 + 100.0) / 2  # Média: 75%
        assert abs(metrics.vehicle_utilization_efficiency - expected_efficiency) < 0.1
    
    def test_capacity_violations_count(self):
        """Testa contagem de violações de capacidade."""
        route = Route(
            stops=[1, 2, 3, 4],
            distances=[5.0, 5.0, 5.0],
            passenger_load=[80, 90, 75],  # 3 violações
            total_distance=15.0,
            total_passengers=245,
            is_valid=False
        )
        
        metrics = self.evaluator.evaluate_routes([route])
        assert metrics.capacity_violations == 3
    
    def test_opposite_stops_violations(self):
        """Testa detecção de violações de paradas opostas."""
        route = self.create_opposite_stops_route()
        metrics = self.evaluator.evaluate_routes([route])
        
        # Deve detectar pelo menos uma violação (1-2 ou 3-4)
        assert metrics.opposite_stops_violations >= 1
    
    def test_load_balancing_perfect(self):
        """Testa índice de balanceamento com cargas perfeitamente balanceadas."""
        routes = [
            Route([1, 2], [10.0], [50], 10.0, 50, True),
            Route([3, 4], [10.0], [50], 10.0, 50, True),
            Route([5, 6], [10.0], [50], 10.0, 50, True)
        ]
        
        metrics = self.evaluator.evaluate_routes(routes)
        assert metrics.load_balancing_index == 1.0  # Perfeitamente balanceado
    
    def test_load_balancing_unbalanced(self):
        """Testa índice de balanceamento com cargas desbalanceadas."""
        routes = [
            Route([1, 2], [10.0], [10], 10.0, 10, True),
            Route([3, 4], [10.0], [50], 10.0, 50, True),
            Route([5, 6], [10.0], [90], 10.0, 90, True)
        ]
        
        metrics = self.evaluator.evaluate_routes(routes)
        assert metrics.load_balancing_index < 1.0  # Desbalanceado
        assert metrics.load_balancing_index > 0.0  # Mas não zero
    
    def test_route_length_variance(self):
        """Testa cálculo de variância do comprimento das rotas."""
        routes = [
            Route([1, 2], [10.0], [30], 10.0, 30, True),
            Route([3, 4], [20.0], [40], 20.0, 40, True),
            Route([5, 6], [30.0], [50], 30.0, 50, True)
        ]
        
        metrics = self.evaluator.evaluate_routes(routes)
        assert metrics.average_route_length == 20.0  # (10+20+30)/3
        assert metrics.route_length_variance == 100.0  # Variância de [10,20,30]
    
    def test_evaluate_solution(self):
        """Testa avaliação de uma solução completa."""
        routes = [self.create_valid_route(), self.create_invalid_route()]
        solution = Solution(
            routes=routes,
            total_vehicles=2,
            total_distance=75.0,
            total_passengers_served=345,
            fitness_time=100.0,
            fitness_vehicle=2.0,
            is_feasible=False
        )
        
        metrics = self.evaluator.evaluate_solution(solution)
        assert metrics.valid_routes_percentage == 50.0
        assert metrics.capacity_violations == 3
    
    def test_validate_route_valid(self):
        """Testa validação de rota válida."""
        route = self.create_valid_route()
        validation = self.evaluator.validate_route(route)
        
        assert validation['capacity_valid'] is True
        assert validation['structure_valid'] is True
        assert validation['load_consistency'] is True
        assert validation['positive_distances'] is True
        assert validation['non_empty'] is True
        assert validation['overall_valid'] is True
    
    def test_validate_route_invalid_capacity(self):
        """Testa validação de rota com violação de capacidade."""
        route = Route(
            stops=[1, 2, 3],
            distances=[10.0, 10.0],
            passenger_load=[80, 90],  # Violações
            total_distance=20.0,
            total_passengers=170,
            is_valid=False
        )
        
        validation = self.evaluator.validate_route(route)
        assert validation['capacity_valid'] is False
        assert validation['overall_valid'] is False
    
    def test_validate_route_structure_invalid(self):
        """Testa validação de rota com estrutura inválida."""
        # Como o Route tem validação no __post_init__, vamos testar
        # que a exceção é lançada para estruturas inválidas
        with pytest.raises(ValueError, match="Número de paradas deve ser distâncias"):
            Route(
                stops=[1, 2],  # 2 paradas
                distances=[10.0, 15.0, 20.0],  # 3 distâncias (deveria ser 1)
                passenger_load=[30, 40, 50],  # 3 cargas
                total_distance=45.0,
                total_passengers=120,
                is_valid=False
            )
    
    def test_get_quality_summary(self):
        """Testa geração de resumo de qualidade."""
        route = self.create_valid_route()
        metrics = self.evaluator.evaluate_routes([route])
        summary = self.evaluator.get_quality_summary(metrics)
        
        assert "Resumo de Qualidade das Rotas" in summary
        assert "100.0%" in summary  # Rotas válidas
        assert "37.00" in summary   # Comprimento médio
        assert isinstance(summary, str)
        assert len(summary) > 100  # Resumo substancial
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Como o Route tem validação no __post_init__, vamos testar
        # que a exceção é lançada para dados inconsistentes
        with pytest.raises(ValueError, match="Número de paradas deve ser distâncias"):
            Route(
                stops=[],  # Lista vazia
                distances=[10.0],  # Mas com distâncias
                passenger_load=[],
                total_distance=10.0,
                total_passengers=0,
                is_valid=False
            )
        
        # Testar com rota válida mas com dados que podem causar problemas no cálculo
        edge_case_route = Route(
            stops=[1],  # Apenas uma parada
            distances=[],  # Sem distâncias
            passenger_load=[],  # Sem carga
            total_distance=0.0,
            total_passengers=0,
            is_valid=True
        )
        
        # O avaliador deve lidar com isso graciosamente
        metrics = self.evaluator.evaluate_routes([edge_case_route])
        assert isinstance(metrics, RouteQualityMetrics)
    
    def test_multiple_routes_comprehensive(self):
        """Teste abrangente com múltiplas rotas de diferentes tipos."""
        routes = [
            self.create_valid_route(),
            self.create_invalid_route(),
            self.create_opposite_stops_route(),
            Route([10, 11, 12], [5.0, 5.0], [20, 30], 10.0, 50, True)
        ]
        
        metrics = self.evaluator.evaluate_routes(routes)
        
        # Verificações básicas
        assert 0 <= metrics.valid_routes_percentage <= 100
        assert metrics.capacity_violations >= 0
        assert metrics.opposite_stops_violations >= 0
        assert metrics.average_route_length > 0
        assert 0 <= metrics.load_balancing_index <= 1
        assert 0 <= metrics.vehicle_utilization_efficiency <= 100