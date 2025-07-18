"""
Testes para o sistema de métricas de transporte público.

Este módulo testa todas as funcionalidades do TransportationMetrics,
incluindo cálculo de métricas específicas do domínio de transporte
e análise de aplicabilidade prática das soluções ACO.
"""

import pytest
import math
from typing import List
from src.rota_aco.metrics.transportation_metrics import TransportationMetrics
from src.rota_aco.metrics.data_models import Route, Solution, DomainMetrics
from src.rota_aco.metrics.exceptions import MetricsCalculationError


class TestTransportationMetrics:
    """Testes para a classe TransportationMetrics."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.calculator = TransportationMetrics()
        
        # Calculator com dados geográficos mock
        self.graph_data = {
            'stop_coordinates': {
                1: (40.7128, -74.0060),  # NYC aproximado
                2: (40.7589, -73.9851),
                3: (40.6892, -74.0445),
                4: (40.7831, -73.9712),
                5: (40.7282, -73.7949)
            },
            'total_area_km2': 150.0
        }
        self.calculator_with_geo = TransportationMetrics(
            graph_data=self.graph_data,
            average_speed_kmh=25.0,
            transfer_time_minutes=3.0
        )
    
    def create_simple_route(self, stops: List[int], distances: List[float], 
                           passengers: List[int]) -> Route:
        """Cria uma rota simples para testes."""
        return Route(
            stops=stops,
            distances=distances,
            passenger_load=passengers,
            total_distance=sum(distances),
            total_passengers=sum(passengers),
            is_valid=True
        )
    
    def create_test_routes(self) -> List[Route]:
        """Cria um conjunto de rotas para testes."""
        routes = [
            self.create_simple_route([1, 2, 3], [5.0, 7.0], [20, 30]),
            self.create_simple_route([3, 4, 5], [6.0, 8.0], [25, 35]),
            self.create_simple_route([1, 4, 2], [4.0, 9.0], [15, 40])
        ]
        return routes
    
    def test_calculate_empty_routes(self):
        """Testa cálculo com lista vazia de rotas."""
        metrics = self.calculator.calculate_transportation_metrics([])
        
        assert metrics.estimated_travel_time == 0.0
        assert metrics.average_transfers == 0.0
        assert metrics.geographic_coverage == 0.0
        assert metrics.load_balancing_index == 0.0
        assert metrics.energy_efficiency == 0.0
        assert metrics.accessibility_index == 0.0
        assert metrics.service_frequency == 0.0
        assert metrics.route_overlap_percentage == 0.0
    
    def test_calculate_single_route(self):
        """Testa cálculo com uma única rota."""
        route = self.create_simple_route([1, 2, 3, 4], [10.0, 15.0, 12.0], [20, 30, 25])
        metrics = self.calculator.calculate_transportation_metrics([route])
        
        assert metrics.estimated_travel_time > 0  # Deve calcular tempo de viagem
        assert metrics.average_transfers == 0.0   # Apenas uma rota
        assert metrics.geographic_coverage > 0    # Deve estimar cobertura
        assert metrics.load_balancing_index == 1.0  # Perfeitamente balanceado (1 rota)
        assert metrics.energy_efficiency > 0      # Distância por passageiro
        assert metrics.accessibility_index > 0    # Paradas por área
        assert metrics.service_frequency > 0      # Frequência estimada
        assert metrics.route_overlap_percentage == 0.0  # Sem sobreposição
    
    def test_estimate_total_travel_time(self):
        """Testa estimativa de tempo total de viagem."""
        routes = self.create_test_routes()
        travel_time = self.calculator._estimate_total_travel_time(routes)
        
        assert travel_time > 0
        
        # Verificar cálculo manual para primeira rota
        route1 = routes[0]
        expected_travel_time = (route1.total_distance / 30.0) * 60  # 30 km/h padrão
        expected_stop_time = len(route1.stops) * 1.0
        expected_route1_time = expected_travel_time + expected_stop_time
        
        # O tempo total deve incluir pelo menos o tempo da primeira rota
        assert travel_time >= expected_route1_time
    
    def test_calculate_average_transfers(self):
        """Testa cálculo de transferências médias."""
        routes = self.create_test_routes()
        avg_transfers = self.calculator._calculate_average_transfers(routes)
        
        # Com rotas que compartilham paradas, deve haver algumas transferências diretas
        assert avg_transfers >= 0.0
        assert avg_transfers <= 1.0  # Máximo 1 transferência para rotas simples
    
    def test_calculate_average_transfers_no_overlap(self):
        """Testa transferências com rotas sem sobreposição."""
        routes = [
            self.create_simple_route([1, 2], [5.0], [20]),
            self.create_simple_route([3, 4], [6.0], [25])
        ]
        avg_transfers = self.calculator._calculate_average_transfers(routes)
        
        # Rotas sem sobreposição devem requerer transferências
        assert avg_transfers > 0.0
    
    def test_calculate_geographic_coverage_basic(self):
        """Testa cálculo básico de cobertura geográfica."""
        routes = self.create_test_routes()
        coverage = self.calculator._calculate_geographic_coverage(routes)
        
        # Deve estimar baseado no número de paradas únicas
        unique_stops = set()
        for route in routes:
            unique_stops.update(route.stops)
        
        expected_coverage = len(unique_stops) * 1.0  # 1 km² por parada
        assert coverage == expected_coverage
    
    def test_calculate_geographic_coverage_with_coordinates(self):
        """Testa cobertura geográfica com coordenadas reais."""
        routes = [
            self.create_simple_route([1, 2, 3], [5.0, 7.0], [20, 30]),
            self.create_simple_route([3, 4, 5], [6.0, 8.0], [25, 35])
        ]
        
        coverage = self.calculator_with_geo._calculate_geographic_coverage(routes)
        
        # Com coordenadas reais, deve calcular área baseada em posições
        assert coverage > 0
        # Deve ser diferente da estimativa básica
        basic_coverage = len(set([1, 2, 3, 4, 5])) * 1.0
        assert coverage != basic_coverage
    
    def test_calculate_load_balancing_perfect(self):
        """Testa balanceamento perfeito de carga."""
        routes = [
            self.create_simple_route([1, 2], [5.0], [50]),
            self.create_simple_route([3, 4], [6.0], [50]),
            self.create_simple_route([5, 6], [7.0], [50])
        ]
        
        balancing = self.calculator._calculate_load_balancing_index(routes)
        assert balancing == 1.0  # Perfeitamente balanceado
    
    def test_calculate_load_balancing_unbalanced(self):
        """Testa balanceamento desigual de carga."""
        routes = [
            self.create_simple_route([1, 2], [5.0], [10]),   # Baixa carga
            self.create_simple_route([3, 4], [6.0], [50]),   # Carga média
            self.create_simple_route([5, 6], [7.0], [90])    # Alta carga
        ]
        
        balancing = self.calculator._calculate_load_balancing_index(routes)
        assert 0.0 < balancing < 1.0  # Desbalanceado
    
    def test_calculate_energy_efficiency(self):
        """Testa cálculo de eficiência energética."""
        routes = self.create_test_routes()
        efficiency = self.calculator._calculate_energy_efficiency(routes)
        
        # Calcular manualmente
        total_distance = sum(route.total_distance for route in routes)
        total_passengers = sum(route.total_passengers for route in routes)
        expected_efficiency = total_distance / total_passengers
        
        assert abs(efficiency - expected_efficiency) < 0.001
        assert efficiency > 0
    
    def test_calculate_energy_efficiency_no_passengers(self):
        """Testa eficiência energética sem passageiros."""
        routes = [
            self.create_simple_route([1, 2], [5.0], [0])  # Sem passageiros
        ]
        
        efficiency = self.calculator._calculate_energy_efficiency(routes)
        assert efficiency == float('inf')  # Infinito quando não há passageiros
    
    def test_calculate_accessibility_index(self):
        """Testa cálculo do índice de acessibilidade."""
        routes = self.create_test_routes()
        accessibility = self.calculator._calculate_accessibility_index(routes)
        
        # Deve ser paradas únicas / área de cobertura
        unique_stops = set()
        for route in routes:
            unique_stops.update(route.stops)
        
        coverage = self.calculator._calculate_geographic_coverage(routes)
        expected_accessibility = len(unique_stops) / coverage
        
        assert abs(accessibility - expected_accessibility) < 0.001
        assert accessibility > 0
    
    def test_calculate_service_frequency(self):
        """Testa cálculo de frequência de serviço."""
        routes = self.create_test_routes()
        frequency = self.calculator._calculate_service_frequency(routes)
        
        assert frequency > 0
        # Frequência deve ser razoável (não muito alta nem muito baixa)
        assert 0.1 <= frequency <= 100.0
    
    def test_calculate_route_overlap_percentage(self):
        """Testa cálculo de sobreposição de rotas."""
        # Rotas com sobreposição
        routes = [
            self.create_simple_route([1, 2, 3], [5.0, 7.0], [20, 30]),
            self.create_simple_route([2, 3, 4], [6.0, 8.0], [25, 35])  # Compartilha 2 e 3
        ]
        
        overlap = self.calculator._calculate_route_overlap_percentage(routes)
        
        # Paradas 2 e 3 são compartilhadas, paradas 1 e 4 são únicas
        # 2 de 4 paradas únicas são compartilhadas = 50%
        assert overlap == 50.0
    
    def test_calculate_route_overlap_no_overlap(self):
        """Testa sobreposição sem paradas compartilhadas."""
        routes = [
            self.create_simple_route([1, 2], [5.0], [20]),
            self.create_simple_route([3, 4], [6.0], [25])
        ]
        
        overlap = self.calculator._calculate_route_overlap_percentage(routes)
        assert overlap == 0.0  # Sem sobreposição
    
    def test_calculate_solution_metrics(self):
        """Testa cálculo de métricas para uma solução completa."""
        routes = self.create_test_routes()
        solution = Solution(
            routes=routes,
            total_vehicles=len(routes),
            total_distance=sum(r.total_distance for r in routes),
            total_passengers_served=sum(r.total_passengers for r in routes),
            fitness_time=100.0,
            fitness_vehicle=3.0,
            is_feasible=True
        )
        
        metrics = self.calculator.calculate_solution_metrics(solution)
        
        assert isinstance(metrics, DomainMetrics)
        assert metrics.estimated_travel_time > 0
        assert metrics.geographic_coverage > 0
        assert metrics.energy_efficiency > 0
    
    def test_get_transportation_summary(self):
        """Testa geração de resumo de métricas de transporte."""
        routes = self.create_test_routes()
        metrics = self.calculator.calculate_transportation_metrics(routes)
        summary = self.calculator.get_transportation_summary(metrics)
        
        assert "Métricas de Transporte Público" in summary
        assert "Eficiência Operacional" in summary
        assert "Cobertura e Acessibilidade" in summary
        assert "Eficiência de Recursos" in summary
        assert isinstance(summary, str)
        assert len(summary) > 200  # Resumo substancial
    
    def test_analyze_route_connectivity(self):
        """Testa análise de conectividade entre rotas."""
        routes = self.create_test_routes()
        connectivity = self.calculator.analyze_route_connectivity(routes)
        
        assert 'connectivity_index' in connectivity
        assert 'transfer_points' in connectivity
        assert 'isolated_routes' in connectivity
        assert 'network_efficiency' in connectivity
        
        # Verificar valores razoáveis
        assert 0.0 <= connectivity['connectivity_index'] <= 1.0
        assert connectivity['transfer_points'] >= 0
        assert connectivity['isolated_routes'] >= 0
        assert 0.0 <= connectivity['network_efficiency'] <= 1.0
    
    def test_analyze_route_connectivity_single_route(self):
        """Testa conectividade com uma única rota."""
        routes = [self.create_simple_route([1, 2, 3], [5.0, 7.0], [20, 30])]
        connectivity = self.calculator.analyze_route_connectivity(routes)
        
        assert connectivity['connectivity_index'] == 0.0
        assert connectivity['transfer_points'] == 0
        assert connectivity['isolated_routes'] == 1
        assert connectivity['network_efficiency'] == 1.0  # 100% eficiente (sem redundância)
    
    def test_analyze_route_connectivity_no_routes(self):
        """Testa conectividade sem rotas."""
        connectivity = self.calculator.analyze_route_connectivity([])
        
        assert connectivity['connectivity_index'] == 0.0
        assert connectivity['transfer_points'] == 0
        assert connectivity['isolated_routes'] == 0
        assert connectivity['network_efficiency'] == 0.0
    
    def test_custom_parameters(self):
        """Testa calculadora com parâmetros customizados."""
        custom_calculator = TransportationMetrics(
            average_speed_kmh=40.0,
            transfer_time_minutes=2.0
        )
        
        routes = [self.create_simple_route([1, 2], [20.0], [50])]
        travel_time = custom_calculator._estimate_total_travel_time(routes)
        
        # Com velocidade maior, tempo deve ser menor
        default_time = self.calculator._estimate_total_travel_time(routes)
        assert travel_time < default_time
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Criar rota com dados problemáticos
        problematic_route = Route(
            stops=[1],
            distances=[],
            passenger_load=[],
            total_distance=0.0,
            total_passengers=0,
            is_valid=True
        )
        
        # O calculador deve lidar com isso graciosamente
        try:
            metrics = self.calculator.calculate_transportation_metrics([problematic_route])
            assert isinstance(metrics, DomainMetrics)
        except MetricsCalculationError:
            # Ou lançar exceção apropriada
            pass
    
    def test_real_world_scenario(self):
        """Teste com cenário realista de transporte público."""
        # Simular sistema de ônibus de uma cidade pequena
        routes = [
            # Rota Centro-Norte
            self.create_simple_route([1, 2, 3, 4, 5], [2.0, 1.5, 2.5, 1.8], [15, 20, 25, 18]),
            # Rota Centro-Sul  
            self.create_simple_route([1, 6, 7, 8], [1.8, 2.2, 1.9], [12, 22, 16]),
            # Rota Leste-Oeste
            self.create_simple_route([3, 9, 10, 7], [3.0, 2.8, 2.1], [18, 14, 20])
        ]
        
        metrics = self.calculator.calculate_transportation_metrics(routes)
        
        # Verificar que todas as métricas são calculadas
        assert metrics.estimated_travel_time > 0
        assert metrics.average_transfers >= 0
        assert metrics.geographic_coverage > 0
        assert 0 <= metrics.load_balancing_index <= 1
        assert metrics.energy_efficiency > 0
        assert metrics.accessibility_index > 0
        assert metrics.service_frequency > 0
        assert 0 <= metrics.route_overlap_percentage <= 100
        
        # Verificar conectividade
        connectivity = self.calculator.analyze_route_connectivity(routes)
        assert connectivity['transfer_points'] > 0  # Deve haver pontos de transferência
        assert connectivity['isolated_routes'] < len(routes)  # Nem todas isoladas
    
    def test_edge_cases(self):
        """Testa casos extremos."""
        # Rota com uma única parada
        single_stop_route = Route(
            stops=[1],
            distances=[],
            passenger_load=[],
            total_distance=0.0,
            total_passengers=0,
            is_valid=True
        )
        
        metrics = self.calculator.calculate_transportation_metrics([single_stop_route])
        assert isinstance(metrics, DomainMetrics)
        
        # Rota muito longa
        long_route = self.create_simple_route(
            list(range(1, 101)),  # 100 paradas
            [1.0] * 99,           # 99 segmentos
            [10] * 99             # Carga constante
        )
        
        metrics_long = self.calculator.calculate_transportation_metrics([long_route])
        assert metrics_long.estimated_travel_time > 0
        assert metrics_long.geographic_coverage > 0