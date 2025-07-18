"""
Sistema de métricas específicas do domínio de transporte público.

Este módulo implementa o TransportationMetrics que calcula métricas
específicas para avaliação da aplicabilidade prática das soluções
de roteamento de transporte público geradas pelos algoritmos ACO.
"""

from typing import List, Dict, Optional, Tuple, Set
import math
import statistics
from .data_models import Route, Solution, DomainMetrics
from .exceptions import MetricsCalculationError


class TransportationMetrics:
    """
    Calculador de métricas específicas do domínio de transporte público
    para avaliação da aplicabilidade prática das soluções ACO.
    """
    
    def __init__(self, graph_data: Optional[Dict] = None, 
                 average_speed_kmh: float = 30.0,
                 transfer_time_minutes: float = 5.0):
        """
        Inicializa o calculador de métricas de transporte.
        
        Args:
            graph_data: Dados do grafo com informações geográficas (opcional)
            average_speed_kmh: Velocidade média dos veículos em km/h (padrão: 30)
            transfer_time_minutes: Tempo médio de transferência em minutos (padrão: 5)
        """
        self.graph_data = graph_data or {}
        self.average_speed_kmh = average_speed_kmh
        self.transfer_time_minutes = transfer_time_minutes
        
        # Dados padrão se não fornecidos
        self.default_stop_coordinates = {}
        self.default_area_km2 = 100.0  # Área padrão em km²
    
    def calculate_transportation_metrics(self, routes: List[Route]) -> DomainMetrics:
        """
        Calcula todas as métricas de transporte para um conjunto de rotas.
        
        Args:
            routes: Lista de rotas para analisar
            
        Returns:
            DomainMetrics com todas as métricas calculadas
            
        Raises:
            MetricsCalculationError: Se houver erro no cálculo das métricas
        """
        try:
            if not routes:
                return self._empty_domain_metrics()
            
            # Calcular métricas individuais
            travel_time = self._estimate_total_travel_time(routes)
            avg_transfers = self._calculate_average_transfers(routes)
            geographic_coverage = self._calculate_geographic_coverage(routes)
            load_balancing = self._calculate_load_balancing_index(routes)
            energy_efficiency = self._calculate_energy_efficiency(routes)
            accessibility_index = self._calculate_accessibility_index(routes)
            service_frequency = self._calculate_service_frequency(routes)
            route_overlap = self._calculate_route_overlap_percentage(routes)
            
            return DomainMetrics(
                estimated_travel_time=travel_time,
                average_transfers=avg_transfers,
                geographic_coverage=geographic_coverage,
                load_balancing_index=load_balancing,
                energy_efficiency=energy_efficiency,
                accessibility_index=accessibility_index,
                service_frequency=service_frequency,
                route_overlap_percentage=route_overlap
            )
            
        except Exception as e:
            raise MetricsCalculationError(f"Erro ao calcular métricas de transporte: {str(e)}")
    
    def calculate_solution_metrics(self, solution: Solution) -> DomainMetrics:
        """
        Calcula métricas de transporte para uma solução completa.
        
        Args:
            solution: Solução para analisar
            
        Returns:
            DomainMetrics da solução
        """
        return self.calculate_transportation_metrics(solution.routes)
    
    def _estimate_total_travel_time(self, routes: List[Route]) -> float:
        """
        Estima o tempo total de viagem para todas as rotas.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Tempo total estimado em minutos
        """
        if not routes:
            return 0.0
        
        total_time_minutes = 0.0
        
        for route in routes:
            # Calcular tempo de viagem baseado na distância e velocidade média
            travel_time = (route.total_distance / self.average_speed_kmh) * 60  # Converter para minutos
            
            # Adicionar tempo de paradas (assumindo 1 minuto por parada)
            stop_time = len(route.stops) * 1.0
            
            # Tempo total da rota
            route_time = travel_time + stop_time
            total_time_minutes += route_time
        
        return total_time_minutes
    
    def _calculate_average_transfers(self, routes: List[Route]) -> float:
        """
        Calcula o número médio de transferências necessárias.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Número médio de transferências por viagem
        """
        if not routes or len(routes) <= 1:
            return 0.0
        
        # Calcular conectividade entre rotas
        total_connections = 0
        total_possible_trips = 0
        
        # Para cada par de paradas, verificar quantas transferências são necessárias
        all_stops = set()
        for route in routes:
            all_stops.update(route.stops)
        
        if len(all_stops) < 2:
            return 0.0
        
        # Criar mapa de paradas para rotas
        stop_to_routes = {}
        for i, route in enumerate(routes):
            for stop in route.stops:
                if stop not in stop_to_routes:
                    stop_to_routes[stop] = []
                stop_to_routes[stop].append(i)
        
        # Calcular transferências médias necessárias
        transfer_counts = []
        stops_list = list(all_stops)
        
        for i in range(len(stops_list)):
            for j in range(i + 1, len(stops_list)):
                stop_a, stop_b = stops_list[i], stops_list[j]
                
                # Verificar se estão na mesma rota (0 transferências)
                routes_a = set(stop_to_routes.get(stop_a, []))
                routes_b = set(stop_to_routes.get(stop_b, []))
                
                if routes_a & routes_b:  # Interseção não vazia
                    transfer_counts.append(0)
                else:
                    # Precisam de pelo menos 1 transferência
                    transfer_counts.append(1)
        
        return statistics.mean(transfer_counts) if transfer_counts else 0.0
    
    def _calculate_geographic_coverage(self, routes: List[Route]) -> float:
        """
        Calcula a cobertura geográfica das rotas.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Área de cobertura em km²
        """
        if not routes:
            return 0.0
        
        # Se temos dados geográficos reais, usar eles
        if 'stop_coordinates' in self.graph_data:
            return self._calculate_real_geographic_coverage(routes)
        
        # Caso contrário, estimar baseado no número de paradas únicas
        unique_stops = set()
        for route in routes:
            unique_stops.update(route.stops)
        
        # Estimar área baseada no número de paradas
        # Assumindo que cada parada cobre aproximadamente 1 km²
        estimated_coverage = len(unique_stops) * 1.0
        
        return min(estimated_coverage, self.default_area_km2)
    
    def _calculate_real_geographic_coverage(self, routes: List[Route]) -> float:
        """
        Calcula cobertura geográfica real usando coordenadas.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Área real de cobertura em km²
        """
        coordinates = self.graph_data.get('stop_coordinates', {})
        
        # Coletar todas as coordenadas das paradas servidas
        served_coordinates = []
        for route in routes:
            for stop in route.stops:
                if stop in coordinates:
                    served_coordinates.append(coordinates[stop])
        
        if len(served_coordinates) < 3:
            return len(served_coordinates) * 1.0  # Estimativa básica
        
        # Calcular área do polígono convexo (simplificado)
        # Para implementação completa, usaria algoritmo de hull convexo
        min_lat = min(coord[0] for coord in served_coordinates)
        max_lat = max(coord[0] for coord in served_coordinates)
        min_lon = min(coord[1] for coord in served_coordinates)
        max_lon = max(coord[1] for coord in served_coordinates)
        
        # Aproximar área do retângulo em km²
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon
        
        # Conversão aproximada de graus para km (1 grau ≈ 111 km)
        area_km2 = (lat_diff * 111) * (lon_diff * 111 * math.cos(math.radians((min_lat + max_lat) / 2)))
        
        return abs(area_km2)
    
    def _calculate_load_balancing_index(self, routes: List[Route]) -> float:
        """
        Calcula índice de balanceamento de carga entre veículos.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Índice de balanceamento (0-1, onde 1 é perfeitamente balanceado)
        """
        if not routes:
            return 0.0
        
        # Calcular carga total de cada rota
        route_loads = [route.total_passengers for route in routes]
        
        if not route_loads or all(load == 0 for load in route_loads):
            return 1.0  # Perfeitamente balanceado se todas as rotas têm carga zero
        
        # Calcular coeficiente de variação (inverso do balanceamento)
        mean_load = statistics.mean(route_loads)
        if mean_load == 0:
            return 1.0
        
        std_load = statistics.stdev(route_loads) if len(route_loads) > 1 else 0.0
        coefficient_of_variation = std_load / mean_load
        
        # Converter para índice de balanceamento (0 a 1, onde 1 é perfeitamente balanceado)
        balancing_index = 1.0 / (1.0 + coefficient_of_variation)
        
        return balancing_index
    
    def _calculate_energy_efficiency(self, routes: List[Route]) -> float:
        """
        Calcula eficiência energética (distância por passageiro).
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Distância média por passageiro em km
        """
        if not routes:
            return 0.0
        
        total_distance = sum(route.total_distance for route in routes)
        total_passengers = sum(route.total_passengers for route in routes)
        
        if total_passengers == 0:
            return float('inf')  # Infinito se não há passageiros
        
        return total_distance / total_passengers
    
    def _calculate_accessibility_index(self, routes: List[Route]) -> float:
        """
        Calcula índice de acessibilidade (paradas por km²).
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Número de paradas por km²
        """
        if not routes:
            return 0.0
        
        # Contar paradas únicas
        unique_stops = set()
        for route in routes:
            unique_stops.update(route.stops)
        
        # Calcular área de cobertura
        coverage_area = self._calculate_geographic_coverage(routes)
        
        if coverage_area == 0:
            return 0.0
        
        return len(unique_stops) / coverage_area
    
    def _calculate_service_frequency(self, routes: List[Route]) -> float:
        """
        Calcula frequência de serviço estimada.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Frequência média de serviço (viagens por hora)
        """
        if not routes:
            return 0.0
        
        # Estimar frequência baseada no número de rotas e tempo de ciclo
        total_routes = len(routes)
        
        # Calcular tempo médio de ciclo (ida e volta)
        avg_route_time = 0.0
        for route in routes:
            # Tempo de viagem + tempo de paradas
            travel_time = (route.total_distance / self.average_speed_kmh) * 60
            stop_time = len(route.stops) * 1.0
            cycle_time = (travel_time + stop_time) * 2  # Ida e volta
            avg_route_time += cycle_time
        
        if total_routes == 0:
            return 0.0
        
        avg_cycle_time_hours = (avg_route_time / total_routes) / 60.0
        
        if avg_cycle_time_hours == 0:
            return 0.0
        
        # Frequência = número de veículos / tempo de ciclo
        # Assumindo 1 veículo por rota
        frequency = total_routes / avg_cycle_time_hours
        
        return frequency
    
    def _calculate_route_overlap_percentage(self, routes: List[Route]) -> float:
        """
        Calcula porcentagem de sobreposição entre rotas.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Porcentagem de paradas que são servidas por múltiplas rotas
        """
        if len(routes) <= 1:
            return 0.0
        
        # Contar quantas rotas servem cada parada
        stop_count = {}
        for route in routes:
            for stop in route.stops:
                stop_count[stop] = stop_count.get(stop, 0) + 1
        
        if not stop_count:
            return 0.0
        
        # Calcular porcentagem de paradas com sobreposição
        overlapping_stops = sum(1 for count in stop_count.values() if count > 1)
        total_unique_stops = len(stop_count)
        
        return (overlapping_stops / total_unique_stops) * 100.0
    
    def _empty_domain_metrics(self) -> DomainMetrics:
        """Retorna métricas vazias para casos sem rotas."""
        return DomainMetrics(
            estimated_travel_time=0.0,
            average_transfers=0.0,
            geographic_coverage=0.0,
            load_balancing_index=0.0,
            energy_efficiency=0.0,
            accessibility_index=0.0,
            service_frequency=0.0,
            route_overlap_percentage=0.0
        )
    
    def get_transportation_summary(self, metrics: DomainMetrics) -> str:
        """
        Gera um resumo textual das métricas de transporte.
        
        Args:
            metrics: Métricas calculadas
            
        Returns:
            String com resumo formatado
        """
        summary = f"""
Métricas de Transporte Público:
==============================

Eficiência Operacional:
- Tempo Total de Viagem: {metrics.estimated_travel_time:.1f} minutos
- Transferências Médias: {metrics.average_transfers:.2f}
- Frequência de Serviço: {metrics.service_frequency:.1f} viagens/hora

Cobertura e Acessibilidade:
- Cobertura Geográfica: {metrics.geographic_coverage:.2f} km²
- Índice de Acessibilidade: {metrics.accessibility_index:.2f} paradas/km²
- Sobreposição de Rotas: {metrics.route_overlap_percentage:.1f}%

Eficiência de Recursos:
- Eficiência Energética: {metrics.energy_efficiency:.2f} km/passageiro
- Balanceamento de Carga: {metrics.load_balancing_index:.3f}
"""
        return summary.strip()
    
    def analyze_route_connectivity(self, routes: List[Route]) -> Dict[str, float]:
        """
        Analisa a conectividade entre rotas do sistema.
        
        Args:
            routes: Lista de rotas
            
        Returns:
            Dict com métricas de conectividade
        """
        if len(routes) <= 1:
            # Para uma única rota, a eficiência é 1.0 (sem redundância)
            network_eff = 1.0 if len(routes) == 1 else 0.0
            return {
                'connectivity_index': 0.0,
                'transfer_points': 0,
                'isolated_routes': len(routes),
                'network_efficiency': network_eff
            }
        
        # Encontrar pontos de transferência (paradas compartilhadas)
        stop_routes = {}
        for i, route in enumerate(routes):
            for stop in route.stops:
                if stop not in stop_routes:
                    stop_routes[stop] = []
                stop_routes[stop].append(i)
        
        transfer_points = sum(1 for routes_list in stop_routes.values() if len(routes_list) > 1)
        
        # Calcular rotas isoladas (sem conexões)
        connected_routes = set()
        for routes_list in stop_routes.values():
            if len(routes_list) > 1:
                connected_routes.update(routes_list)
        
        isolated_routes = len(routes) - len(connected_routes)
        
        # Índice de conectividade (baseado em paradas compartilhadas)
        total_unique_stops = len(stop_routes)
        shared_stops = len([routes_list for routes_list in stop_routes.values() if len(routes_list) > 1])
        connectivity_index = shared_stops / total_unique_stops if total_unique_stops > 0 else 0.0
        
        # Eficiência da rede
        total_stops = sum(len(route.stops) for route in routes)
        unique_stops = len(stop_routes)
        network_efficiency = unique_stops / total_stops if total_stops > 0 else 0.0
        
        return {
            'connectivity_index': connectivity_index,
            'transfer_points': transfer_points,
            'isolated_routes': isolated_routes,
            'network_efficiency': network_efficiency
        }