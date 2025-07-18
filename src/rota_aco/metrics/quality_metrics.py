"""
Sistema de avaliação de qualidade de rotas para o projeto Rota_ACO.

Este módulo implementa o RouteQualityEvaluator que calcula métricas de qualidade
das rotas geradas pelos algoritmos ACO, incluindo validação de restrições,
cobertura de demanda e eficiência de utilização.
"""

from typing import List, Dict, Set, Optional, Tuple
import statistics
import math
from .data_models import Route, Solution, RouteQualityMetrics
from .exceptions import MetricsCalculationError


class RouteQualityEvaluator:
    """
    Avaliador de qualidade de rotas que calcula métricas abrangentes
    para validação científica dos algoritmos ACO.
    """
    
    def __init__(self, capacity_limit: int = 70, graph_data: Optional[Dict] = None):
        """
        Inicializa o avaliador de qualidade.
        
        Args:
            capacity_limit: Limite de capacidade por veículo (padrão: 70 passageiros)
            graph_data: Dados do grafo para cálculos de cobertura (opcional)
        """
        self.capacity_limit = capacity_limit
        self.graph_data = graph_data or {}
        
    def evaluate_routes(self, routes: List[Route]) -> RouteQualityMetrics:
        """
        Avalia a qualidade de um conjunto de rotas.
        
        Args:
            routes: Lista de rotas para avaliar
            
        Returns:
            RouteQualityMetrics com todas as métricas calculadas
            
        Raises:
            MetricsCalculationError: Se houver erro no cálculo das métricas
        """
        try:
            if not routes:
                return self._empty_metrics()
            
            # Calcular métricas individuais
            valid_percentage = self._calculate_valid_routes_percentage(routes)
            demand_coverage = self._calculate_demand_coverage_percentage(routes)
            utilization_efficiency = self._calculate_vehicle_utilization_efficiency(routes)
            capacity_violations = self._count_capacity_violations(routes)
            opposite_violations = self._count_opposite_stops_violations(routes)
            avg_length, length_variance = self._calculate_route_length_metrics(routes)
            load_balancing = self._calculate_load_balancing_index(routes)
            
            return RouteQualityMetrics(
                valid_routes_percentage=valid_percentage,
                demand_coverage_percentage=demand_coverage,
                vehicle_utilization_efficiency=utilization_efficiency,
                capacity_violations=capacity_violations,
                opposite_stops_violations=opposite_violations,
                average_route_length=avg_length,
                route_length_variance=length_variance,
                load_balancing_index=load_balancing
            )
            
        except Exception as e:
            raise MetricsCalculationError(f"Erro ao calcular métricas de qualidade: {str(e)}")
    
    def evaluate_solution(self, solution: Solution) -> RouteQualityMetrics:
        """
        Avalia a qualidade de uma solução completa.
        
        Args:
            solution: Solução para avaliar
            
        Returns:
            RouteQualityMetrics da solução
        """
        return self.evaluate_routes(solution.routes)
    
    def _calculate_valid_routes_percentage(self, routes: List[Route]) -> float:
        """Calcula a porcentagem de rotas válidas."""
        if not routes:
            return 0.0
        
        valid_count = sum(1 for route in routes if route.is_valid)
        return (valid_count / len(routes)) * 100.0
    
    def _calculate_demand_coverage_percentage(self, routes: List[Route]) -> float:
        """
        Calcula a porcentagem de cobertura de demanda.
        
        Considera todas as paradas únicas visitadas pelas rotas.
        """
        if not routes:
            return 0.0
        
        # Coletar todas as paradas visitadas
        visited_stops: Set[int] = set()
        for route in routes:
            visited_stops.update(route.stops)
        
        # Se temos dados do grafo, usar o total de paradas do grafo
        if 'total_stops' in self.graph_data:
            total_stops = self.graph_data['total_stops']
        else:
            # Caso contrário, assumir que visitamos todas as paradas possíveis
            total_stops = len(visited_stops) if visited_stops else 1
        
        return (len(visited_stops) / total_stops) * 100.0
    
    def _calculate_vehicle_utilization_efficiency(self, routes: List[Route]) -> float:
        """
        Calcula a eficiência de utilização dos veículos.
        
        Baseado na ocupação média em relação à capacidade máxima.
        """
        if not routes:
            return 0.0
        
        total_utilization = 0.0
        valid_routes = 0
        
        for route in routes:
            if route.passenger_load:
                # Calcular ocupação média da rota
                avg_load = statistics.mean(route.passenger_load)
                utilization = (avg_load / self.capacity_limit) * 100.0
                total_utilization += min(utilization, 100.0)  # Cap at 100%
                valid_routes += 1
        
        return total_utilization / valid_routes if valid_routes > 0 else 0.0
    
    def _count_capacity_violations(self, routes: List[Route]) -> int:
        """Conta violações de capacidade (> 70 passageiros)."""
        violations = 0
        
        for route in routes:
            for load in route.passenger_load:
                if load > self.capacity_limit:
                    violations += 1
        
        return violations
    
    def _count_opposite_stops_violations(self, routes: List[Route]) -> int:
        """
        Conta violações de paradas opostas na mesma rota.
        
        Assume que paradas opostas têm IDs que diferem por um padrão específico.
        Esta implementação é simplificada - pode ser refinada com dados reais do grafo.
        """
        violations = 0
        
        for route in routes:
            stops_set = set(route.stops)
            
            # Verificar se há paradas opostas na mesma rota
            # Implementação simplificada: assumir que paradas ímpares e pares consecutivas são opostas
            for stop in route.stops:
                opposite_stop = stop + 1 if stop % 2 == 0 else stop - 1
                if opposite_stop in stops_set and opposite_stop != stop:
                    violations += 1
        
        return violations // 2  # Dividir por 2 para não contar duas vezes a mesma violação
    
    def _calculate_route_length_metrics(self, routes: List[Route]) -> Tuple[float, float]:
        """
        Calcula métricas de comprimento das rotas.
        
        Returns:
            Tuple com (comprimento médio, variância do comprimento)
        """
        if not routes:
            return 0.0, 0.0
        
        lengths = [route.total_distance for route in routes]
        
        avg_length = statistics.mean(lengths)
        variance = statistics.variance(lengths) if len(lengths) > 1 else 0.0
        
        return avg_length, variance
    
    def _calculate_load_balancing_index(self, routes: List[Route]) -> float:
        """
        Calcula índice de balanceamento de carga entre veículos.
        
        Valores próximos a 1.0 indicam melhor balanceamento.
        Valores menores indicam desbalanceamento.
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
    
    def _empty_metrics(self) -> RouteQualityMetrics:
        """Retorna métricas vazias para casos sem rotas."""
        return RouteQualityMetrics(
            valid_routes_percentage=0.0,
            demand_coverage_percentage=0.0,
            vehicle_utilization_efficiency=0.0,
            capacity_violations=0,
            opposite_stops_violations=0,
            average_route_length=0.0,
            route_length_variance=0.0,
            load_balancing_index=0.0
        )
    
    def validate_route(self, route: Route) -> Dict[str, bool]:
        """
        Valida uma rota individual contra todas as restrições.
        
        Args:
            route: Rota para validar
            
        Returns:
            Dict com resultados de validação para cada restrição
        """
        validation_results = {
            'capacity_valid': all(load <= self.capacity_limit for load in route.passenger_load),
            'structure_valid': len(route.stops) == len(route.distances) + 1,
            'load_consistency': len(route.passenger_load) == len(route.distances),
            'positive_distances': all(dist >= 0 for dist in route.distances),
            'non_empty': len(route.stops) > 0
        }
        
        # Verificar paradas opostas
        stops_set = set(route.stops)
        has_opposite_violations = False
        for stop in route.stops:
            opposite_stop = stop + 1 if stop % 2 == 0 else stop - 1
            if opposite_stop in stops_set and opposite_stop != stop:
                has_opposite_violations = True
                break
        
        validation_results['no_opposite_stops'] = not has_opposite_violations
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results
    
    def get_quality_summary(self, metrics: RouteQualityMetrics) -> str:
        """
        Gera um resumo textual das métricas de qualidade.
        
        Args:
            metrics: Métricas calculadas
            
        Returns:
            String com resumo formatado
        """
        summary = f"""
Resumo de Qualidade das Rotas:
=============================

Validação:
- Rotas válidas: {metrics.valid_routes_percentage:.1f}%
- Violações de capacidade: {metrics.capacity_violations}
- Violações de paradas opostas: {metrics.opposite_stops_violations}

Cobertura e Eficiência:
- Cobertura de demanda: {metrics.demand_coverage_percentage:.1f}%
- Eficiência de utilização: {metrics.vehicle_utilization_efficiency:.1f}%
- Índice de balanceamento: {metrics.load_balancing_index:.3f}

Características das Rotas:
- Comprimento médio: {metrics.average_route_length:.2f}
- Variância do comprimento: {metrics.route_length_variance:.2f}
"""
        return summary.strip()