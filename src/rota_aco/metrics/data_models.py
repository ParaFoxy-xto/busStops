"""
Modelos de dados para o sistema de métricas expandido.

Este módulo define todas as estruturas de dados utilizadas para armazenar
e processar informações sobre execuções ACO, rotas, soluções e métricas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid


@dataclass
class Route:
    """Representa uma rota individual no sistema de transporte."""
    stops: List[int]  # IDs das paradas visitadas
    distances: List[float]  # Distâncias entre paradas consecutivas
    passenger_load: List[int]  # Carga de passageiros em cada segmento
    total_distance: float
    total_passengers: int
    is_valid: bool
    capacity_violations: int = 0
    opposite_stops_violations: int = 0
    
    def __post_init__(self):
        """Validação básica dos dados da rota."""
        if len(self.stops) != len(self.distances) + 1:
            raise ValueError("Número de paradas deve ser distâncias + 1")
        if len(self.passenger_load) != len(self.distances):
            raise ValueError("Carga de passageiros deve ter mesmo tamanho que distâncias")


@dataclass
class Solution:
    """Representa uma solução completa (conjunto de rotas)."""
    routes: List[Route]
    total_vehicles: int
    total_distance: float
    total_passengers_served: int
    fitness_time: float
    fitness_vehicle: float
    is_feasible: bool
    generation_time: float = 0.0
    
    def __post_init__(self):
        """Calcula métricas derivadas da solução."""
        if not self.routes:
            self.total_vehicles = 0
            self.total_distance = 0.0
            self.total_passengers_served = 0
        else:
            self.total_vehicles = len(self.routes)
            self.total_distance = sum(route.total_distance for route in self.routes)
            self.total_passengers_served = sum(route.total_passengers for route in self.routes)


@dataclass
class IterationData:
    """Dados de uma iteração específica do algoritmo ACO."""
    iteration: int
    best_fitness: float
    avg_fitness: float
    population_variance: float
    best_solution: Solution
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionData:
    """Dados completos de uma execução ACO."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm_type: str = ""  # 'ACS-TIME' ou 'ACS-VEHICLE'
    config: Dict[str, Any] = field(default_factory=dict)
    routes: List[Route] = field(default_factory=list)
    iterations_data: List[IterationData] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    final_solution: Optional[Solution] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class RouteQualityMetrics:
    """Métricas de qualidade das rotas geradas."""
    valid_routes_percentage: float
    demand_coverage_percentage: float
    vehicle_utilization_efficiency: float
    capacity_violations: int
    opposite_stops_violations: int
    average_route_length: float
    route_length_variance: float
    load_balancing_index: float
    
    def to_dict(self) -> Dict[str, float]:
        """Converte métricas para dicionário."""
        return {
            'valid_routes_percentage': self.valid_routes_percentage,
            'demand_coverage_percentage': self.demand_coverage_percentage,
            'vehicle_utilization_efficiency': self.vehicle_utilization_efficiency,
            'capacity_violations': self.capacity_violations,
            'opposite_stops_violations': self.opposite_stops_violations,
            'average_route_length': self.average_route_length,
            'route_length_variance': self.route_length_variance,
            'load_balancing_index': self.load_balancing_index
        }


@dataclass
class ConvergenceMetrics:
    """Métricas de análise de convergência."""
    convergence_point: int
    final_stability: float
    improvement_rate: float
    plateau_detection: bool
    convergence_speed: float
    total_iterations: int
    best_fitness_evolution: List[float] = field(default_factory=list)
    avg_fitness_evolution: List[float] = field(default_factory=list)
    variance_evolution: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte métricas para dicionário."""
        return {
            'convergence_point': self.convergence_point,
            'final_stability': self.final_stability,
            'improvement_rate': self.improvement_rate,
            'plateau_detection': self.plateau_detection,
            'convergence_speed': self.convergence_speed,
            'total_iterations': self.total_iterations
        }


@dataclass
class ComparativeMetrics:
    """Métricas comparativas entre múltiplas execuções."""
    total_executions: int
    successful_executions: int
    success_rate: float
    avg_execution_time: float
    std_execution_time: float
    avg_best_fitness: float
    std_best_fitness: float
    median_best_fitness: float
    avg_convergence_iterations: float
    relative_efficiency: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Converte métricas para dicionário."""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.success_rate,
            'avg_execution_time': self.avg_execution_time,
            'std_execution_time': self.std_execution_time,
            'avg_best_fitness': self.avg_best_fitness,
            'std_best_fitness': self.std_best_fitness,
            'median_best_fitness': self.median_best_fitness,
            'avg_convergence_iterations': self.avg_convergence_iterations,
            'relative_efficiency': self.relative_efficiency
        }


@dataclass
class DomainMetrics:
    """Métricas específicas do domínio de transporte público."""
    estimated_travel_time: float
    average_transfers: float
    geographic_coverage: float
    load_balancing_index: float
    energy_efficiency: float  # distance per passenger
    accessibility_index: float  # stops per km²
    service_frequency: float
    route_overlap_percentage: float
    
    def to_dict(self) -> Dict[str, float]:
        """Converte métricas para dicionário."""
        return {
            'estimated_travel_time': self.estimated_travel_time,
            'average_transfers': self.average_transfers,
            'geographic_coverage': self.geographic_coverage,
            'load_balancing_index': self.load_balancing_index,
            'energy_efficiency': self.energy_efficiency,
            'accessibility_index': self.accessibility_index,
            'service_frequency': self.service_frequency,
            'route_overlap_percentage': self.route_overlap_percentage
        }


@dataclass
class ExecutionSummary:
    """Resumo executivo de uma ou múltiplas execuções."""
    total_executions: int
    successful_executions: int
    success_rate: float
    avg_execution_time: float
    avg_iterations_to_convergence: float
    best_overall_fitness: float
    algorithm_type: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetricsReport:
    """Relatório completo de métricas."""
    execution_summary: ExecutionSummary
    quality_metrics: RouteQualityMetrics
    convergence_analysis: ConvergenceMetrics
    comparative_analysis: Optional[ComparativeMetrics]
    domain_metrics: DomainMetrics
    visualizations: List[str] = field(default_factory=list)  # Paths para arquivos gerados
    timestamp: datetime = field(default_factory=datetime.now)
    config_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte relatório completo para dicionário."""
        return {
            'execution_summary': {
                'total_executions': self.execution_summary.total_executions,
                'successful_executions': self.execution_summary.successful_executions,
                'success_rate': self.execution_summary.success_rate,
                'avg_execution_time': self.execution_summary.avg_execution_time,
                'avg_iterations_to_convergence': self.execution_summary.avg_iterations_to_convergence,
                'best_overall_fitness': self.execution_summary.best_overall_fitness,
                'algorithm_type': self.execution_summary.algorithm_type
            },
            'quality_metrics': self.quality_metrics.to_dict(),
            'convergence_analysis': self.convergence_analysis.to_dict(),
            'comparative_analysis': self.comparative_analysis.to_dict() if self.comparative_analysis else None,
            'domain_metrics': self.domain_metrics.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'visualizations': self.visualizations,
            'config_used': self.config_used
        }