"""
Sistema de Métricas Expandido para Rota_ACO

Este módulo fornece análise detalhada da performance dos algoritmos ACO
aplicados ao VRP (Vehicle Routing Problem), incluindo métricas de qualidade,
análise de convergência e relatórios acadêmicos.
"""

from .data_models import (
    ExecutionData,
    IterationData,
    Route,
    Solution,
    RouteQualityMetrics,
    ConvergenceMetrics,
    ComparativeMetrics,
    DomainMetrics,
    ExecutionSummary,
    MetricsReport
)

from .config import MetricsConfig
from .exceptions import (
    MetricsSystemError,
    DataCollectionError,
    MetricsCalculationError,
    VisualizationError,
    ReportGenerationError
)
from .report_generator import ReportGenerator, ReportSection

__version__ = "1.0.0"
__all__ = [
    "ExecutionData",
    "IterationData", 
    "Route",
    "Solution",
    "RouteQualityMetrics",
    "ConvergenceMetrics",
    "ComparativeMetrics",
    "DomainMetrics",
    "ExecutionSummary",
    "MetricsReport",
    "MetricsConfig",
    "MetricsSystemError",
    "DataCollectionError",
    "MetricsCalculationError",
    "VisualizationError",
    "ReportGenerationError",
    "ReportGenerator",
    "ReportSection"
]