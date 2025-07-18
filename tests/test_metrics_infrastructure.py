"""
Testes para a infraestrutura básica do sistema de métricas.

Este módulo testa as estruturas de dados fundamentais, configurações
e exceções do sistema de métricas expandido.
"""

import pytest
import os
import tempfile
import json
from datetime import datetime
from unittest.mock import patch

from src.rota_aco.metrics import (
    ExecutionData,
    IterationData,
    Route,
    Solution,
    RouteQualityMetrics,
    ConvergenceMetrics,
    ComparativeMetrics,
    DomainMetrics,
    ExecutionSummary,
    MetricsReport,
    MetricsConfig,
    MetricsSystemError,
    DataCollectionError,
    MetricsCalculationError,
    VisualizationError
)


class TestDataModels:
    """Testes para os modelos de dados."""
    
    def test_route_creation(self):
        """Testa criação de rota válida."""
        route = Route(
            stops=[1, 2, 3],
            distances=[100.0, 150.0],
            passenger_load=[20, 15],
            total_distance=250.0,
            total_passengers=35,
            is_valid=True
        )
        
        assert len(route.stops) == 3
        assert len(route.distances) == 2
        assert len(route.passenger_load) == 2
        assert route.total_distance == 250.0
        assert route.total_passengers == 35
        assert route.is_valid is True
    
    def test_route_validation_error(self):
        """Testa validação de dados inválidos na rota."""
        with pytest.raises(ValueError, match="Número de paradas deve ser distâncias \\+ 1"):
            Route(
                stops=[1, 2],  # 2 paradas
                distances=[100.0, 150.0, 200.0],  # 3 distâncias (deveria ser 1)
                passenger_load=[20, 15, 10],
                total_distance=450.0,
                total_passengers=45,
                is_valid=True
            )
    
    def test_solution_creation(self):
        """Testa criação de solução."""
        route1 = Route([1, 2], [100.0], [20], 100.0, 20, True)
        route2 = Route([3, 4], [150.0], [25], 150.0, 25, True)
        
        solution = Solution(
            routes=[route1, route2],
            total_vehicles=2,
            total_distance=250.0,
            total_passengers_served=45,
            fitness_time=250.0,
            fitness_vehicle=2.0,
            is_feasible=True
        )
        
        assert len(solution.routes) == 2
        assert solution.total_vehicles == 2
        assert solution.total_distance == 250.0
        assert solution.total_passengers_served == 45
    
    def test_solution_auto_calculation(self):
        """Testa cálculo automático de métricas da solução."""
        route1 = Route([1, 2], [100.0], [20], 100.0, 20, True)
        route2 = Route([3, 4], [150.0], [25], 150.0, 25, True)
        
        solution = Solution(
            routes=[route1, route2],
            total_vehicles=0,  # Será recalculado
            total_distance=0.0,  # Será recalculado
            total_passengers_served=0,  # Será recalculado
            fitness_time=250.0,
            fitness_vehicle=2.0,
            is_feasible=True
        )
        
        # Valores devem ser recalculados automaticamente
        assert solution.total_vehicles == 2
        assert solution.total_distance == 250.0
        assert solution.total_passengers_served == 45
    
    def test_execution_data_creation(self):
        """Testa criação de dados de execução."""
        execution_data = ExecutionData(
            algorithm_type="ACS-TIME",
            config={"ants": 10, "iterations": 100},
            execution_time=45.5,
            success=True
        )
        
        assert execution_data.algorithm_type == "ACS-TIME"
        assert execution_data.config["ants"] == 10
        assert execution_data.execution_time == 45.5
        assert execution_data.success is True
        assert execution_data.execution_id is not None  # UUID gerado automaticamente
    
    def test_metrics_to_dict_conversion(self):
        """Testa conversão de métricas para dicionário."""
        quality_metrics = RouteQualityMetrics(
            valid_routes_percentage=95.0,
            demand_coverage_percentage=98.5,
            vehicle_utilization_efficiency=85.0,
            capacity_violations=2,
            opposite_stops_violations=0,
            average_route_length=5.5,
            route_length_variance=1.2,
            load_balancing_index=0.8
        )
        
        metrics_dict = quality_metrics.to_dict()
        
        assert metrics_dict['valid_routes_percentage'] == 95.0
        assert metrics_dict['demand_coverage_percentage'] == 98.5
        assert metrics_dict['capacity_violations'] == 2
        assert len(metrics_dict) == 8


class TestMetricsConfig:
    """Testes para configuração do sistema de métricas."""
    
    def test_default_config_creation(self):
        """Testa criação de configuração padrão."""
        config = MetricsConfig()
        
        assert config.enable_convergence_analysis is True
        assert config.convergence_threshold == 0.001
        assert config.vehicle_capacity == 70
        assert config.output_formats == ['png', 'svg']
        assert config.report_format == 'markdown'
    
    def test_config_validation(self):
        """Testa validação de configuração inválida."""
        with pytest.raises(ValueError, match="convergence_threshold deve ser positivo"):
            MetricsConfig(convergence_threshold=-0.1)
        
        with pytest.raises(ValueError, match="vehicle_capacity deve ser positivo"):
            MetricsConfig(vehicle_capacity=0)
        
        with pytest.raises(ValueError, match="confidence_level deve estar entre 0 e 1"):
            MetricsConfig(confidence_level=1.5)
    
    def test_config_directory_creation(self):
        """Testa criação automática de diretórios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MetricsConfig(base_output_dir=temp_dir)
            
            # Diretórios devem ser criados automaticamente
            assert os.path.exists(os.path.join(temp_dir, 'execution_data'))
            assert os.path.exists(os.path.join(temp_dir, 'reports'))
            assert os.path.exists(os.path.join(temp_dir, 'visualizations'))
            assert os.path.exists(os.path.join(temp_dir, 'comparisons'))
    
    def test_config_to_dict(self):
        """Testa conversão de configuração para dicionário."""
        config = MetricsConfig()
        config_dict = config.to_dict()
        
        assert 'convergence_analysis' in config_dict
        assert 'quality_metrics' in config_dict
        assert 'visualizations' in config_dict
        assert config_dict['convergence_analysis']['convergence_threshold'] == 0.001
    
    def test_config_save_and_load(self):
        """Testa salvamento e carregamento de configuração."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Criar e salvar configuração
            original_config = MetricsConfig(
                convergence_threshold=0.005,
                vehicle_capacity=80,
                figure_dpi=600
            )
            original_config.save_to_file(config_path)
            
            # Carregar configuração
            loaded_config = MetricsConfig.load_from_file(config_path)
            
            assert loaded_config.convergence_threshold == 0.005
            assert loaded_config.vehicle_capacity == 80
            assert loaded_config.figure_dpi == 600
        
        finally:
            os.unlink(config_path)
    
    def test_config_load_nonexistent_file(self):
        """Testa carregamento de arquivo inexistente."""
        with patch('builtins.print') as mock_print:
            config = MetricsConfig.load_from_file('nonexistent_file.json')
            
            # Deve retornar configuração padrão
            assert config.convergence_threshold == 0.001
            mock_print.assert_called()


class TestExceptions:
    """Testes para exceções personalizadas."""
    
    def test_base_exception(self):
        """Testa exceção base."""
        error = MetricsSystemError("Erro base", "Detalhes do erro")
        
        assert str(error) == "Erro base\nDetalhes: Detalhes do erro"
        assert error.message == "Erro base"
        assert error.details == "Detalhes do erro"
    
    def test_data_collection_error(self):
        """Testa exceção de coleta de dados."""
        error = DataCollectionError(
            "Erro na coleta",
            execution_id="test-123",
            details="Falha na conexão"
        )
        
        assert "Erro na coleta" in str(error)
        assert "test-123" in str(error)
        assert "Falha na conexão" in str(error)
        assert error.execution_id == "test-123"
    
    def test_metrics_calculation_error(self):
        """Testa exceção de cálculo de métricas."""
        error = MetricsCalculationError(
            "Erro no cálculo",
            metric_type="convergence",
            details="Divisão por zero"
        )
        
        assert "Erro no cálculo" in str(error)
        assert "convergence" in str(error)
        assert error.metric_type == "convergence"
    
    def test_visualization_error(self):
        """Testa exceção de visualização."""
        error = VisualizationError(
            "Erro no gráfico",
            chart_type="line_plot",
            details="Dados insuficientes"
        )
        
        assert "Erro no gráfico" in str(error)
        assert "line_plot" in str(error)
        assert error.chart_type == "line_plot"


class TestSpecialConfigs:
    """Testes para configurações especiais."""
    
    def test_academic_config(self):
        """Testa configuração acadêmica."""
        from src.rota_aco.metrics.config import create_academic_config
        
        config = create_academic_config()
        
        assert config.figure_dpi == 600
        assert 'pdf' in config.output_formats
        assert config.enable_statistical_tests is True
        assert config.include_raw_data is True
    
    def test_fast_config(self):
        """Testa configuração rápida."""
        from src.rota_aco.metrics.config import create_fast_config
        
        config = create_fast_config()
        
        assert config.enable_visualizations is False
        assert config.enable_reports is False
        assert config.max_iterations_to_store == 1000
        assert config.enable_parallel_processing is True


if __name__ == "__main__":
    pytest.main([__file__])