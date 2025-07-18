"""
Testes unitários para o motor de visualização do sistema de métricas.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.rota_aco.metrics.visualization_engine import VisualizationEngine
from src.rota_aco.metrics.data_models import (
    ConvergenceMetrics, ComparativeMetrics, RouteQualityMetrics,
    ExecutionData, IterationData, Solution, Route
)
from src.rota_aco.metrics.exceptions import VisualizationError


class TestVisualizationEngine:
    """Testes para a classe VisualizationEngine."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def viz_engine(self, temp_dir):
        """Cria instância do motor de visualização para testes."""
        return VisualizationEngine(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_convergence_data(self):
        """Cria dados de convergência de exemplo."""
        return ConvergenceMetrics(
            convergence_point=50,
            final_stability=0.001,
            improvement_rate=0.05,
            plateau_detection=True,
            convergence_speed=0.02,
            total_iterations=100,
            best_fitness_evolution=[100.0 - i * 0.5 for i in range(100)],
            avg_fitness_evolution=[110.0 - i * 0.3 for i in range(100)],
            variance_evolution=[10.0 - i * 0.1 for i in range(100)]
        )
    
    @pytest.fixture
    def sample_comparative_data(self):
        """Cria dados comparativos de exemplo."""
        return ComparativeMetrics(
            total_executions=10,
            successful_executions=9,
            success_rate=90.0,
            avg_execution_time=45.5,
            std_execution_time=5.2,
            avg_best_fitness=85.3,
            std_best_fitness=3.1,
            median_best_fitness=86.0,
            avg_convergence_iterations=65.2,
            relative_efficiency=1.2
        )
    
    @pytest.fixture
    def sample_quality_data(self):
        """Cria dados de qualidade de exemplo."""
        return RouteQualityMetrics(
            valid_routes_percentage=95.0,
            demand_coverage_percentage=88.5,
            vehicle_utilization_efficiency=0.75,
            capacity_violations=2,
            opposite_stops_violations=1,
            average_route_length=12.5,
            route_length_variance=2.3,
            load_balancing_index=0.85
        )
    
    def test_initialization(self, temp_dir):
        """Testa inicialização do motor de visualização."""
        viz_engine = VisualizationEngine(output_dir=temp_dir, style="academic")
        
        assert viz_engine.output_dir == Path(temp_dir)
        assert viz_engine.style == "academic"
        assert viz_engine.output_dir.exists()
    
    def test_initialization_creates_directory(self):
        """Testa se a inicialização cria o diretório de saída."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "new_viz_dir"
            viz_engine = VisualizationEngine(output_dir=str(output_path))
            
            assert output_path.exists()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_convergence_plot(self, mock_close, mock_savefig, 
                                     viz_engine, sample_convergence_data):
        """Testa geração de gráfico de convergência."""
        result_path = viz_engine.generate_convergence_plot(
            sample_convergence_data, "ACS-TIME"
        )
        
        assert result_path is not None
        assert "convergence_acs-time" in result_path.lower()
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_comparison_bar_chart(self, mock_close, mock_savefig,
                                         viz_engine, sample_comparative_data):
        """Testa geração de gráfico de barras comparativo."""
        algorithm_names = ["ACS-TIME", "ACS-VEHICLE"]
        
        result_path = viz_engine.generate_comparison_bar_chart(
            sample_comparative_data, algorithm_names
        )
        
        assert result_path is not None
        assert "comparison_bar_chart" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_quality_metrics_chart(self, mock_close, mock_savefig,
                                          viz_engine, sample_quality_data):
        """Testa geração de gráfico de métricas de qualidade."""
        result_path = viz_engine.generate_quality_metrics_chart(sample_quality_data)
        
        assert result_path is not None
        assert "quality_metrics" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_convergence_plot_with_invalid_data(self, viz_engine):
        """Testa geração de gráfico com dados inválidos."""
        invalid_data = ConvergenceMetrics(
            convergence_point=50,
            final_stability=0.001,
            improvement_rate=0.05,
            plateau_detection=True,
            convergence_speed=0.02,
            total_iterations=100,
            best_fitness_evolution=[],  # Lista vazia
            avg_fitness_evolution=[],
            variance_evolution=[]
        )
        
        with pytest.raises(VisualizationError):
            viz_engine.generate_convergence_plot(invalid_data)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_multiple_convergence_comparison(self, mock_close, mock_savefig, viz_engine):
        """Testa comparação de convergência múltipla."""
        # Cria dados de execução de exemplo
        execution1 = ExecutionData(
            algorithm_type="ACS-TIME",
            iterations_data=[
                IterationData(i, 100.0 - i, 110.0 - i, 5.0, 
                            Solution([], 0, 0.0, 0, 100.0 - i, 0.0, True))
                for i in range(50)
            ]
        )
        
        execution2 = ExecutionData(
            algorithm_type="ACS-VEHICLE",
            iterations_data=[
                IterationData(i, 95.0 - i * 0.8, 105.0 - i * 0.8, 4.0,
                            Solution([], 0, 0.0, 0, 95.0 - i * 0.8, 0.0, True))
                for i in range(60)
            ]
        )
        
        result_path = viz_engine.generate_multiple_convergence_comparison([execution1, execution2])
        
        assert result_path is not None
        assert "multiple_convergence" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_get_generated_files_empty_directory(self, viz_engine):
        """Testa obtenção de arquivos quando diretório está vazio."""
        files = viz_engine.get_generated_files()
        assert files == []
    
    def test_clear_output_directory(self, viz_engine):
        """Testa limpeza do diretório de saída."""
        # Cria alguns arquivos de teste
        test_files = [
            viz_engine.output_dir / "test1.png",
            viz_engine.output_dir / "test2.svg",
            viz_engine.output_dir / "test3.png"
        ]
        
        for file in test_files:
            file.touch()
        
        # Verifica se arquivos foram criados
        assert len(list(viz_engine.output_dir.glob("*.png"))) == 2
        assert len(list(viz_engine.output_dir.glob("*.svg"))) == 1
        
        # Limpa diretório
        viz_engine.clear_output_directory()
        
        # Verifica se arquivos foram removidos
        assert len(list(viz_engine.output_dir.glob("*.png"))) == 0
        assert len(list(viz_engine.output_dir.glob("*.svg"))) == 0
    
    def test_matplotlib_style_setup(self, viz_engine):
        """Testa se o estilo matplotlib foi configurado corretamente."""
        import matplotlib.pyplot as plt
        
        # Verifica algumas configurações importantes
        assert list(plt.rcParams['figure.figsize']) == [10.0, 6.0]
        assert plt.rcParams['font.size'] == 12
        assert plt.rcParams['axes.grid'] == True
    
    @patch('matplotlib.pyplot.savefig', side_effect=Exception("Erro de salvamento"))
    def test_error_handling_in_plot_generation(self, mock_savefig, 
                                             viz_engine, sample_convergence_data):
        """Testa tratamento de erros durante geração de gráficos."""
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_convergence_plot(sample_convergence_data)
        
        assert "Erro ao gerar gráfico de convergência" in str(exc_info.value)
    
    def test_convergence_plot_with_convergence_point_beyond_data(self, viz_engine):
        """Testa gráfico de convergência com ponto de convergência além dos dados."""
        convergence_data = ConvergenceMetrics(
            convergence_point=150,  # Além do tamanho dos dados
            final_stability=0.001,
            improvement_rate=0.05,
            plateau_detection=True,
            convergence_speed=0.02,
            total_iterations=100,
            best_fitness_evolution=[100.0 - i * 0.5 for i in range(100)],
            avg_fitness_evolution=[110.0 - i * 0.3 for i in range(100)],
            variance_evolution=[10.0 - i * 0.1 for i in range(100)]
        )
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            result_path = viz_engine.generate_convergence_plot(convergence_data)
            assert result_path is not None


class TestVisualizationEngineIntegration:
    """Testes de integração para o motor de visualização."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_visualization_pipeline(self, temp_dir):
        """Testa pipeline completo de visualização."""
        viz_engine = VisualizationEngine(output_dir=temp_dir)
        
        # Dados de exemplo
        convergence_data = ConvergenceMetrics(
            convergence_point=30,
            final_stability=0.002,
            improvement_rate=0.03,
            plateau_detection=False,
            convergence_speed=0.025,
            total_iterations=50,
            best_fitness_evolution=[90.0 - i * 0.8 for i in range(50)],
            avg_fitness_evolution=[95.0 - i * 0.6 for i in range(50)],
            variance_evolution=[8.0 - i * 0.15 for i in range(50)]
        )
        
        comparative_data = ComparativeMetrics(
            total_executions=5,
            successful_executions=5,
            success_rate=100.0,
            avg_execution_time=30.2,
            std_execution_time=2.1,
            avg_best_fitness=75.8,
            std_best_fitness=1.9,
            median_best_fitness=76.0,
            avg_convergence_iterations=45.0,
            relative_efficiency=1.1
        )
        
        quality_data = RouteQualityMetrics(
            valid_routes_percentage=98.0,
            demand_coverage_percentage=92.0,
            vehicle_utilization_efficiency=0.82,
            capacity_violations=0,
            opposite_stops_violations=0,
            average_route_length=15.2,
            route_length_variance=1.8,
            load_balancing_index=0.91
        )
        
        # Gera todas as visualizações
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            conv_path = viz_engine.generate_convergence_plot(convergence_data, "ACS-TIME")
            comp_path = viz_engine.generate_comparison_bar_chart(comparative_data, ["ACS-TIME"])
            qual_path = viz_engine.generate_quality_metrics_chart(quality_data)
        
        # Verifica se todos os arquivos foram "gerados"
        assert conv_path is not None
        assert comp_path is not None
        assert qual_path is not None
        
        # Verifica se os nomes dos arquivos estão corretos
        assert "convergence" in conv_path
        assert "comparison" in comp_path
        assert "quality" in qual_path


class TestAdvancedVisualizations:
    """Testes para visualizações avançadas."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def viz_engine(self, temp_dir):
        """Cria instância do motor de visualização para testes."""
        return VisualizationEngine(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_routes(self):
        """Cria rotas de exemplo para testes."""
        from src.rota_aco.metrics.data_models import Route
        return [
            Route(stops=[1, 2, 3, 4], distances=[1.0, 1.5, 2.0], 
                  passenger_load=[10, 15, 20], total_distance=4.5, 
                  total_passengers=45, is_valid=True),
            Route(stops=[1, 5, 6, 7], distances=[1.2, 1.8, 1.0], 
                  passenger_load=[12, 18, 8], total_distance=4.0, 
                  total_passengers=38, is_valid=True),
            Route(stops=[2, 8, 9, 3], distances=[2.0, 1.0, 1.5], 
                  passenger_load=[20, 10, 15], total_distance=4.5, 
                  total_passengers=45, is_valid=True)
        ]
    
    @pytest.fixture
    def sample_stop_coordinates(self):
        """Cria coordenadas de paradas de exemplo."""
        return {
            1: (0.0, 0.0), 2: (1.0, 1.0), 3: (2.0, 2.0), 4: (3.0, 1.0),
            5: (1.0, 3.0), 6: (2.0, 4.0), 7: (3.0, 3.0), 8: (0.5, 2.5), 9: (1.5, 1.5)
        }
    
    @pytest.fixture
    def sample_executions_data(self):
        """Cria dados de execuções de exemplo."""
        from src.rota_aco.metrics.data_models import ExecutionData, IterationData, Solution
        
        executions = []
        for exec_id in range(3):
            iterations = []
            for i in range(50):
                solution = Solution([], 0, 0.0, 0, 100.0 - i - exec_id, 0.0, True)
                iter_data = IterationData(i, 100.0 - i - exec_id, 105.0 - i - exec_id, 5.0, solution)
                iterations.append(iter_data)
            
            final_solution = Solution([], 0, 0.0, 0, 50.0 - exec_id, 0.0, True)
            execution = ExecutionData(
                algorithm_type=f"ACS-TEST-{exec_id}",
                iterations_data=iterations,
                final_solution=final_solution
            )
            executions.append(execution)
        
        return executions
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_stop_utilization_heatmap(self, mock_close, mock_savefig,
                                             viz_engine, sample_routes, sample_stop_coordinates):
        """Testa geração de heatmap de utilização de paradas."""
        result_path = viz_engine.generate_stop_utilization_heatmap(
            sample_routes, sample_stop_coordinates
        )
        
        assert result_path is not None
        assert "stop_utilization_heatmap" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_heatmap_with_empty_routes(self, viz_engine, sample_stop_coordinates):
        """Testa heatmap com lista de rotas vazia."""
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_stop_utilization_heatmap([], sample_stop_coordinates)
        
        assert "Dados insuficientes para gerar heatmap" in str(exc_info.value)
    
    def test_heatmap_with_empty_coordinates(self, viz_engine, sample_routes):
        """Testa heatmap com coordenadas vazias."""
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_stop_utilization_heatmap(sample_routes, {})
        
        assert "Dados insuficientes para gerar heatmap" in str(exc_info.value)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_fitness_distribution_histogram(self, mock_close, mock_savefig,
                                                   viz_engine, sample_executions_data):
        """Testa geração de histograma de distribuição de fitness."""
        result_path = viz_engine.generate_fitness_distribution_histogram(sample_executions_data)
        
        assert result_path is not None
        assert "fitness_distribution" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_histogram_with_empty_executions(self, viz_engine):
        """Testa histograma com lista de execuções vazia."""
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_fitness_distribution_histogram([])
        
        assert "Nenhum dado de execução fornecido" in str(exc_info.value)
    
    def test_histogram_with_no_fitness_data(self, viz_engine):
        """Testa histograma com execuções sem dados de fitness."""
        from src.rota_aco.metrics.data_models import ExecutionData
        
        empty_execution = ExecutionData(algorithm_type="EMPTY")
        
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_fitness_distribution_histogram([empty_execution])
        
        assert "Nenhum valor de fitness encontrado" in str(exc_info.value)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_convergence_confidence_intervals(self, mock_close, mock_savefig,
                                                     viz_engine, sample_executions_data):
        """Testa geração de gráfico com intervalos de confiança."""
        result_path = viz_engine.generate_convergence_confidence_intervals(
            sample_executions_data, confidence_level=0.95
        )
        
        assert result_path is not None
        assert "convergence_confidence" in result_path
        assert result_path.endswith(".png")
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_confidence_intervals_with_empty_executions(self, viz_engine):
        """Testa intervalos de confiança com lista de execuções vazia."""
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_convergence_confidence_intervals([])
        
        assert "Nenhum dado de execução fornecido" in str(exc_info.value)
    
    def test_confidence_intervals_with_no_iteration_data(self, viz_engine):
        """Testa intervalos de confiança com execuções sem dados de iteração."""
        from src.rota_aco.metrics.data_models import ExecutionData
        
        empty_execution = ExecutionData(algorithm_type="EMPTY")
        
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_convergence_confidence_intervals([empty_execution])
        
        assert "Nenhum dado de iteração encontrado" in str(exc_info.value)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_confidence_intervals_with_different_confidence_levels(self, mock_close, mock_savefig,
                                                                 viz_engine, sample_executions_data):
        """Testa intervalos de confiança com diferentes níveis de confiança."""
        # Testa com 90% de confiança
        result_path = viz_engine.generate_convergence_confidence_intervals(
            sample_executions_data, confidence_level=0.90
        )
        
        assert result_path is not None
        assert "convergence_confidence" in result_path
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_heatmap_with_routes_without_stops_attribute(self, viz_engine, sample_stop_coordinates):
        """Testa heatmap com rotas que não têm atributo 'stops'."""
        # Cria objetos mock sem atributo 'stops'
        mock_routes = [type('MockRoute', (), {})() for _ in range(3)]
        
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_stop_utilization_heatmap(mock_routes, sample_stop_coordinates)
        
        assert "Dados insuficientes para gerar heatmap" in str(exc_info.value)
    
    def test_heatmap_with_mismatched_coordinates(self, viz_engine, sample_routes):
        """Testa heatmap com coordenadas que não correspondem às paradas das rotas."""
        # Coordenadas que não correspondem às paradas das rotas
        mismatched_coords = {100: (0.0, 0.0), 101: (1.0, 1.0)}
        
        with pytest.raises(VisualizationError) as exc_info:
            viz_engine.generate_stop_utilization_heatmap(sample_routes, mismatched_coords)
        
        assert "Nenhuma coordenada válida encontrada" in str(exc_info.value)


class TestPresentationFormatting:
    """Testes para funcionalidades de formatação para apresentação."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def academic_viz_engine(self, temp_dir):
        """Cria instância com estilo acadêmico."""
        return VisualizationEngine(output_dir=temp_dir, style="academic")
    
    @pytest.fixture
    def presentation_viz_engine(self, temp_dir):
        """Cria instância com estilo de apresentação."""
        return VisualizationEngine(output_dir=temp_dir, style="presentation")
    
    @pytest.fixture
    def sample_convergence_data(self):
        """Cria dados de convergência de exemplo."""
        return ConvergenceMetrics(
            convergence_point=30,
            final_stability=0.002,
            improvement_rate=0.03,
            plateau_detection=False,
            convergence_speed=0.025,
            total_iterations=50,
            best_fitness_evolution=[90.0 - i * 0.8 for i in range(50)],
            avg_fitness_evolution=[95.0 - i * 0.6 for i in range(50)],
            variance_evolution=[8.0 - i * 0.15 for i in range(50)]
        )
    
    @pytest.fixture
    def sample_comparative_data(self):
        """Cria dados comparativos de exemplo."""
        return ComparativeMetrics(
            total_executions=5,
            successful_executions=5,
            success_rate=100.0,
            avg_execution_time=30.2,
            std_execution_time=2.1,
            avg_best_fitness=75.8,
            std_best_fitness=1.9,
            median_best_fitness=76.0,
            avg_convergence_iterations=45.0,
            relative_efficiency=1.1
        )
    
    def test_academic_style_initialization(self, academic_viz_engine):
        """Testa inicialização com estilo acadêmico."""
        assert academic_viz_engine.style == "academic"
        assert academic_viz_engine.export_formats == ['png']
        
        # Verifica se configurações acadêmicas foram aplicadas
        import matplotlib.pyplot as plt
        assert plt.rcParams['font.family'] == ['serif']
        assert list(plt.rcParams['figure.figsize']) == [10.0, 6.0]
    
    def test_presentation_style_initialization(self, presentation_viz_engine):
        """Testa inicialização com estilo de apresentação."""
        assert presentation_viz_engine.style == "presentation"
        
        # Verifica se configurações de apresentação foram aplicadas
        import matplotlib.pyplot as plt
        assert plt.rcParams['font.family'] == ['sans-serif']
        assert list(plt.rcParams['figure.figsize']) == [12.0, 8.0]
        assert plt.rcParams['font.size'] == 14
    
    def test_set_export_formats(self, academic_viz_engine):
        """Testa configuração de formatos de exportação."""
        # Testa formatos válidos
        academic_viz_engine.set_export_formats(['png', 'svg', 'pdf'])
        assert academic_viz_engine.export_formats == ['png', 'svg', 'pdf']
        
        # Testa formatos inválidos (devem ser filtrados)
        academic_viz_engine.set_export_formats(['png', 'invalid', 'svg'])
        assert academic_viz_engine.export_formats == ['png', 'svg']
        
        # Testa lista vazia (deve usar fallback)
        academic_viz_engine.set_export_formats([])
        assert academic_viz_engine.export_formats == ['png']
        
        # Testa formatos completamente inválidos
        academic_viz_engine.set_export_formats(['invalid1', 'invalid2'])
        assert academic_viz_engine.export_formats == ['png']
    
    @patch('matplotlib.pyplot.close')
    def test_generate_formatted_convergence_plot(self, mock_close,
                                               academic_viz_engine, sample_convergence_data):
        """Testa geração de gráfico de convergência formatado."""
        # Configura múltiplos formatos
        academic_viz_engine.set_export_formats(['png', 'svg'])
        
        # Mock do método _save_with_multiple_formats
        with patch.object(academic_viz_engine, '_save_with_multiple_formats') as mock_save:
            mock_save.return_value = ['test1.png', 'test2.svg']
            
            result_paths = academic_viz_engine.generate_formatted_convergence_plot(
                sample_convergence_data, "ACS-TIME", 
                subtitle="Teste de Subtítulo", 
                footer="Teste de Rodapé"
            )
            
            assert isinstance(result_paths, list)
            assert len(result_paths) == 2  # PNG e SVG
            assert result_paths == ['test1.png', 'test2.svg']
            
            # Verifica se o método de salvamento foi chamado
            mock_save.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.close')
    def test_generate_formatted_comparison_chart(self, mock_close,
                                               presentation_viz_engine, sample_comparative_data):
        """Testa geração de gráfico comparativo formatado."""
        algorithm_names = ["ACS-TIME", "ACS-VEHICLE"]
        
        # Mock do método _save_with_multiple_formats
        with patch.object(presentation_viz_engine, '_save_with_multiple_formats') as mock_save:
            mock_save.return_value = ['test_comparison.png']
            
            result_paths = presentation_viz_engine.generate_formatted_comparison_chart(
                sample_comparative_data, algorithm_names,
                subtitle="Comparação de Algoritmos",
                footer="Dados de Teste"
            )
            
            assert isinstance(result_paths, list)
            assert len(result_paths) == 1  # Apenas PNG por padrão
            assert result_paths == ['test_comparison.png']
            
            mock_save.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.close')
    def test_create_presentation_summary(self, mock_close, academic_viz_engine):
        """Testa criação de resumo de apresentação."""
        report_data = {"test": "data"}  # Dados fictícios
        
        # Mock do método _save_with_multiple_formats
        with patch.object(academic_viz_engine, '_save_with_multiple_formats') as mock_save:
            mock_save.return_value = ['test_summary.png']
            
            result_paths = academic_viz_engine.create_presentation_summary(
                report_data, title="Teste de Resumo"
            )
            
            assert isinstance(result_paths, list)
            assert len(result_paths) == 1
            assert result_paths == ['test_summary.png']
            
            mock_save.assert_called_once()
            mock_close.assert_called_once()
    
    def test_formatted_convergence_with_invalid_data(self, academic_viz_engine):
        """Testa gráfico formatado com dados inválidos."""
        invalid_data = ConvergenceMetrics(
            convergence_point=50,
            final_stability=0.001,
            improvement_rate=0.05,
            plateau_detection=True,
            convergence_speed=0.02,
            total_iterations=100,
            best_fitness_evolution=[],  # Lista vazia
            avg_fitness_evolution=[],
            variance_evolution=[]
        )
        
        with pytest.raises(VisualizationError) as exc_info:
            academic_viz_engine.generate_formatted_convergence_plot(invalid_data)
        
        assert "Dados de convergência estão vazios ou inválidos" in str(exc_info.value)
    
    @patch('matplotlib.pyplot.close')
    def test_save_with_multiple_formats_error_handling(self, mock_close,
                                                     academic_viz_engine, sample_convergence_data):
        """Testa tratamento de erros ao salvar em múltiplos formatos."""
        academic_viz_engine.set_export_formats(['png', 'svg'])
        
        # Mock do método _save_with_multiple_formats para simular erro
        with patch.object(academic_viz_engine, '_save_with_multiple_formats') as mock_save:
            mock_save.return_value = []  # Simula falha em todos os formatos
            
            result_paths = academic_viz_engine.generate_formatted_convergence_plot(
                sample_convergence_data
            )
            
            # Deve retornar lista vazia se todos os formatos falharem
            assert result_paths == []
            mock_save.assert_called_once()
            mock_close.assert_called_once()
    
    def test_get_generated_files_multiple_formats(self, academic_viz_engine):
        """Testa obtenção de arquivos gerados em múltiplos formatos."""
        # Cria arquivos de teste em diferentes formatos
        test_files = [
            academic_viz_engine.output_dir / "test1.png",
            academic_viz_engine.output_dir / "test2.svg",
            academic_viz_engine.output_dir / "test3.pdf",
            academic_viz_engine.output_dir / "test4.eps"
        ]
        
        for file in test_files:
            file.touch()
        
        generated_files = academic_viz_engine.get_generated_files()
        
        # Deve incluir todos os formatos suportados
        assert len(generated_files) == 4
        assert any("test1.png" in path for path in generated_files)
        assert any("test2.svg" in path for path in generated_files)
        assert any("test3.pdf" in path for path in generated_files)
        assert any("test4.eps" in path for path in generated_files)
    
    def test_clear_output_directory_multiple_formats(self, academic_viz_engine):
        """Testa limpeza de diretório com múltiplos formatos."""
        # Cria arquivos de teste em diferentes formatos
        test_files = [
            academic_viz_engine.output_dir / "test1.png",
            academic_viz_engine.output_dir / "test2.svg",
            academic_viz_engine.output_dir / "test3.pdf",
            academic_viz_engine.output_dir / "test4.eps"
        ]
        
        for file in test_files:
            file.touch()
        
        # Verifica se arquivos foram criados
        assert len(academic_viz_engine.get_generated_files()) == 4
        
        # Limpa diretório
        academic_viz_engine.clear_output_directory()
        
        # Verifica se todos os arquivos foram removidos
        assert len(academic_viz_engine.get_generated_files()) == 0
    
    @patch('matplotlib.pyplot.close')
    def test_presentation_formatting_with_long_algorithm_names(self, mock_close,
                                                             academic_viz_engine, sample_comparative_data):
        """Testa formatação com nomes de algoritmos longos."""
        long_algorithm_names = [
            "Very-Long-Algorithm-Name-ACS-TIME",
            "Another-Very-Long-Algorithm-Name-ACS-VEHICLE"
        ]
        
        # Mock do método _save_with_multiple_formats
        with patch.object(academic_viz_engine, '_save_with_multiple_formats') as mock_save:
            mock_save.return_value = ['test_long_names.png']
            
            result_paths = academic_viz_engine.generate_formatted_comparison_chart(
                sample_comparative_data, long_algorithm_names
            )
            
            assert isinstance(result_paths, list)
            assert len(result_paths) == 1
            assert result_paths == ['test_long_names.png']
            mock_save.assert_called_once()
            mock_close.assert_called_once()