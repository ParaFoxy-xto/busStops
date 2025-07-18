"""
Testes de integração para funcionalidades de métricas no CLI.

Este módulo testa a integração do sistema de métricas expandido
com a interface de linha de comando.
"""

import os
import sys
import tempfile
import shutil
import json
import pytest
from unittest.mock import patch, MagicMock
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rota_aco.cli.run import (
    setup_arg_parser, 
    setup_metrics_config,
    run_single_execution_with_metrics,
    generate_metrics_reports
)
from rota_aco.metrics.config import MetricsConfig


class TestCLIMetricsFlags:
    """Testa os flags de métricas no CLI."""
    
    def test_metrics_flag_parsing(self):
        """Testa se o flag --metrics é parseado corretamente."""
        parser = setup_arg_parser()
        
        # Test with metrics flag
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0', 
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--metrics'
        ])
        
        assert args.metrics is True
        assert args.report_output == "output/metrics"  # default value
    
    def test_report_output_flag_parsing(self):
        """Testa se o flag --report-output é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0', 
            '--exit-lon', '1.0',
            '--metrics',
            '--report-output', '/custom/path'
        ])
        
        assert args.metrics is True
        assert args.report_output == "/custom/path"
    
    def test_compare_runs_flag_parsing(self):
        """Testa se o flag --compare-runs é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--compare-runs', '5'
        ])
        
        assert args.compare_runs == 5
    
    def test_convergence_analysis_flag_parsing(self):
        """Testa se o flag --convergence-analysis é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--convergence-analysis'
        ])
        
        assert args.convergence_analysis is True
    
    def test_metrics_config_flag_parsing(self):
        """Testa se o flag --metrics-config é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--metrics-config', '/path/to/config.json'
        ])
        
        assert args.metrics_config == "/path/to/config.json"


class TestAdvancedCLIFlags:
    """Testa os flags avançados de métricas no CLI."""
    
    def test_statistical_tests_flag_parsing(self):
        """Testa se o flag --statistical-tests é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--statistical-tests'
        ])
        
        assert args.statistical_tests is True
    
    def test_confidence_level_flag_parsing(self):
        """Testa se o flag --confidence-level é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--confidence-level', '0.99'
        ])
        
        assert args.confidence_level == 0.99
    
    def test_export_raw_data_flag_parsing(self):
        """Testa se o flag --export-raw-data é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--export-raw-data'
        ])
        
        assert args.export_raw_data is True
    
    def test_visualization_formats_flag_parsing(self):
        """Testa se o flag --visualization-formats é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--visualization-formats', 'png', 'svg', 'pdf'
        ])
        
        assert args.visualization_formats == ['png', 'svg', 'pdf']
    
    def test_academic_mode_flag_parsing(self):
        """Testa se o flag --academic-mode é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--academic-mode'
        ])
        
        assert args.academic_mode is True
    
    def test_fast_mode_flag_parsing(self):
        """Testa se o flag --fast-mode é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--fast-mode'
        ])
        
        assert args.fast_mode is True
    
    def test_parallel_executions_flag_parsing(self):
        """Testa se o flag --parallel-executions é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--parallel-executions'
        ])
        
        assert args.parallel_executions is True
    
    def test_seed_flag_parsing(self):
        """Testa se o flag --seed é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--seed', '42'
        ])
        
        assert args.seed == 42
    
    def test_save_execution_data_flag_parsing(self):
        """Testa se o flag --save-execution-data é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--save-execution-data'
        ])
        
        assert args.save_execution_data is True
    
    def test_load_previous_data_flag_parsing(self):
        """Testa se o flag --load-previous-data é parseado corretamente."""
        parser = setup_arg_parser()
        
        args = parser.parse_args([
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--load-previous-data', '/path/to/data'
        ])
        
        assert args.load_previous_data == "/path/to/data"


class TestMetricsConfigSetup:
    """Testa a configuração do sistema de métricas."""
    
    def test_default_metrics_config(self):
        """Testa configuração padrão de métricas."""
        # Mock args without special flags
        args = argparse.Namespace(
            metrics_config=None,
            convergence_analysis=False,
            report_output=None
        )
        
        config = setup_metrics_config(args)
        
        assert isinstance(config, MetricsConfig)
        assert config.enable_convergence_analysis is True  # default
        assert config.enable_quality_metrics is True
    
    def test_convergence_analysis_config(self):
        """Testa configuração com análise de convergência."""
        args = argparse.Namespace(
            metrics_config=None,
            convergence_analysis=True,
            report_output=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.enable_convergence_analysis is True
        assert config.enable_visualizations is True
        assert config.enable_reports is True
        assert config.enable_statistical_tests is True
    
    def test_custom_report_output_config(self):
        """Testa configuração com diretório personalizado."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_output = os.path.join(temp_dir, "custom_metrics")
            
            args = argparse.Namespace(
                metrics_config=None,
                convergence_analysis=False,
                report_output=custom_output
            )
            
            config = setup_metrics_config(args)
            
            assert config.base_output_dir == custom_output
            # Check if directories were created
            assert os.path.exists(config.get_execution_data_path())
            assert os.path.exists(config.get_reports_path())
    
    def test_custom_config_file_loading(self):
        """Testa carregamento de arquivo de configuração personalizado."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'convergence_analysis': {
                    'convergence_threshold': 0.005,
                    'stability_window': 100
                },
                'visualizations': {
                    'figure_dpi': 600,
                    'output_formats': ['png', 'svg', 'pdf']
                }
            }
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            args = argparse.Namespace(
                metrics_config=config_file,
                convergence_analysis=False,
                report_output=None
            )
            
            config = setup_metrics_config(args)
            
            assert config.convergence_threshold == 0.005
            assert config.stability_window == 100
            assert config.figure_dpi == 600
            assert 'pdf' in config.output_formats
            
        finally:
            os.unlink(config_file)
    
    def test_invalid_config_file_fallback(self):
        """Testa fallback quando arquivo de configuração é inválido."""
        args = argparse.Namespace(
            metrics_config="/nonexistent/config.json",
            convergence_analysis=False,
            report_output=None
        )
        
        # Should not raise exception, should use default config
        config = setup_metrics_config(args)
        
        assert isinstance(config, MetricsConfig)
        # Should have default values
        assert config.convergence_threshold == 0.001


class TestAdvancedMetricsConfigSetup:
    """Testa configurações avançadas do sistema de métricas."""
    
    def test_academic_mode_config(self):
        """Testa configuração do modo acadêmico."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=True,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=False,
            visualization_formats=None,
            fast_mode=False,
            parallel_executions=False,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.figure_dpi == 600
        assert 'pdf' in config.output_formats
        assert config.enable_statistical_tests is True
        assert config.confidence_level == 0.95
        assert config.include_raw_data is True
        assert config.enable_detailed_logging is True
    
    def test_fast_mode_config(self):
        """Testa configuração do modo rápido."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=False,
            visualization_formats=None,
            fast_mode=True,
            parallel_executions=False,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.enable_visualizations is False
        assert config.enable_reports is False
        assert config.enable_statistical_tests is False
        assert config.max_iterations_to_store == 1000
        assert config.enable_parallel_processing is True
    
    def test_statistical_tests_config(self):
        """Testa configuração de testes estatísticos."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=True,
            confidence_level=0.99,
            export_raw_data=False,
            visualization_formats=None,
            fast_mode=False,
            parallel_executions=False,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.enable_statistical_tests is True
        assert config.confidence_level == 0.99
    
    def test_export_raw_data_config(self):
        """Testa configuração de exportação de dados brutos."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=True,
            visualization_formats=None,
            fast_mode=False,
            parallel_executions=False,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.include_raw_data is True
    
    def test_visualization_formats_config(self):
        """Testa configuração de formatos de visualização."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=False,
            visualization_formats=['svg', 'pdf'],
            fast_mode=False,
            parallel_executions=False,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.output_formats == ['svg', 'pdf']
    
    def test_parallel_executions_config(self):
        """Testa configuração de execuções paralelas."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=False,
            visualization_formats=None,
            fast_mode=False,
            parallel_executions=True,
            seed=None
        )
        
        config = setup_metrics_config(args)
        
        assert config.enable_parallel_processing is True
    
    @patch('random.seed')
    @patch('numpy.random.seed')
    def test_seed_config(self, mock_np_seed, mock_random_seed):
        """Testa configuração de semente para reprodutibilidade."""
        args = argparse.Namespace(
            metrics_config=None,
            academic_mode=False,
            convergence_analysis=False,
            report_output=None,
            statistical_tests=False,
            confidence_level=0.95,
            export_raw_data=False,
            visualization_formats=None,
            fast_mode=False,
            parallel_executions=False,
            seed=42
        )
        
        config = setup_metrics_config(args)
        
        # Verify seeds were set
        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)


class TestMetricsIntegration:
    """Testa a integração do sistema de métricas com execução ACO."""
    
    @pytest.fixture
    def mock_aco_components(self):
        """Fixture que cria componentes ACO mockados."""
        # Mock meta_graph
        meta_graph = MagicMock()
        meta_graph.nodes.return_value = [1, 2, 3, 4]
        meta_graph.edges.return_value = [(1, 2), (2, 3), (3, 4)]
        
        # Mock meta_edges
        meta_edges = {
            (1, 2): {'time': 100, 'distance': 1000},
            (2, 3): {'time': 150, 'distance': 1500},
            (3, 4): {'time': 200, 'distance': 2000}
        }
        
        # Mock other parameters
        stops_to_visit = [2, 3]
        start_node = 1
        exit_node = 4
        all_opposites = {}
        
        return {
            'meta_graph': meta_graph,
            'meta_edges': meta_edges,
            'stops_to_visit': stops_to_visit,
            'start_node': start_node,
            'exit_node': exit_node,
            'all_opposites': all_opposites
        }
    
    @pytest.fixture
    def mock_args(self):
        """Fixture que cria argumentos mockados."""
        return argparse.Namespace(
            ants=10,
            iterations=5,
            verbose=False
        )
    
    @patch('rota_aco.cli.run.run_aco_with_metrics')
    def test_single_execution_with_metrics(self, mock_run_aco, mock_aco_components, mock_args):
        """Testa execução única com métricas."""
        # Setup mock return value
        mock_result = ([[1, 2, 3, 4]], 500.0, 1, 0.95)
        mock_execution_data = MagicMock()
        mock_execution_data.success = True
        mock_execution_data.execution_time = 30.5
        mock_run_aco.return_value = (mock_result, mock_execution_data)
        
        # Setup parameters
        aco_params = {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1, 'Q': 1.0}
        problem_params = {'capacity': 70, 'max_route_length': 100}
        quality_weights = {'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5}
        metrics_config = MetricsConfig()
        
        # Execute function
        result, execution_data = run_single_execution_with_metrics(
            mock_aco_components['meta_graph'],
            mock_aco_components['meta_edges'],
            mock_aco_components['stops_to_visit'],
            mock_aco_components['start_node'],
            mock_aco_components['exit_node'],
            mock_aco_components['all_opposites'],
            aco_params,
            problem_params,
            quality_weights,
            mock_args,
            metrics_config
        )
        
        # Verify results
        assert result == mock_result
        assert execution_data == mock_execution_data
        
        # Verify run_aco_with_metrics was called with correct parameters
        mock_run_aco.assert_called_once()
        call_args = mock_run_aco.call_args
        assert call_args[1]['n_ants'] == 10
        assert call_args[1]['n_iterations'] == 5
        assert call_args[1]['verbose'] is False


class TestReportGeneration:
    """Testa a geração de relatórios de métricas."""
    
    def test_generate_reports_with_no_data(self):
        """Testa geração de relatórios sem dados."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_config = MetricsConfig()
            metrics_config.base_output_dir = temp_dir
            
            args = argparse.Namespace(verbose=False)
            
            # Should not raise exception
            generate_metrics_reports([], metrics_config, args)
            generate_metrics_reports(None, metrics_config, args)
    
    def test_generate_reports_with_failed_executions(self):
        """Testa geração de relatórios com execuções falhadas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_config = MetricsConfig()
            metrics_config.base_output_dir = temp_dir
            
            # Mock failed execution data
            failed_execution = MagicMock()
            failed_execution.success = False
            
            args = argparse.Namespace(verbose=False)
            
            # Should not raise exception
            generate_metrics_reports([failed_execution], metrics_config, args)
    
    @patch('rota_aco.cli.run.ReportGenerator')
    @patch('rota_aco.cli.run.VisualizationEngine')
    def test_generate_reports_single_execution(self, mock_viz_engine, mock_report_gen):
        """Testa geração de relatórios para execução única."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_config = MetricsConfig()
            metrics_config.base_output_dir = temp_dir
            metrics_config.enable_visualizations = True
            
            # Mock successful execution data
            successful_execution = MagicMock()
            successful_execution.success = True
            successful_execution.iterations_data = [MagicMock(), MagicMock()]
            
            # Mock report generator
            mock_report_instance = MagicMock()
            mock_report_instance.generate_single_execution_report.return_value = "report.md"
            mock_report_gen.return_value = mock_report_instance
            
            # Mock visualization engine
            mock_viz_instance = MagicMock()
            mock_viz_instance.generate_convergence_plot.return_value = "convergence.png"
            mock_viz_engine.return_value = mock_viz_instance
            
            args = argparse.Namespace(verbose=False)
            
            # Execute function
            generate_metrics_reports([successful_execution], metrics_config, args)
            
            # Verify report generation was called
            mock_report_instance.generate_single_execution_report.assert_called_once_with(successful_execution)
            
            # Verify visualization generation was called
            mock_viz_instance.generate_convergence_plot.assert_called_once()
    
    @patch('rota_aco.cli.run.ReportGenerator')
    @patch('rota_aco.cli.run.VisualizationEngine')
    def test_generate_reports_multiple_executions(self, mock_viz_engine, mock_report_gen):
        """Testa geração de relatórios para múltiplas execuções."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_config = MetricsConfig()
            metrics_config.base_output_dir = temp_dir
            metrics_config.enable_visualizations = True
            
            # Mock multiple successful executions
            execution1 = MagicMock()
            execution1.success = True
            execution1.iterations_data = [MagicMock()]
            
            execution2 = MagicMock()
            execution2.success = True
            execution2.iterations_data = [MagicMock()]
            
            executions = [execution1, execution2]
            
            # Mock report generator
            mock_report_instance = MagicMock()
            mock_report_instance.generate_comparative_report.return_value = "comparison.md"
            mock_report_gen.return_value = mock_report_instance
            
            # Mock visualization engine
            mock_viz_instance = MagicMock()
            mock_viz_instance.generate_convergence_plot.return_value = "convergence.png"
            mock_viz_instance.generate_comparison_plots.return_value = ["comparison1.png", "comparison2.png"]
            mock_viz_engine.return_value = mock_viz_instance
            
            args = argparse.Namespace(verbose=False)
            
            # Execute function
            generate_metrics_reports(executions, metrics_config, args)
            
            # Verify comparative report generation was called
            mock_report_instance.generate_comparative_report.assert_called_once_with(executions)
            
            # Verify convergence plots were generated for each execution
            assert mock_viz_instance.generate_convergence_plot.call_count == 2
            
            # Verify comparison plots were generated
            mock_viz_instance.generate_comparison_plots.assert_called_once_with(executions)


class TestCLIIntegrationEndToEnd:
    """Testes de integração end-to-end do CLI com métricas."""
    
    @patch('rota_aco.cli.run.load_graph')
    @patch('rota_aco.cli.run.get_bus_stops')
    @patch('rota_aco.cli.run.find_nearest_node')
    @patch('rota_aco.cli.run.build_meta_graph')
    @patch('rota_aco.cli.run.run_aco_with_metrics')
    def test_cli_with_metrics_flag(self, mock_run_aco, mock_build_meta, mock_find_node, 
                                  mock_get_stops, mock_load_graph):
        """Testa execução completa do CLI com flag --metrics."""
        # Setup mocks
        mock_load_graph.return_value = MagicMock()
        mock_get_stops.return_value = [1, 2, 3]
        mock_find_node.side_effect = [1, 4]  # start and exit nodes
        
        mock_meta_graph = MagicMock()
        mock_meta_graph.nodes.return_value = [1, 2, 3, 4]
        mock_meta_graph.edges.return_value = [(1, 2), (2, 3), (3, 4)]
        mock_build_meta.return_value = (mock_meta_graph, {}, [2, 3], {}, None, None)
        
        mock_result = ([[1, 2, 3, 4]], 500.0, 1, 0.95)
        mock_execution_data = MagicMock()
        mock_execution_data.success = True
        mock_run_aco.return_value = (mock_result, mock_execution_data)
        
        # Test arguments
        test_args = [
            '--graph', 'test.graphml',
            '--start-lat', '0.0',
            '--start-lon', '0.0',
            '--exit-lat', '1.0',
            '--exit-lon', '1.0',
            '--metrics'
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_args.extend(['--report-output', temp_dir])
            
            # Mock sys.argv
            with patch.object(sys, 'argv', ['run.py'] + test_args):
                from rota_aco.cli.run import main
                
                # Should not raise exception
                try:
                    main()
                except SystemExit as e:
                    # main() calls sys.exit(0) on success
                    assert e.code == 0
                
                # Verify metrics integration was used
                mock_run_aco.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])