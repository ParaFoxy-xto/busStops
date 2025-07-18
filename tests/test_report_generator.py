"""
Testes unitários para o ReportGenerator.

Este módulo testa a funcionalidade de geração de relatórios em markdown,
incluindo formatação de tabelas, seções e estrutura geral.
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch

from src.rota_aco.metrics.report_generator import ReportGenerator, ReportSection
from src.rota_aco.metrics.data_models import (
    MetricsReport, ExecutionSummary, RouteQualityMetrics,
    ConvergenceMetrics, ComparativeMetrics, DomainMetrics
)
from src.rota_aco.metrics.config import MetricsConfig
from src.rota_aco.metrics.exceptions import ReportGenerationError


class TestReportGenerator:
    """Testes para a classe ReportGenerator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Configuração de teste."""
        config = MetricsConfig()
        config.base_output_dir = temp_dir
        config.reports_dir = 'reports'
        return config
    
    @pytest.fixture
    def sample_execution_summary(self):
        """Resumo de execução de exemplo."""
        return ExecutionSummary(
            total_executions=10,
            successful_executions=9,
            success_rate=0.9,
            avg_execution_time=45.5,
            avg_iterations_to_convergence=150.0,
            best_overall_fitness=0.8542,
            algorithm_type="ACS-TIME"
        )
    
    @pytest.fixture
    def sample_quality_metrics(self):
        """Métricas de qualidade de exemplo."""
        return RouteQualityMetrics(
            valid_routes_percentage=0.95,
            demand_coverage_percentage=0.88,
            vehicle_utilization_efficiency=0.82,
            capacity_violations=2,
            opposite_stops_violations=1,
            average_route_length=12.5,
            route_length_variance=3.2,
            load_balancing_index=0.75
        )
    
    @pytest.fixture
    def sample_convergence_metrics(self):
        """Métricas de convergência de exemplo."""
        return ConvergenceMetrics(
            convergence_point=120,
            final_stability=0.005,
            improvement_rate=0.02,
            plateau_detection=False,
            convergence_speed=0.6,
            total_iterations=200,
            best_fitness_evolution=[0.5, 0.7, 0.8, 0.85],
            avg_fitness_evolution=[0.4, 0.6, 0.75, 0.82],
            variance_evolution=[0.1, 0.08, 0.05, 0.02]
        )
    
    @pytest.fixture
    def sample_comparative_metrics(self):
        """Métricas comparativas de exemplo."""
        return ComparativeMetrics(
            total_executions=10,
            successful_executions=9,
            success_rate=0.9,
            avg_execution_time=45.5,
            std_execution_time=5.2,
            avg_best_fitness=0.8542,
            std_best_fitness=0.0234,
            median_best_fitness=0.8567,
            avg_convergence_iterations=150.0,
            relative_efficiency=1.15
        )
    
    @pytest.fixture
    def sample_domain_metrics(self):
        """Métricas de domínio de exemplo."""
        return DomainMetrics(
            estimated_travel_time=35.5,
            average_transfers=1.2,
            geographic_coverage=25.8,
            load_balancing_index=0.75,
            energy_efficiency=2.3,
            accessibility_index=3.5,
            service_frequency=4.2,
            route_overlap_percentage=0.15
        )
    
    @pytest.fixture
    def sample_metrics_report(self, sample_execution_summary, sample_quality_metrics,
                             sample_convergence_metrics, sample_comparative_metrics,
                             sample_domain_metrics):
        """Relatório de métricas completo de exemplo."""
        return MetricsReport(
            execution_summary=sample_execution_summary,
            quality_metrics=sample_quality_metrics,
            convergence_analysis=sample_convergence_metrics,
            comparative_analysis=sample_comparative_metrics,
            domain_metrics=sample_domain_metrics,
            visualizations=['chart1.png', 'chart2.svg'],
            config_used={'param1': 'value1', 'param2': 42}
        )
    
    def test_init(self, config):
        """Testa inicialização do ReportGenerator."""
        generator = ReportGenerator(config)
        assert generator.config == config
        assert os.path.exists(config.get_reports_path())
    
    def test_generate_report_basic(self, config, sample_metrics_report):
        """Testa geração básica de relatório."""
        generator = ReportGenerator(config)
        
        output_path = generator.generate_report(sample_metrics_report, "test_report.md")
        
        assert os.path.exists(output_path)
        assert output_path.endswith("test_report.md")
        
        # Verificar conteúdo básico
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "# Relatório de Métricas - ACS-TIME" in content
        assert "## Resumo Executivo" in content
        assert "## Métricas de Qualidade das Rotas" in content
        assert "## Análise de Convergência" in content
        assert "## Conclusões e Recomendações" in content
    
    def test_generate_report_auto_filename(self, config, sample_metrics_report):
        """Testa geração de relatório com nome automático."""
        generator = ReportGenerator(config)
        
        output_path = generator.generate_report(sample_metrics_report)
        
        assert os.path.exists(output_path)
        filename = os.path.basename(output_path)
        assert filename.startswith("metrics_report_acs_time_")
        assert filename.endswith(".md")
    
    def test_generate_header_section(self, config, sample_metrics_report):
        """Testa geração da seção de cabeçalho."""
        generator = ReportGenerator(config)
        
        section = generator._generate_header_section(sample_metrics_report)
        
        assert section.title == "Header"
        assert section.level == 1
        assert "# Relatório de Métricas - ACS-TIME" in section.content
        assert "**Execuções Analisadas:** 10" in section.content
        assert "**Taxa de Sucesso:** 90.0%" in section.content
    
    def test_generate_executive_summary_section(self, config, sample_execution_summary):
        """Testa geração da seção de resumo executivo."""
        generator = ReportGenerator(config)
        
        section = generator._generate_executive_summary_section(sample_execution_summary)
        
        assert section.title == "Resumo Executivo"
        assert "## Resumo Executivo" in section.content
        assert "| **Total de Execuções** | 10 |" in section.content
        assert "| **Taxa de Sucesso** | 90.0% |" in section.content
        assert "| **Tempo Médio de Execução** | 45.50s |" in section.content
    
    def test_generate_quality_metrics_section(self, config, sample_quality_metrics):
        """Testa geração da seção de métricas de qualidade."""
        generator = ReportGenerator(config)
        
        section = generator._generate_quality_metrics_section(sample_quality_metrics)
        
        assert section.title == "Métricas de Qualidade"
        assert "## Métricas de Qualidade das Rotas" in section.content
        assert "| **Rotas Válidas** | 95.0% |" in section.content
        assert "| **Violações de Capacidade** | 2 |" in section.content
        assert "| **Comprimento Médio das Rotas** | 12.50 km |" in section.content
    
    def test_generate_convergence_analysis_section(self, config, sample_convergence_metrics):
        """Testa geração da seção de análise de convergência."""
        generator = ReportGenerator(config)
        
        section = generator._generate_convergence_analysis_section(sample_convergence_metrics)
        
        assert section.title == "Análise de Convergência"
        assert "## Análise de Convergência" in section.content
        assert "| **Ponto de Convergência** | Iteração 120 |" in section.content
        assert "| **Plateau Detectado** | Não detectado |" in section.content
        assert "Convergência rápida" in section.content  # baseado na velocidade 0.6
    
    def test_generate_comparative_analysis_section(self, config, sample_comparative_metrics):
        """Testa geração da seção de análise comparativa."""
        generator = ReportGenerator(config)
        
        section = generator._generate_comparative_analysis_section(sample_comparative_metrics)
        
        assert section.title == "Análise Comparativa"
        assert "## Análise Comparativa" in section.content
        assert "| **Taxa de Sucesso** | 90.0% |" in section.content
        assert "| **Fitness Médio ± Desvio** | 0.8542 ± 0.0234 |" in section.content
        assert "alta consistência" in section.content  # baseado no std baixo
    
    def test_generate_domain_metrics_section(self, config, sample_domain_metrics):
        """Testa geração da seção de métricas de domínio."""
        generator = ReportGenerator(config)
        
        section = generator._generate_domain_metrics_section(sample_domain_metrics)
        
        assert section.title == "Métricas de Transporte"
        assert "## Métricas de Transporte Público" in section.content
        assert "| **Tempo Total de Viagem Estimado** | 35.5 min |" in section.content
        assert "| **Eficiência Energética** | 2.30 km/passageiro |" in section.content
        assert "Boa densidade de paradas" in section.content  # baseado no índice 3.5
    
    def test_generate_visualizations_section(self, config):
        """Testa geração da seção de visualizações."""
        generator = ReportGenerator(config)
        visualizations = ['charts/convergence.png', 'charts/comparison.svg']
        
        section = generator._generate_visualizations_section(visualizations)
        
        assert section.title == "Visualizações"
        assert "## Visualizações Geradas" in section.content
        assert "1. **convergence.png**" in section.content
        assert "2. **comparison.svg**" in section.content
        assert "Como Interpretar as Visualizações" in section.content
    
    def test_generate_configuration_section(self, config):
        """Testa geração da seção de configuração."""
        generator = ReportGenerator(config)
        test_config = {'param1': 'value1', 'param2': 42}
        
        section = generator._generate_configuration_section(test_config)
        
        assert section.title == "Configuração"
        assert "## Configuração Utilizada" in section.content
        assert "```json" in section.content
        assert '"param1": "value1"' in section.content
        assert '"param2": 42' in section.content
    
    def test_generate_configuration_section_empty(self, config):
        """Testa geração da seção de configuração vazia."""
        generator = ReportGenerator(config)
        
        section = generator._generate_configuration_section({})
        
        assert "Configuração padrão utilizada" in section.content
    
    def test_generate_conclusions_section(self, config, sample_metrics_report):
        """Testa geração da seção de conclusões."""
        generator = ReportGenerator(config)
        
        section = generator._generate_conclusions_section(sample_metrics_report)
        
        assert section.title == "Conclusões"
        assert "## Conclusões e Recomendações" in section.content
        assert "### Performance Geral" in section.content
        assert "### Qualidade das Soluções" in section.content
        assert "### Comportamento de Convergência" in section.content
        assert "### Adequação para Apresentação Acadêmica" in section.content
    
    def test_generate_conclusions_with_recommendations(self, config, sample_metrics_report):
        """Testa geração de conclusões com recomendações."""
        # Modificar métricas para gerar recomendações
        sample_metrics_report.execution_summary.success_rate = 0.6  # Baixa taxa de sucesso
        sample_metrics_report.quality_metrics.valid_routes_percentage = 0.7  # Baixa qualidade
        
        generator = ReportGenerator(config)
        section = generator._generate_conclusions_section(sample_metrics_report)
        
        assert "### Recomendações" in section.content
        assert "Investigar causas de falha" in section.content
        assert "Revisar restrições" in section.content
    
    def test_generate_summary_table(self, config, sample_metrics_report):
        """Testa geração de tabela resumo comparativa."""
        generator = ReportGenerator(config)
        
        # Criar segundo relatório
        report2 = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=5,
                successful_executions=4,
                success_rate=0.8,
                avg_execution_time=52.3,
                avg_iterations_to_convergence=180.0,
                best_overall_fitness=0.8234,
                algorithm_type="ACS-VEHICLE"
            ),
            quality_metrics=RouteQualityMetrics(
                valid_routes_percentage=0.92,
                demand_coverage_percentage=0.85,
                vehicle_utilization_efficiency=0.78,
                capacity_violations=3,
                opposite_stops_violations=0,
                average_route_length=11.8,
                route_length_variance=2.9,
                load_balancing_index=0.72
            ),
            convergence_analysis=sample_metrics_report.convergence_analysis,
            comparative_analysis=None,
            domain_metrics=sample_metrics_report.domain_metrics
        )
        
        table = generator.generate_summary_table([sample_metrics_report, report2])
        
        assert "## Tabela Resumo Comparativa" in table
        assert "| ACS-TIME | 10 | 90.0% | 0.8542 | 45.50s | 95.0% |" in table
        assert "| ACS-VEHICLE | 5 | 80.0% | 0.8234 | 52.30s | 92.0% |" in table
    
    def test_generate_summary_table_empty(self, config):
        """Testa geração de tabela resumo com lista vazia."""
        generator = ReportGenerator(config)
        
        table = generator.generate_summary_table([])
        
        assert "Nenhum relatório fornecido para comparação" in table
    
    def test_report_structure_completeness(self, config, sample_metrics_report):
        """Testa se o relatório contém todas as seções esperadas."""
        generator = ReportGenerator(config)
        
        output_path = generator.generate_report(sample_metrics_report)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar presença de todas as seções principais
        expected_sections = [
            "# Relatório de Métricas",
            "## Resumo Executivo",
            "## Métricas de Qualidade das Rotas",
            "## Análise de Convergência",
            "## Análise Comparativa",
            "## Métricas de Transporte Público",
            "## Visualizações Geradas",
            "## Configuração Utilizada",
            "## Conclusões e Recomendações"
        ]
        
        for section in expected_sections:
            assert section in content, f"Seção '{section}' não encontrada no relatório"
    
    def test_markdown_table_formatting(self, config, sample_metrics_report):
        """Testa formatação correta das tabelas markdown."""
        generator = ReportGenerator(config)
        
        output_path = generator.generate_report(sample_metrics_report)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar formatação de tabelas
        assert "| Métrica | Valor |" in content
        assert "|---------|-------|" in content
        assert "| **" in content  # Texto em negrito nas tabelas
    
    def test_error_handling(self, config):
        """Testa tratamento de erros na geração de relatórios."""
        generator = ReportGenerator(config)
        
        # Criar relatório inválido
        invalid_report = None
        
        with pytest.raises(ReportGenerationError):
            generator.generate_report(invalid_report)
    
    @patch('builtins.open', side_effect=IOError("Erro de escrita"))
    def test_file_write_error(self, mock_open, config, sample_metrics_report):
        """Testa tratamento de erro de escrita de arquivo."""
        generator = ReportGenerator(config)
        
        with pytest.raises(ReportGenerationError):
            generator.generate_report(sample_metrics_report)
    
    def test_report_section_dataclass(self):
        """Testa a dataclass ReportSection."""
        section = ReportSection("Test Title", "Test content", 3)
        
        assert section.title == "Test Title"
        assert section.content == "Test content"
        assert section.level == 3
        
        # Testar valor padrão
        section_default = ReportSection("Title", "Content")
        assert section_default.level == 2
    
    def test_generate_comprehensive_report(self, config, sample_metrics_report):
        """Testa geração de relatório abrangente."""
        generator = ReportGenerator(config)
        
        # Criar segundo relatório para comparação
        report2 = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=5,
                successful_executions=4,
                success_rate=0.8,
                avg_execution_time=52.3,
                avg_iterations_to_convergence=180.0,
                best_overall_fitness=0.8234,
                algorithm_type="ACS-VEHICLE"
            ),
            quality_metrics=RouteQualityMetrics(
                valid_routes_percentage=0.92,
                demand_coverage_percentage=0.85,
                vehicle_utilization_efficiency=0.78,
                capacity_violations=3,
                opposite_stops_violations=0,
                average_route_length=11.8,
                route_length_variance=2.9,
                load_balancing_index=0.72
            ),
            convergence_analysis=sample_metrics_report.convergence_analysis,
            comparative_analysis=None,
            domain_metrics=sample_metrics_report.domain_metrics
        )
        
        output_path = generator.generate_comprehensive_report([sample_metrics_report, report2])
        
        assert os.path.exists(output_path)
        assert output_path.endswith(".md")
        
        # Verificar conteúdo
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "# Relatório Comparativo de Algoritmos ACO" in content
        assert "## Tabela Resumo Comparativa" in content
        assert "## Análise Detalhada por Algoritmo" in content
        assert "## Conclusões Comparativas" in content
        assert "## Recomendações Baseadas na Comparação" in content
        assert "ACS-TIME" in content
        assert "ACS-VEHICLE" in content
    
    def test_generate_comprehensive_report_empty_list(self, config):
        """Testa geração de relatório abrangente com lista vazia."""
        generator = ReportGenerator(config)
        
        with pytest.raises(ReportGenerationError):
            generator.generate_comprehensive_report([])
    
    def test_generate_comparative_header(self, config, sample_metrics_report):
        """Testa geração de cabeçalho comparativo."""
        generator = ReportGenerator(config)
        
        # Criar segundo relatório
        report2 = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=5,
                successful_executions=4,
                success_rate=0.8,
                avg_execution_time=52.3,
                avg_iterations_to_convergence=180.0,
                best_overall_fitness=0.8234,
                algorithm_type="ACS-VEHICLE"
            ),
            quality_metrics=sample_metrics_report.quality_metrics,
            convergence_analysis=sample_metrics_report.convergence_analysis,
            comparative_analysis=None,
            domain_metrics=sample_metrics_report.domain_metrics
        )
        
        header = generator._generate_comparative_header([sample_metrics_report, report2])
        
        assert "# Relatório Comparativo de Algoritmos ACO" in header
        assert "ACS-TIME, ACS-VEHICLE" in header
        assert "Total de Execuções Analisadas:** 15" in header  # 10 + 5
    
    def test_generate_algorithm_summary(self, config, sample_metrics_report):
        """Testa geração de resumo de algoritmo."""
        generator = ReportGenerator(config)
        
        summary = generator._generate_algorithm_summary(sample_metrics_report)
        
        assert "**Performance Geral:**" in summary
        assert "Taxa de Sucesso: 90.0%" in summary
        assert "**Qualidade das Soluções:**" in summary
        assert "Rotas Válidas: 95.0%" in summary
        assert "**Convergência:**" in summary
        assert "Ponto de Convergência: Iteração 120" in summary
    
    def test_generate_comparative_conclusions(self, config, sample_metrics_report):
        """Testa geração de conclusões comparativas."""
        generator = ReportGenerator(config)
        
        # Criar relatórios com diferentes características
        fast_report = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=5,
                successful_executions=5,
                success_rate=1.0,
                avg_execution_time=20.0,  # Mais rápido
                avg_iterations_to_convergence=100.0,
                best_overall_fitness=0.8000,  # Fitness menor
                algorithm_type="FAST-ACO"
            ),
            quality_metrics=RouteQualityMetrics(
                valid_routes_percentage=0.85,  # Qualidade menor
                demand_coverage_percentage=0.80,
                vehicle_utilization_efficiency=0.75,
                capacity_violations=5,
                opposite_stops_violations=2,
                average_route_length=10.0,
                route_length_variance=2.0,
                load_balancing_index=0.65
            ),
            convergence_analysis=sample_metrics_report.convergence_analysis,
            comparative_analysis=None,
            domain_metrics=sample_metrics_report.domain_metrics
        )
        
        conclusions = generator._generate_comparative_conclusions([sample_metrics_report, fast_report])
        
        assert "## Conclusões Comparativas" in conclusions
        assert "### Melhores Performances por Critério" in conclusions
        assert "**Maior Taxa de Sucesso:** FAST-ACO (100.0%)" in conclusions
        assert "**Melhor Fitness:** ACS-TIME (0.8542)" in conclusions
        assert "**Mais Rápido:** FAST-ACO (20.00s)" in conclusions
        assert "**Melhor Qualidade:** ACS-TIME (95.0%)" in conclusions
        assert "### Análise de Trade-offs" in conclusions
    
    def test_generate_comparative_recommendations(self, config, sample_metrics_report):
        """Testa geração de recomendações comparativas."""
        generator = ReportGenerator(config)
        
        # Criar relatório com problemas para gerar recomendações
        problematic_report = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=10,
                successful_executions=6,  # Taxa baixa
                success_rate=0.6,
                avg_execution_time=150.0,  # Muito lento
                avg_iterations_to_convergence=300.0,
                best_overall_fitness=0.7000,
                algorithm_type="SLOW-ACO"
            ),
            quality_metrics=sample_metrics_report.quality_metrics,
            convergence_analysis=sample_metrics_report.convergence_analysis,
            comparative_analysis=None,
            domain_metrics=sample_metrics_report.domain_metrics
        )
        
        recommendations = generator._generate_comparative_recommendations([sample_metrics_report, problematic_report])
        
        assert "## Recomendações Baseadas na Comparação" in recommendations
        assert "**Algoritmo Recomendado:**" in recommendations
        assert "Investigar baixa taxa de sucesso" in recommendations
        assert "Otimizar performance" in recommendations
        assert "**Para Apresentação Acadêmica:**" in recommendations
        assert "**Para Validação Científica:**" in recommendations
    
    def test_generate_report_metadata(self, config, sample_metrics_report):
        """Testa geração de metadados do relatório."""
        generator = ReportGenerator(config)
        
        metadata = generator._generate_report_metadata([sample_metrics_report])
        
        assert "## Metadados do Relatório" in metadata
        assert "### Informações de Geração" in metadata
        assert "**Algoritmos Analisados:** 1" in metadata
        assert "**Total de Execuções:** 10" in metadata
        assert "**Total de Visualizações:** 2" in metadata
        assert "### Configurações Utilizadas" in metadata
        assert "### Arquivos de Visualização" in metadata
        assert "**ACS-TIME:**" in metadata
    
    def test_export_to_json(self, config, sample_metrics_report):
        """Testa exportação para JSON."""
        generator = ReportGenerator(config)
        
        output_path = generator.export_to_json(sample_metrics_report, "test_export.json")
        
        assert os.path.exists(output_path)
        assert output_path.endswith("test_export.json")
        
        # Verificar conteúdo JSON
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'execution_summary' in data
        assert 'quality_metrics' in data
        assert 'convergence_analysis' in data
        assert 'domain_metrics' in data
        assert data['execution_summary']['algorithm_type'] == 'ACS-TIME'
        assert data['execution_summary']['total_executions'] == 10
    
    def test_export_to_json_auto_filename(self, config, sample_metrics_report):
        """Testa exportação para JSON com nome automático."""
        generator = ReportGenerator(config)
        
        output_path = generator.export_to_json(sample_metrics_report)
        
        assert os.path.exists(output_path)
        filename = os.path.basename(output_path)
        assert filename.startswith("metrics_data_acs_time_")
        assert filename.endswith(".json")
    
    def test_comprehensive_report_with_visualizations(self, config, sample_metrics_report):
        """Testa relatório abrangente com visualizações."""
        generator = ReportGenerator(config)
        
        # Adicionar mais visualizações
        sample_metrics_report.visualizations = [
            'charts/convergence_acs_time.png',
            'charts/quality_acs_time.svg',
            'charts/comparison.png'
        ]
        
        output_path = generator.generate_comprehensive_report([sample_metrics_report])
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "**Visualizações:** 3 gráficos gerados" in content
        assert "**Total de Visualizações:** 3" in content
    
    def test_comprehensive_report_metadata_with_multiple_configs(self, config):
        """Testa metadados com múltiplas configurações."""
        generator = ReportGenerator(config)
        
        # Criar relatórios com configurações diferentes
        report1 = MetricsReport(
            execution_summary=ExecutionSummary(
                total_executions=5,
                successful_executions=5,
                success_rate=1.0,
                avg_execution_time=30.0,
                avg_iterations_to_convergence=100.0,
                best_overall_fitness=0.8500,
                algorithm_type="CONFIG-1"
            ),
            quality_metrics=RouteQualityMetrics(
                valid_routes_percentage=0.95,
                demand_coverage_percentage=0.88,
                vehicle_utilization_efficiency=0.82,
                capacity_violations=1,
                opposite_stops_violations=0,
                average_route_length=12.0,
                route_length_variance=2.5,
                load_balancing_index=0.80
            ),
            convergence_analysis=ConvergenceMetrics(
                convergence_point=80,
                final_stability=0.003,
                improvement_rate=0.025,
                plateau_detection=False,
                convergence_speed=0.7,
                total_iterations=150
            ),
            comparative_analysis=None,
            domain_metrics=DomainMetrics(
                estimated_travel_time=30.0,
                average_transfers=1.0,
                geographic_coverage=20.0,
                load_balancing_index=0.80,
                energy_efficiency=2.0,
                accessibility_index=4.0,
                service_frequency=5.0,
                route_overlap_percentage=0.10
            ),
            config_used={'algorithm': 'config1', 'iterations': 150}
        )
        
        metadata = generator._generate_report_metadata([report1])
        
        assert "### Configurações Utilizadas" in metadata
        assert '"algorithm": "config1"' in metadata
        assert '"iterations": 150' in metadata
    
    def test_error_handling_comprehensive_report(self, config):
        """Testa tratamento de erros no relatório abrangente."""
        generator = ReportGenerator(config)
        
        # Teste com lista vazia
        with pytest.raises(ReportGenerationError):
            generator.generate_comprehensive_report([])
    
    @patch('builtins.open', side_effect=IOError("Erro de escrita"))
    def test_comprehensive_report_write_error(self, mock_open, config, sample_metrics_report):
        """Testa tratamento de erro de escrita no relatório abrangente."""
        generator = ReportGenerator(config)
        
        with pytest.raises(ReportGenerationError):
            generator.generate_comprehensive_report([sample_metrics_report])
    
    @patch('builtins.open', side_effect=IOError("Erro de escrita"))
    def test_json_export_write_error(self, mock_open, config, sample_metrics_report):
        """Testa tratamento de erro de escrita na exportação JSON."""
        generator = ReportGenerator(config)
        
        with pytest.raises(ReportGenerationError):
            generator.export_to_json(sample_metrics_report)


if __name__ == "__main__":
    pytest.main([__file__])