"""
Testes para o sistema de análise estatística comparativa.

Este módulo testa todas as funcionalidades do StatisticalAnalyzer,
incluindo análise de múltiplas execuções, comparações side-by-side
e cálculo de métricas de eficiência relativa.
"""

import pytest
from datetime import datetime
from typing import List
from src.rota_aco.metrics.statistical_analyzer import StatisticalAnalyzer
from src.rota_aco.metrics.data_models import ExecutionData, Solution, Route, ComparativeMetrics
from src.rota_aco.metrics.exceptions import MetricsCalculationError


class TestStatisticalAnalyzer:
    """Testes para a classe StatisticalAnalyzer."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.analyzer = StatisticalAnalyzer()
    
    def create_mock_solution(self, fitness_time: float = 100.0, fitness_vehicle: float = 5.0) -> Solution:
        """Cria uma solução mock para testes."""
        route = Route(
            stops=[1, 2, 3],
            distances=[10.0, 15.0],
            passenger_load=[30, 40],
            total_distance=25.0,
            total_passengers=70,
            is_valid=True
        )
        
        return Solution(
            routes=[route],
            total_vehicles=1,
            total_distance=25.0,
            total_passengers_served=70,
            fitness_time=fitness_time,
            fitness_vehicle=fitness_vehicle,
            is_feasible=True
        )
    
    def create_successful_execution(self, execution_time: float = 120.0, 
                                  fitness_time: float = 100.0,
                                  fitness_vehicle: float = 5.0) -> ExecutionData:
        """Cria uma execução bem-sucedida para testes."""
        solution = self.create_mock_solution(fitness_time, fitness_vehicle)
        
        return ExecutionData(
            algorithm_type="ACS-TIME",
            execution_time=execution_time,
            final_solution=solution,
            success=True,
            iterations_data=[]  # Simplified for testing
        )
    
    def create_failed_execution(self, error_message: str = "Test error") -> ExecutionData:
        """Cria uma execução falhada para testes."""
        return ExecutionData(
            algorithm_type="ACS-TIME",
            execution_time=0.0,
            success=False,
            error_message=error_message
        )
    
    def create_multiple_executions(self, count: int, success_rate: float = 1.0) -> List[ExecutionData]:
        """Cria múltiplas execuções para testes."""
        executions = []
        successful_count = int(count * success_rate)
        
        # Criar execuções bem-sucedidas
        for i in range(successful_count):
            execution_time = 100.0 + i * 10.0  # Tempos variados
            fitness_time = 150.0 - i * 5.0     # Fitness melhorando
            fitness_vehicle = 5.0 + i * 0.5    # Fitness piorando
            
            executions.append(self.create_successful_execution(execution_time, fitness_time, fitness_vehicle))
        
        # Criar execuções falhadas
        for i in range(count - successful_count):
            executions.append(self.create_failed_execution(f"Error type {i % 3}"))
        
        return executions
    
    def test_analyze_empty_executions(self):
        """Testa análise com lista vazia de execuções."""
        metrics = self.analyzer.analyze_multiple_executions([])
        
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_execution_time == 0.0
        assert metrics.avg_best_fitness == 0.0
        assert metrics.relative_efficiency == 1.0
    
    def test_analyze_single_successful_execution(self):
        """Testa análise com uma única execução bem-sucedida."""
        execution = self.create_successful_execution(120.0, 100.0, 5.0)
        metrics = self.analyzer.analyze_multiple_executions([execution])
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.success_rate == 100.0
        assert metrics.avg_execution_time == 120.0
        assert metrics.std_execution_time == 0.0  # Apenas uma execução
        assert metrics.avg_best_fitness == 5.0    # Menor entre time e vehicle
        assert metrics.median_best_fitness == 5.0
    
    def test_analyze_single_failed_execution(self):
        """Testa análise com uma única execução falhada."""
        execution = self.create_failed_execution()
        metrics = self.analyzer.analyze_multiple_executions([execution])
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_execution_time == 0.0
        assert metrics.avg_best_fitness == 0.0
    
    def test_analyze_multiple_successful_executions(self):
        """Testa análise com múltiplas execuções bem-sucedidas."""
        executions = self.create_multiple_executions(5, success_rate=1.0)
        metrics = self.analyzer.analyze_multiple_executions(executions)
        
        assert metrics.total_executions == 5
        assert metrics.successful_executions == 5
        assert metrics.success_rate == 100.0
        assert metrics.avg_execution_time > 0
        assert metrics.std_execution_time > 0  # Deve haver variação
        assert metrics.avg_best_fitness > 0
        assert metrics.avg_convergence_iterations == 0  # Sem dados de iteração
    
    def test_analyze_mixed_executions(self):
        """Testa análise com mix de execuções bem-sucedidas e falhadas."""
        executions = self.create_multiple_executions(10, success_rate=0.7)  # 70% sucesso
        metrics = self.analyzer.analyze_multiple_executions(executions)
        
        assert metrics.total_executions == 10
        assert metrics.successful_executions == 7
        assert metrics.success_rate == 70.0
        assert metrics.avg_execution_time > 0  # Apenas execuções bem-sucedidas
        assert metrics.avg_best_fitness > 0
    
    def test_compare_algorithm_configurations(self):
        """Testa comparação entre duas configurações."""
        config_a_executions = self.create_multiple_executions(5, success_rate=0.8)
        config_b_executions = self.create_multiple_executions(5, success_rate=1.0)
        
        comparison = self.analyzer.compare_algorithm_configurations(
            config_a_executions, config_b_executions, "Config A", "Config B"
        )
        
        assert "Config A" in comparison
        assert "Config B" in comparison
        assert "comparison" in comparison
        
        # Verificar estrutura das métricas
        assert "metrics" in comparison["Config A"]
        assert "summary" in comparison["Config A"]
        assert isinstance(comparison["Config A"]["metrics"], ComparativeMetrics)
        
        # Verificar comparação
        comp = comparison["comparison"]
        assert "success_rate_difference" in comp
        assert "avg_time_difference" in comp
        assert "relative_efficiency" in comp
        assert "better_config" in comp
        
        # Config B deve ter melhor taxa de sucesso
        assert comp["success_rate_difference"] > 0  # B - A > 0
    
    def test_calculate_aggregated_statistics(self):
        """Testa cálculo de estatísticas agregadas."""
        executions = self.create_multiple_executions(8, success_rate=0.75)
        stats = self.analyzer.calculate_aggregated_statistics(executions)
        
        assert "execution_time" in stats
        assert "fitness" in stats
        assert "convergence" in stats
        assert "success" in stats
        
        # Verificar estatísticas de execução
        exec_stats = stats["execution_time"]
        assert "mean" in exec_stats
        assert "median" in exec_stats
        assert "std_dev" in exec_stats
        assert "min" in exec_stats
        assert "max" in exec_stats
        
        # Verificar estatísticas de sucesso
        success_stats = stats["success"]
        assert success_stats["total_executions"] == 8
        assert success_stats["successful_executions"] == 6  # 75% de 8
        assert success_stats["success_rate"] == 75.0
        assert success_stats["failure_rate"] == 25.0
    
    def test_calculate_aggregated_statistics_empty(self):
        """Testa estatísticas agregadas com lista vazia."""
        stats = self.analyzer.calculate_aggregated_statistics([])
        assert stats == {}
    
    def test_calculate_success_rate(self):
        """Testa cálculo detalhado da taxa de sucesso."""
        executions = self.create_multiple_executions(10, success_rate=0.6)
        success_data = self.analyzer.calculate_success_rate(executions)
        
        assert success_data["success_rate"] == 60.0
        assert success_data["failure_rate"] == 40.0
        assert success_data["total_executions"] == 10
        assert success_data["successful_executions"] == 6
        assert success_data["failed_executions"] == 4
        assert "error_types" in success_data
    
    def test_calculate_success_rate_empty(self):
        """Testa cálculo de taxa de sucesso com lista vazia."""
        success_data = self.analyzer.calculate_success_rate([])
        
        assert success_data["success_rate"] == 0.0
        assert success_data["total_executions"] == 0
        assert success_data["successful_executions"] == 0
    
    def test_calculate_relative_efficiency(self):
        """Testa cálculo de eficiência relativa."""
        # Baseline: execuções mais lentas mas com melhor qualidade (fitness menor)
        baseline_executions = [
            self.create_successful_execution(200.0, 50.0, 3.0),  # Lento, boa qualidade (fitness=3.0)
            self.create_successful_execution(180.0, 55.0, 3.5)   # fitness=3.5
        ]
        
        # Comparação: execuções mais rápidas mas com qualidade pior (fitness maior)
        comparison_executions = [
            self.create_successful_execution(100.0, 80.0, 6.0),  # Rápido, qualidade pior (fitness=6.0)
            self.create_successful_execution(120.0, 85.0, 6.5)   # fitness=6.5
        ]
        
        efficiency = self.analyzer.calculate_relative_efficiency(
            baseline_executions, comparison_executions
        )
        
        assert "time_efficiency" in efficiency
        assert "quality_efficiency" in efficiency
        assert "success_efficiency" in efficiency
        assert "overall_efficiency" in efficiency
        
        # Baseline é mais lento, então time_efficiency > 1
        assert efficiency["time_efficiency"] > 1.0
        
        # Baseline tem melhor qualidade (fitness menor)
        # A fórmula é baseline/comparison, então se baseline < comparison, resultado < 1
        assert efficiency["quality_efficiency"] < 1.0  # Baseline é melhor (menor fitness)
        
        # Ambos têm 100% sucesso, então success_efficiency = 1
        assert efficiency["success_efficiency"] == 1.0
    
    def test_generate_comparison_report(self):
        """Testa geração de relatório de comparação."""
        config_a_executions = self.create_multiple_executions(3, success_rate=1.0)
        config_b_executions = self.create_multiple_executions(3, success_rate=0.67)
        
        comparison = self.analyzer.compare_algorithm_configurations(
            config_a_executions, config_b_executions, "Algoritmo A", "Algoritmo B"
        )
        
        report = self.analyzer.generate_comparison_report(comparison)
        
        assert "Relatório de Comparação de Configurações" in report
        assert "Algoritmo A" in report
        assert "Algoritmo B" in report
        assert "Análise Comparativa" in report
        assert "Configuração Superior" in report
        assert isinstance(report, str)
        assert len(report) > 200  # Relatório substancial
    
    def test_generate_comparison_report_empty(self):
        """Testa geração de relatório com dados vazios."""
        report = self.analyzer.generate_comparison_report({})
        assert "Nenhum dado de comparação disponível" in report
    
    def test_descriptive_stats_calculation(self):
        """Testa cálculo de estatísticas descritivas."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = self.analyzer._calculate_descriptive_stats(values, "Test")
        
        assert stats["count"] == 5
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["std_dev"] > 0
        assert stats["variance"] > 0
    
    def test_descriptive_stats_empty(self):
        """Testa estatísticas descritivas com lista vazia."""
        stats = self.analyzer._calculate_descriptive_stats([], "Empty")
        
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std_dev"] == 0.0
    
    def test_descriptive_stats_single_value(self):
        """Testa estatísticas descritivas com um único valor."""
        stats = self.analyzer._calculate_descriptive_stats([42.0], "Single")
        
        assert stats["count"] == 1
        assert stats["mean"] == 42.0
        assert stats["median"] == 42.0
        assert stats["std_dev"] == 0.0  # Sem variação
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
    
    def test_determine_better_config(self):
        """Testa determinação da melhor configuração."""
        # Config A: melhor em todos os aspectos
        metrics_a = ComparativeMetrics(
            total_executions=5, successful_executions=5, success_rate=100.0,
            avg_execution_time=100.0, std_execution_time=10.0,
            avg_best_fitness=50.0, std_best_fitness=5.0,
            median_best_fitness=50.0, avg_convergence_iterations=100.0
        )
        
        # Config B: pior em todos os aspectos
        metrics_b = ComparativeMetrics(
            total_executions=5, successful_executions=4, success_rate=80.0,
            avg_execution_time=150.0, std_execution_time=15.0,
            avg_best_fitness=70.0, std_best_fitness=8.0,
            median_best_fitness=70.0, avg_convergence_iterations=120.0
        )
        
        better = self.analyzer._determine_better_config(metrics_a, metrics_b, "A", "B")
        assert better == "A"
        
        # Teste de empate
        better_tie = self.analyzer._determine_better_config(metrics_a, metrics_a, "A", "B")
        assert better_tie == "Empate"
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Criar execução com dados problemáticos
        problematic_execution = ExecutionData(
            algorithm_type="TEST",
            execution_time=-1.0,  # Tempo inválido
            success=True,
            final_solution=None  # Sem solução apesar de sucesso
        )
        
        # O analisador deve lidar com isso graciosamente
        try:
            metrics = self.analyzer.analyze_multiple_executions([problematic_execution])
            assert isinstance(metrics, ComparativeMetrics)
        except MetricsCalculationError:
            # Ou lançar exceção apropriada
            pass
    
    def test_fitness_selection_logic(self):
        """Testa lógica de seleção do melhor fitness."""
        # Fitness time menor (melhor)
        execution1 = self.create_successful_execution(100.0, 50.0, 80.0)
        metrics1 = self.analyzer.analyze_multiple_executions([execution1])
        assert metrics1.avg_best_fitness == 50.0  # Deve escolher o menor
        
        # Fitness vehicle menor (melhor)
        execution2 = self.create_successful_execution(100.0, 90.0, 40.0)
        metrics2 = self.analyzer.analyze_multiple_executions([execution2])
        assert metrics2.avg_best_fitness == 40.0  # Deve escolher o menor
    
    def test_comprehensive_comparison(self):
        """Teste abrangente de comparação entre configurações."""
        # Configuração A: Rápida mas menos confiável
        config_a = [
            self.create_successful_execution(80.0, 60.0, 4.0),
            self.create_successful_execution(90.0, 65.0, 4.2),
            self.create_failed_execution("Timeout"),
            self.create_successful_execution(85.0, 62.0, 4.1)
        ]
        
        # Configuração B: Mais lenta mas mais confiável
        config_b = [
            self.create_successful_execution(120.0, 45.0, 3.0),
            self.create_successful_execution(130.0, 48.0, 3.2),
            self.create_successful_execution(125.0, 46.0, 3.1),
            self.create_successful_execution(128.0, 47.0, 3.15)
        ]
        
        comparison = self.analyzer.compare_algorithm_configurations(
            config_a, config_b, "Rápido", "Confiável"
        )
        
        # Verificar que a comparação captura as diferenças esperadas
        assert comparison["Rápido"]["metrics"].success_rate < 100.0
        assert comparison["Confiável"]["metrics"].success_rate == 100.0
        assert comparison["Rápido"]["metrics"].avg_execution_time < comparison["Confiável"]["metrics"].avg_execution_time
        assert comparison["comparison"]["success_rate_difference"] > 0  # Confiável - Rápido > 0
    
    def test_calculate_fitness_evaluation_metrics(self):
        """Testa cálculo de métricas de avaliações de fitness."""
        # Criar execuções com dados de iteração
        executions = []
        for i in range(3):
            execution = self.create_successful_execution(100.0 + i * 10, 50.0 + i * 5, 3.0 + i * 0.5)
            # Simular dados de iteração
            execution.iterations_data = [None] * (20 + i * 5)  # 20, 25, 30 iterações
            execution.config = {'population_size': 40}  # População de 40
            executions.append(execution)
        
        metrics = self.analyzer.calculate_fitness_evaluation_metrics(executions)
        
        assert 'total_evaluations' in metrics
        assert 'avg_evaluations_per_execution' in metrics
        assert 'evaluations_per_second' in metrics
        assert 'efficiency_score' in metrics
        
        # Verificar cálculos
        expected_total = (20 * 40) + (25 * 40) + (30 * 40)  # 3000
        assert metrics['total_evaluations'] == expected_total
        assert metrics['avg_evaluations_per_execution'] == expected_total / 3
        assert metrics['evaluations_per_second'] > 0
        assert 0 <= metrics['efficiency_score'] <= 1.0
    
    def test_calculate_fitness_evaluation_metrics_empty(self):
        """Testa métricas de avaliação com lista vazia."""
        metrics = self.analyzer.calculate_fitness_evaluation_metrics([])
        
        assert metrics['total_evaluations'] == 0.0
        assert metrics['avg_evaluations_per_execution'] == 0.0
        assert metrics['evaluations_per_second'] == 0.0
        assert metrics['efficiency_score'] == 0.0
    
    def test_compare_fitness_evaluation_efficiency(self):
        """Testa comparação de eficiência de avaliações de fitness."""
        # Config A: Menos iterações, população menor
        config_a = []
        for i in range(2):
            execution = self.create_successful_execution(100.0, 50.0, 3.0)
            execution.iterations_data = [None] * 15  # 15 iterações
            execution.config = {'population_size': 30}  # População menor
            config_a.append(execution)
        
        # Config B: Mais iterações, população maior
        config_b = []
        for i in range(2):
            execution = self.create_successful_execution(150.0, 45.0, 2.8)
            execution.iterations_data = [None] * 25  # 25 iterações
            execution.config = {'population_size': 50}  # População maior
            config_b.append(execution)
        
        comparison = self.analyzer.compare_fitness_evaluation_efficiency(config_a, config_b)
        
        assert 'config_a_metrics' in comparison
        assert 'config_b_metrics' in comparison
        assert 'comparison' in comparison
        
        comp = comparison['comparison']
        assert 'evaluations_difference' in comp
        assert 'speed_difference' in comp
        assert 'efficiency_ratio' in comp
        assert 'better_config' in comp
        
        # Config B deve ter mais avaliações por execução
        assert comp['evaluations_difference'] > 0  # B - A > 0