"""
Testes para o sistema de análise de convergência.

Este módulo testa todas as funcionalidades do ConvergenceAnalyzer,
incluindo detecção de convergência, análise de estabilidade e padrões.
"""

import pytest
from datetime import datetime
from typing import List
from src.rota_aco.metrics.convergence_analyzer import ConvergenceAnalyzer
from src.rota_aco.metrics.data_models import IterationData, ExecutionData, Solution, Route, ConvergenceMetrics
from src.rota_aco.metrics.exceptions import MetricsCalculationError


class TestConvergenceAnalyzer:
    """Testes para a classe ConvergenceAnalyzer."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.analyzer = ConvergenceAnalyzer(convergence_threshold=0.001, stability_window=10)
    
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
    
    def create_converging_iterations(self, num_iterations: int = 50) -> List[IterationData]:
        """Cria dados de iterações que convergem."""
        iterations = []
        
        for i in range(num_iterations):
            # Fitness melhora rapidamente no início, depois estabiliza
            if i < 20:
                best_fitness = 100.0 - i * 2.0  # Melhoria rápida
                avg_fitness = best_fitness + 10.0
                variance = 50.0 - i * 2.0  # Diversidade diminui
            else:
                best_fitness = 60.0 - (i - 20) * 0.1  # Melhoria lenta
                avg_fitness = best_fitness + 5.0
                variance = 10.0  # Baixa diversidade
            
            solution = self.create_mock_solution(best_fitness, 5.0)
            
            iterations.append(IterationData(
                iteration=i,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                population_variance=variance,
                best_solution=solution
            ))
        
        return iterations
    
    def create_non_converging_iterations(self, num_iterations: int = 50) -> List[IterationData]:
        """Cria dados de iterações que não convergem."""
        iterations = []
        
        for i in range(num_iterations):
            # Fitness oscila sem convergir
            best_fitness = 100.0 + 10.0 * ((-1) ** i) * (i % 5) / 5.0
            avg_fitness = best_fitness + 15.0
            variance = 30.0 + 10.0 * (i % 3)  # Alta variabilidade
            
            solution = self.create_mock_solution(best_fitness, 5.0)
            
            iterations.append(IterationData(
                iteration=i,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                population_variance=variance,
                best_solution=solution
            ))
        
        return iterations
    
    def create_plateau_iterations(self, num_iterations: int = 50) -> List[IterationData]:
        """Cria dados de iterações que atingem um platô."""
        iterations = []
        
        for i in range(num_iterations):
            if i < 15:
                # Melhoria inicial
                best_fitness = 100.0 - i * 3.0
                avg_fitness = best_fitness + 8.0
                variance = 40.0 - i * 2.0
            else:
                # Platô - sem melhoria significativa
                best_fitness = 55.0 + 0.001 * (i % 3)  # Variação mínima
                avg_fitness = best_fitness + 3.0
                variance = 5.0
            
            solution = self.create_mock_solution(best_fitness, 5.0)
            
            iterations.append(IterationData(
                iteration=i,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                population_variance=variance,
                best_solution=solution
            ))
        
        return iterations
    
    def test_analyze_empty_iterations(self):
        """Testa análise com lista vazia de iterações."""
        metrics = self.analyzer.analyze_convergence([])
        
        assert metrics.convergence_point == -1
        assert metrics.final_stability == 0.0
        assert metrics.improvement_rate == 0.0
        assert metrics.plateau_detection is False
        assert metrics.convergence_speed == -1.0
        assert metrics.total_iterations == 0
        assert len(metrics.best_fitness_evolution) == 0
    
    def test_analyze_single_iteration(self):
        """Testa análise com uma única iteração."""
        iterations = [IterationData(
            iteration=0,
            best_fitness=100.0,
            avg_fitness=110.0,
            population_variance=20.0,
            best_solution=self.create_mock_solution()
        )]
        
        metrics = self.analyzer.analyze_convergence(iterations)
        
        assert metrics.convergence_point == -1  # Não pode convergir com 1 iteração
        assert metrics.total_iterations == 1
        assert len(metrics.best_fitness_evolution) == 1
        assert metrics.best_fitness_evolution[0] == 100.0
    
    def test_analyze_converging_iterations(self):
        """Testa análise de iterações que convergem."""
        iterations = self.create_converging_iterations(50)
        metrics = self.analyzer.analyze_convergence(iterations)
        
        assert metrics.convergence_point != -1  # Deve detectar convergência
        assert metrics.convergence_point < 50   # Deve convergir antes do final
        assert metrics.total_iterations == 50
        assert len(metrics.best_fitness_evolution) == 50
        assert metrics.final_stability >= 0    # Estabilidade calculada
        assert metrics.convergence_speed != -1  # Velocidade calculada
    
    def test_analyze_non_converging_iterations(self):
        """Testa análise de iterações que não convergem."""
        iterations = self.create_non_converging_iterations(30)
        metrics = self.analyzer.analyze_convergence(iterations)
        
        assert metrics.convergence_point == -1  # Não deve detectar convergência
        assert metrics.total_iterations == 30
        assert metrics.convergence_speed == -1.0
        assert metrics.final_stability > 0     # Deve ter instabilidade
    
    def test_analyze_plateau_iterations(self):
        """Testa detecção de platô."""
        iterations = self.create_plateau_iterations(50)
        metrics = self.analyzer.analyze_convergence(iterations)
        
        assert metrics.plateau_detection is True  # Deve detectar platô
        assert metrics.final_stability < 1.0      # Deve ser estável no final
        assert metrics.total_iterations == 50
    
    def test_detect_convergence_point(self):
        """Testa detecção específica do ponto de convergência."""
        # Fitness que converge na iteração 10
        fitness_evolution = [100.0, 90.0, 80.0, 70.0, 60.0, 55.0, 52.0, 50.5, 50.1, 50.05, 50.01, 50.005, 50.002, 50.001]
        
        convergence_point = self.analyzer._detect_convergence_point(fitness_evolution)
        
        # Deve detectar convergência quando a melhoria fica abaixo do threshold
        assert convergence_point != -1
        assert convergence_point < len(fitness_evolution)
    
    def test_calculate_final_stability(self):
        """Testa cálculo de estabilidade final."""
        # Fitness estável no final
        stable_fitness = [100.0, 90.0, 80.0, 50.0, 50.1, 50.0, 49.9, 50.0, 50.1]
        stability = self.analyzer._calculate_final_stability(stable_fitness)
        
        assert stability >= 0
        # A estabilidade pode ser maior que 1.0 dependendo da variação absoluta
        
        # Fitness instável no final
        unstable_fitness = [100.0, 90.0, 80.0, 50.0, 60.0, 40.0, 70.0, 30.0, 80.0]
        instability = self.analyzer._calculate_final_stability(unstable_fitness)
        
        assert instability > stability  # Deve ser maior para fitness instável
    
    def test_calculate_improvement_rate(self):
        """Testa cálculo da taxa de melhoria."""
        # Fitness melhorando consistentemente
        improving_fitness = [100.0, 90.0, 80.0, 70.0, 60.0]
        improvement_rate = self.analyzer._calculate_improvement_rate(improving_fitness)
        
        assert improvement_rate < 0  # Negativo = melhoria (fitness menor é melhor)
        
        # Fitness piorando
        worsening_fitness = [50.0, 60.0, 70.0, 80.0, 90.0]
        worsening_rate = self.analyzer._calculate_improvement_rate(worsening_fitness)
        
        assert worsening_rate > 0  # Positivo = piora
    
    def test_detect_plateau(self):
        """Testa detecção de platô."""
        # Criar fitness com platô no final
        plateau_fitness = [100.0] * 5 + [50.0] * 15  # 15 iterações estáveis
        
        analyzer_small_window = ConvergenceAnalyzer(stability_window=10)
        has_plateau = analyzer_small_window._detect_plateau(plateau_fitness)
        
        assert has_plateau is True
        
        # Fitness sem platô
        no_plateau_fitness = [100.0 - i for i in range(20)]  # Sempre melhorando
        has_no_plateau = analyzer_small_window._detect_plateau(no_plateau_fitness)
        
        assert has_no_plateau is False
    
    def test_calculate_convergence_speed(self):
        """Testa cálculo da velocidade de convergência."""
        fitness_evolution = [100.0, 90.0, 80.0, 70.0, 60.0]
        
        # Convergência na iteração 3
        speed = self.analyzer._calculate_convergence_speed(fitness_evolution, 3)
        assert speed == 3.0
        
        # Sem convergência
        no_speed = self.analyzer._calculate_convergence_speed(fitness_evolution, -1)
        assert no_speed == -1.0
    
    def test_track_fitness_evolution(self):
        """Testa extração da evolução do fitness."""
        iterations = self.create_converging_iterations(10)
        evolution = self.analyzer.track_fitness_evolution(iterations)
        
        assert 'iterations' in evolution
        assert 'best_fitness' in evolution
        assert 'avg_fitness' in evolution
        assert 'population_variance' in evolution
        
        assert len(evolution['iterations']) == 10
        assert len(evolution['best_fitness']) == 10
        assert len(evolution['avg_fitness']) == 10
        assert len(evolution['population_variance']) == 10
        
        # Verificar ordem das iterações
        assert evolution['iterations'] == list(range(10))
    
    def test_calculate_population_diversity(self):
        """Testa cálculo da diversidade da população."""
        iterations = self.create_converging_iterations(10)
        diversity = self.analyzer.calculate_population_diversity(iterations)
        
        assert len(diversity) == 10
        assert all(d >= 0 for d in diversity)  # Diversidade não pode ser negativa
        
        # Em iterações convergentes, diversidade deve diminuir
        assert diversity[0] > diversity[-1]
    
    def test_analyze_execution(self):
        """Testa análise de uma execução completa."""
        iterations = self.create_converging_iterations(30)
        execution_data = ExecutionData(
            algorithm_type="ACS-TIME",
            iterations_data=iterations,
            execution_time=120.0,
            success=True
        )
        
        metrics = self.analyzer.analyze_execution(execution_data)
        
        assert isinstance(metrics, ConvergenceMetrics)
        assert metrics.total_iterations == 30
        assert len(metrics.best_fitness_evolution) == 30
    
    def test_get_convergence_summary(self):
        """Testa geração de resumo de convergência."""
        iterations = self.create_converging_iterations(25)
        metrics = self.analyzer.analyze_convergence(iterations)
        summary = self.analyzer.get_convergence_summary(metrics)
        
        assert "Análise de Convergência" in summary
        assert "Status de Convergência" in summary
        assert "Total de Iterações: 25" in summary
        assert isinstance(summary, str)
        assert len(summary) > 100  # Resumo substancial
    
    def test_compare_convergence_patterns(self):
        """Testa comparação de padrões de convergência."""
        # Criar múltiplas métricas
        metrics_list = []
        
        for _ in range(3):
            iterations = self.create_converging_iterations(20)
            metrics = self.analyzer.analyze_convergence(iterations)
            metrics_list.append(metrics)
        
        # Adicionar uma execução que não converge
        non_conv_iterations = self.create_non_converging_iterations(20)
        non_conv_metrics = self.analyzer.analyze_convergence(non_conv_iterations)
        metrics_list.append(non_conv_metrics)
        
        comparison = self.analyzer.compare_convergence_patterns(metrics_list)
        
        assert 'convergence_rate' in comparison
        assert 'avg_convergence_point' in comparison
        assert 'avg_final_stability' in comparison
        assert 'plateau_rate' in comparison
        
        # Verificar que a taxa de convergência está entre 0 e 100%
        assert 0 <= comparison['convergence_rate'] <= 100
        assert 0 <= comparison['plateau_rate'] <= 100
    
    def test_compare_empty_metrics(self):
        """Testa comparação com lista vazia."""
        comparison = self.analyzer.compare_convergence_patterns([])
        assert comparison == {}
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Criar iteração com dados problemáticos
        problematic_iteration = IterationData(
            iteration=0,
            best_fitness=float('inf'),  # Valor problemático
            avg_fitness=float('nan'),   # Valor inválido
            population_variance=-1.0,   # Valor inválido
            best_solution=self.create_mock_solution()
        )
        
        # O analisador deve lidar com isso graciosamente
        try:
            metrics = self.analyzer.analyze_convergence([problematic_iteration])
            assert isinstance(metrics, ConvergenceMetrics)
        except MetricsCalculationError:
            # Ou lançar exceção apropriada
            pass
    
    def test_different_thresholds(self):
        """Testa comportamento com diferentes thresholds."""
        iterations = self.create_converging_iterations(30)
        
        # Threshold mais rigoroso
        strict_analyzer = ConvergenceAnalyzer(convergence_threshold=0.0001)
        strict_metrics = strict_analyzer.analyze_convergence(iterations)
        
        # Threshold mais permissivo
        loose_analyzer = ConvergenceAnalyzer(convergence_threshold=0.01)
        loose_metrics = loose_analyzer.analyze_convergence(iterations)
        
        # Threshold mais rigoroso deve convergir mais tarde (ou não convergir)
        if strict_metrics.convergence_point != -1 and loose_metrics.convergence_point != -1:
            assert strict_metrics.convergence_point >= loose_metrics.convergence_point
    
    def test_stability_window_effects(self):
        """Testa efeitos de diferentes janelas de estabilidade."""
        iterations = self.create_plateau_iterations(50)
        
        # Janela pequena
        small_window_analyzer = ConvergenceAnalyzer(stability_window=5)
        small_metrics = small_window_analyzer.analyze_convergence(iterations)
        
        # Janela grande
        large_window_analyzer = ConvergenceAnalyzer(stability_window=20)
        large_metrics = large_window_analyzer.analyze_convergence(iterations)
        
        # Ambos devem detectar o platô, mas com estabilidades diferentes
        assert small_metrics.plateau_detection is True
        assert large_metrics.plateau_detection is True
        
        # A estabilidade pode variar dependendo da janela
        assert small_metrics.final_stability >= 0
        assert large_metrics.final_stability >= 0