"""
Sistema de análise de convergência para algoritmos ACO.

Este módulo implementa o ConvergenceAnalyzer que analisa a evolução
da convergência dos algoritmos ACO, detectando pontos de convergência,
estabilidade e padrões de melhoria.
"""

from typing import List, Dict, Optional, Tuple
import statistics
import numpy as np
from .data_models import IterationData, ExecutionData, ConvergenceMetrics
from .exceptions import MetricsCalculationError


class ConvergenceAnalyzer:
    """
    Analisador de convergência que calcula métricas detalhadas sobre
    a evolução dos algoritmos ACO ao longo das iterações.
    """
    
    def __init__(self, convergence_threshold: float = 0.001, stability_window: int = 50):
        """
        Inicializa o analisador de convergência.
        
        Args:
            convergence_threshold: Threshold para detectar convergência (padrão: 0.001)
            stability_window: Janela de iterações para calcular estabilidade (padrão: 50)
        """
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window
    
    def analyze_convergence(self, iterations_data: List[IterationData]) -> ConvergenceMetrics:
        """
        Analisa a convergência de uma execução ACO.
        
        Args:
            iterations_data: Lista de dados de iterações
            
        Returns:
            ConvergenceMetrics com análise completa
            
        Raises:
            MetricsCalculationError: Se houver erro na análise
        """
        try:
            if not iterations_data:
                return self._empty_convergence_metrics()
            
            # Extrair séries temporais
            best_fitness_evolution = [iter_data.best_fitness for iter_data in iterations_data]
            avg_fitness_evolution = [iter_data.avg_fitness for iter_data in iterations_data]
            variance_evolution = [iter_data.population_variance for iter_data in iterations_data]
            
            # Calcular métricas de convergência
            convergence_point = self._detect_convergence_point(best_fitness_evolution)
            final_stability = self._calculate_final_stability(best_fitness_evolution)
            improvement_rate = self._calculate_improvement_rate(best_fitness_evolution)
            plateau_detection = self._detect_plateau(best_fitness_evolution)
            convergence_speed = self._calculate_convergence_speed(best_fitness_evolution, convergence_point)
            
            return ConvergenceMetrics(
                convergence_point=convergence_point,
                final_stability=final_stability,
                improvement_rate=improvement_rate,
                plateau_detection=plateau_detection,
                convergence_speed=convergence_speed,
                total_iterations=len(iterations_data),
                best_fitness_evolution=best_fitness_evolution,
                avg_fitness_evolution=avg_fitness_evolution,
                variance_evolution=variance_evolution
            )
            
        except Exception as e:
            raise MetricsCalculationError(f"Erro ao analisar convergência: {str(e)}")
    
    def analyze_execution(self, execution_data: ExecutionData) -> ConvergenceMetrics:
        """
        Analisa a convergência de uma execução completa.
        
        Args:
            execution_data: Dados da execução
            
        Returns:
            ConvergenceMetrics da execução
        """
        return self.analyze_convergence(execution_data.iterations_data)
    
    def _detect_convergence_point(self, fitness_evolution: List[float]) -> int:
        """
        Detecta o ponto de convergência baseado no threshold de melhoria.
        
        Args:
            fitness_evolution: Evolução do fitness ao longo das iterações
            
        Returns:
            Iteração onde a convergência foi detectada (-1 se não convergiu)
        """
        if len(fitness_evolution) < 5:  # Precisa de pelo menos 5 pontos
            return -1
        
        # Procurar por uma janela de estabilidade
        stability_window = min(5, len(fitness_evolution) // 4)  # Janela adaptativa
        
        for i in range(stability_window, len(fitness_evolution)):
            # Verificar estabilidade na janela atual
            window_values = fitness_evolution[i-stability_window:i+1]
            
            # Calcular variação relativa na janela
            min_val = min(window_values)
            max_val = max(window_values)
            
            if min_val != 0:
                relative_variation = (max_val - min_val) / abs(min_val)
            else:
                relative_variation = max_val - min_val
            
            # Se a variação está abaixo do threshold, consideramos convergência
            if relative_variation < self.convergence_threshold * 10:  # Threshold mais permissivo
                return i - stability_window + 1
        
        return -1  # Não convergiu
    
    def _calculate_final_stability(self, fitness_evolution: List[float]) -> float:
        """
        Calcula a estabilidade final baseada no desvio padrão das últimas iterações.
        
        Args:
            fitness_evolution: Evolução do fitness
            
        Returns:
            Desvio padrão das últimas N iterações (menor = mais estável)
        """
        if len(fitness_evolution) < 2:
            return 0.0
        
        # Usar janela de estabilidade ou todas as iterações se houver menos
        window_size = min(self.stability_window, len(fitness_evolution))
        final_values = fitness_evolution[-window_size:]
        
        if len(final_values) < 2:
            return 0.0
        
        return statistics.stdev(final_values)
    
    def _calculate_improvement_rate(self, fitness_evolution: List[float]) -> float:
        """
        Calcula a taxa de melhoria média por iteração.
        
        Args:
            fitness_evolution: Evolução do fitness
            
        Returns:
            Taxa de melhoria média (positiva = melhoria, negativa = piora)
        """
        if len(fitness_evolution) < 2:
            return 0.0
        
        total_improvement = 0.0
        valid_improvements = 0
        
        for i in range(1, len(fitness_evolution)):
            if fitness_evolution[i-1] != 0:
                improvement = (fitness_evolution[i] - fitness_evolution[i-1]) / abs(fitness_evolution[i-1])
                total_improvement += improvement
                valid_improvements += 1
        
        return total_improvement / valid_improvements if valid_improvements > 0 else 0.0
    
    def _detect_plateau(self, fitness_evolution: List[float]) -> bool:
        """
        Detecta se o algoritmo atingiu um platô (estagnação).
        
        Args:
            fitness_evolution: Evolução do fitness
            
        Returns:
            True se detectou platô, False caso contrário
        """
        if len(fitness_evolution) < self.stability_window:
            return False
        
        # Verificar as últimas iterações
        recent_values = fitness_evolution[-self.stability_window:]
        
        # Calcular variação máxima nas últimas iterações
        min_val = min(recent_values)
        max_val = max(recent_values)
        
        if min_val == 0:
            variation = max_val - min_val
        else:
            variation = (max_val - min_val) / abs(min_val)
        
        # Se a variação é muito pequena, consideramos platô
        return variation < self.convergence_threshold * 10
    
    def _calculate_convergence_speed(self, fitness_evolution: List[float], convergence_point: int) -> float:
        """
        Calcula a velocidade de convergência.
        
        Args:
            fitness_evolution: Evolução do fitness
            convergence_point: Ponto onde convergiu (-1 se não convergiu)
            
        Returns:
            Velocidade de convergência (iterações até convergir, -1 se não convergiu)
        """
        if convergence_point == -1 or len(fitness_evolution) < 2:
            return -1.0
        
        return float(convergence_point)
    
    def _empty_convergence_metrics(self) -> ConvergenceMetrics:
        """Retorna métricas vazias para casos sem dados."""
        return ConvergenceMetrics(
            convergence_point=-1,
            final_stability=0.0,
            improvement_rate=0.0,
            plateau_detection=False,
            convergence_speed=-1.0,
            total_iterations=0,
            best_fitness_evolution=[],
            avg_fitness_evolution=[],
            variance_evolution=[]
        )
    
    def track_fitness_evolution(self, iterations_data: List[IterationData]) -> Dict[str, List[float]]:
        """
        Extrai e organiza a evolução do fitness para análise.
        
        Args:
            iterations_data: Dados das iterações
            
        Returns:
            Dict com séries temporais organizadas
        """
        return {
            'iterations': [iter_data.iteration for iter_data in iterations_data],
            'best_fitness': [iter_data.best_fitness for iter_data in iterations_data],
            'avg_fitness': [iter_data.avg_fitness for iter_data in iterations_data],
            'population_variance': [iter_data.population_variance for iter_data in iterations_data]
        }
    
    def calculate_population_diversity(self, iterations_data: List[IterationData]) -> List[float]:
        """
        Calcula a diversidade da população ao longo das iterações.
        
        Args:
            iterations_data: Dados das iterações
            
        Returns:
            Lista com índices de diversidade por iteração
        """
        diversity_indices = []
        
        for iter_data in iterations_data:
            # Usar a variância como proxy para diversidade
            # Valores maiores = maior diversidade
            diversity_indices.append(iter_data.population_variance)
        
        return diversity_indices
    
    def get_convergence_summary(self, metrics: ConvergenceMetrics) -> str:
        """
        Gera um resumo textual da análise de convergência.
        
        Args:
            metrics: Métricas de convergência
            
        Returns:
            String com resumo formatado
        """
        convergence_status = "Convergiu" if metrics.convergence_point != -1 else "Não convergiu"
        plateau_status = "Sim" if metrics.plateau_detection else "Não"
        
        summary = f"""
Análise de Convergência:
=======================

Status de Convergência: {convergence_status}
Ponto de Convergência: {metrics.convergence_point if metrics.convergence_point != -1 else 'N/A'}
Total de Iterações: {metrics.total_iterations}

Métricas de Estabilidade:
- Estabilidade Final: {metrics.final_stability:.6f}
- Taxa de Melhoria: {metrics.improvement_rate:.6f}
- Velocidade de Convergência: {metrics.convergence_speed if metrics.convergence_speed != -1 else 'N/A'}

Detecção de Padrões:
- Platô Detectado: {plateau_status}

Evolução do Fitness:
- Melhor Fitness Inicial: {metrics.best_fitness_evolution[0] if metrics.best_fitness_evolution else 'N/A'}
- Melhor Fitness Final: {metrics.best_fitness_evolution[-1] if metrics.best_fitness_evolution else 'N/A'}
"""
        return summary.strip()
    
    def compare_convergence_patterns(self, metrics_list: List[ConvergenceMetrics]) -> Dict[str, float]:
        """
        Compara padrões de convergência entre múltiplas execuções.
        
        Args:
            metrics_list: Lista de métricas de convergência
            
        Returns:
            Dict com estatísticas comparativas
        """
        if not metrics_list:
            return {}
        
        convergence_points = [m.convergence_point for m in metrics_list if m.convergence_point != -1]
        convergence_speeds = [m.convergence_speed for m in metrics_list if m.convergence_speed != -1]
        final_stabilities = [m.final_stability for m in metrics_list]
        improvement_rates = [m.improvement_rate for m in metrics_list]
        
        comparison = {
            'convergence_rate': len(convergence_points) / len(metrics_list) * 100,
            'avg_convergence_point': statistics.mean(convergence_points) if convergence_points else -1,
            'avg_convergence_speed': statistics.mean(convergence_speeds) if convergence_speeds else -1,
            'avg_final_stability': statistics.mean(final_stabilities) if final_stabilities else 0,
            'avg_improvement_rate': statistics.mean(improvement_rates) if improvement_rates else 0,
            'plateau_rate': sum(1 for m in metrics_list if m.plateau_detection) / len(metrics_list) * 100
        }
        
        return comparison