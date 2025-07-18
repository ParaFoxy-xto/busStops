"""
Sistema de análise estatística comparativa para algoritmos ACO.

Este módulo implementa o StatisticalAnalyzer que calcula estatísticas
agregadas e comparativas entre múltiplas execuções de algoritmos ACO,
permitindo análise side-by-side e cálculo de métricas de eficiência relativa.
"""

from typing import List, Dict, Optional, Tuple, Any
import statistics
import math
from .data_models import ExecutionData, ComparativeMetrics, RouteQualityMetrics, ConvergenceMetrics
from .exceptions import MetricsCalculationError


class StatisticalAnalyzer:
    """
    Analisador estatístico que calcula métricas comparativas entre
    múltiplas execuções de algoritmos ACO para validação científica.
    """
    
    def __init__(self):
        """Inicializa o analisador estatístico."""
        pass
    
    def analyze_multiple_executions(self, executions: List[ExecutionData]) -> ComparativeMetrics:
        """
        Analisa múltiplas execuções e calcula métricas comparativas.
        
        Args:
            executions: Lista de dados de execuções
            
        Returns:
            ComparativeMetrics com análise estatística completa
            
        Raises:
            MetricsCalculationError: Se houver erro na análise
        """
        try:
            if not executions:
                return self._empty_comparative_metrics()
            
            # Filtrar execuções bem-sucedidas
            successful_executions = [ex for ex in executions if ex.success]
            
            # Calcular estatísticas básicas
            total_executions = len(executions)
            successful_count = len(successful_executions)
            success_rate = (successful_count / total_executions) * 100.0 if total_executions > 0 else 0.0
            
            # Calcular estatísticas de tempo de execução
            execution_times = [ex.execution_time for ex in executions if ex.execution_time > 0]
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            std_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            
            # Calcular estatísticas de fitness (apenas execuções bem-sucedidas)
            if successful_executions:
                best_fitness_values = []
                convergence_iterations = []
                fitness_evaluations = []
                
                for ex in successful_executions:
                    if ex.final_solution:
                        # Usar o menor fitness (melhor) entre time e vehicle
                        best_fitness = min(ex.final_solution.fitness_time, ex.final_solution.fitness_vehicle)
                        best_fitness_values.append(best_fitness)
                    
                    # Calcular iterações até convergência (se disponível)
                    if ex.iterations_data:
                        convergence_iterations.append(len(ex.iterations_data))
                        # Estimar número de avaliações de fitness (iterações * população estimada)
                        # Assumindo população de 50 por padrão (pode ser configurável)
                        estimated_population_size = ex.config.get('population_size', 50) if ex.config else 50
                        fitness_evaluations.append(len(ex.iterations_data) * estimated_population_size)
                
                avg_best_fitness = statistics.mean(best_fitness_values) if best_fitness_values else 0.0
                std_best_fitness = statistics.stdev(best_fitness_values) if len(best_fitness_values) > 1 else 0.0
                median_best_fitness = statistics.median(best_fitness_values) if best_fitness_values else 0.0
                avg_convergence_iterations = statistics.mean(convergence_iterations) if convergence_iterations else 0.0
                avg_fitness_evaluations = statistics.mean(fitness_evaluations) if fitness_evaluations else 0.0
            else:
                avg_best_fitness = 0.0
                std_best_fitness = 0.0
                median_best_fitness = 0.0
                avg_convergence_iterations = 0.0
                avg_fitness_evaluations = 0.0
            
            return ComparativeMetrics(
                total_executions=total_executions,
                successful_executions=successful_count,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                std_execution_time=std_execution_time,
                avg_best_fitness=avg_best_fitness,
                std_best_fitness=std_best_fitness,
                median_best_fitness=median_best_fitness,
                avg_convergence_iterations=avg_convergence_iterations,
                relative_efficiency=1.0  # Base efficiency
            )
            
        except Exception as e:
            raise MetricsCalculationError(f"Erro ao analisar múltiplas execuções: {str(e)}")
    
    def compare_algorithm_configurations(self, 
                                       config_a_executions: List[ExecutionData],
                                       config_b_executions: List[ExecutionData],
                                       config_a_name: str = "Config A",
                                       config_b_name: str = "Config B") -> Dict[str, Any]:
        """
        Compara duas configurações de algoritmo side-by-side.
        
        Args:
            config_a_executions: Execuções da configuração A
            config_b_executions: Execuções da configuração B
            config_a_name: Nome da configuração A
            config_b_name: Nome da configuração B
            
        Returns:
            Dict com comparação detalhada entre as configurações
        """
        try:
            metrics_a = self.analyze_multiple_executions(config_a_executions)
            metrics_b = self.analyze_multiple_executions(config_b_executions)
            
            # Calcular eficiência relativa (B em relação a A)
            if metrics_a.avg_execution_time > 0 and metrics_a.avg_best_fitness > 0:
                time_efficiency = metrics_a.avg_execution_time / metrics_b.avg_execution_time if metrics_b.avg_execution_time > 0 else 1.0
                fitness_efficiency = metrics_a.avg_best_fitness / metrics_b.avg_best_fitness if metrics_b.avg_best_fitness > 0 else 1.0
                relative_efficiency = (time_efficiency + fitness_efficiency) / 2.0
            else:
                relative_efficiency = 1.0
            
            metrics_b.relative_efficiency = relative_efficiency
            
            comparison = {
                config_a_name: {
                    'metrics': metrics_a,
                    'summary': self._create_metrics_summary(metrics_a)
                },
                config_b_name: {
                    'metrics': metrics_b,
                    'summary': self._create_metrics_summary(metrics_b)
                },
                'comparison': {
                    'success_rate_difference': metrics_b.success_rate - metrics_a.success_rate,
                    'avg_time_difference': metrics_b.avg_execution_time - metrics_a.avg_execution_time,
                    'avg_fitness_difference': metrics_b.avg_best_fitness - metrics_a.avg_best_fitness,
                    'relative_efficiency': relative_efficiency,
                    'better_config': self._determine_better_config(metrics_a, metrics_b, config_a_name, config_b_name)
                }
            }
            
            return comparison
            
        except Exception as e:
            raise MetricsCalculationError(f"Erro ao comparar configurações: {str(e)}")
    
    def calculate_aggregated_statistics(self, executions: List[ExecutionData]) -> Dict[str, Dict[str, float]]:
        """
        Calcula estatísticas agregadas detalhadas para múltiplas execuções.
        
        Args:
            executions: Lista de execuções para analisar
            
        Returns:
            Dict com estatísticas organizadas por categoria
        """
        if not executions:
            return {}
        
        successful_executions = [ex for ex in executions if ex.success]
        
        # Estatísticas de execução
        execution_times = [ex.execution_time for ex in executions if ex.execution_time > 0]
        execution_stats = self._calculate_descriptive_stats(execution_times, "Tempo de Execução")
        
        # Estatísticas de fitness
        fitness_values = []
        for ex in successful_executions:
            if ex.final_solution:
                fitness_values.append(min(ex.final_solution.fitness_time, ex.final_solution.fitness_vehicle))
        
        fitness_stats = self._calculate_descriptive_stats(fitness_values, "Fitness")
        
        # Estatísticas de convergência
        convergence_iterations = [len(ex.iterations_data) for ex in successful_executions if ex.iterations_data]
        convergence_stats = self._calculate_descriptive_stats(convergence_iterations, "Iterações de Convergência")
        
        # Estatísticas de sucesso
        success_stats = {
            'total_executions': len(executions),
            'successful_executions': len(successful_executions),
            'success_rate': (len(successful_executions) / len(executions)) * 100.0,
            'failure_rate': ((len(executions) - len(successful_executions)) / len(executions)) * 100.0
        }
        
        return {
            'execution_time': execution_stats,
            'fitness': fitness_stats,
            'convergence': convergence_stats,
            'success': success_stats
        }
    
    def calculate_success_rate(self, executions: List[ExecutionData]) -> Dict[str, float]:
        """
        Calcula taxa de sucesso detalhada das execuções.
        
        Args:
            executions: Lista de execuções
            
        Returns:
            Dict com métricas de taxa de sucesso
        """
        if not executions:
            return {'success_rate': 0.0, 'total_executions': 0, 'successful_executions': 0}
        
        successful_count = sum(1 for ex in executions if ex.success)
        total_count = len(executions)
        
        # Analisar tipos de falha
        error_types = {}
        for ex in executions:
            if not ex.success and ex.error_message:
                error_type = ex.error_message.split(':')[0] if ':' in ex.error_message else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'success_rate': (successful_count / total_count) * 100.0,
            'failure_rate': ((total_count - successful_count) / total_count) * 100.0,
            'total_executions': total_count,
            'successful_executions': successful_count,
            'failed_executions': total_count - successful_count,
            'error_types': error_types
        }
    
    def calculate_relative_efficiency(self, 
                                    baseline_executions: List[ExecutionData],
                                    comparison_executions: List[ExecutionData]) -> Dict[str, float]:
        """
        Calcula eficiência relativa entre duas configurações.
        
        Args:
            baseline_executions: Execuções da configuração baseline
            comparison_executions: Execuções da configuração para comparar
            
        Returns:
            Dict com métricas de eficiência relativa
        """
        baseline_metrics = self.analyze_multiple_executions(baseline_executions)
        comparison_metrics = self.analyze_multiple_executions(comparison_executions)
        
        efficiency_metrics = {}
        
        # Eficiência de tempo
        if baseline_metrics.avg_execution_time > 0 and comparison_metrics.avg_execution_time > 0:
            efficiency_metrics['time_efficiency'] = baseline_metrics.avg_execution_time / comparison_metrics.avg_execution_time
        else:
            efficiency_metrics['time_efficiency'] = 1.0
        
        # Eficiência de qualidade (fitness menor é melhor)
        if baseline_metrics.avg_best_fitness > 0 and comparison_metrics.avg_best_fitness > 0:
            efficiency_metrics['quality_efficiency'] = baseline_metrics.avg_best_fitness / comparison_metrics.avg_best_fitness
        else:
            efficiency_metrics['quality_efficiency'] = 1.0
        
        # Eficiência de sucesso
        if baseline_metrics.success_rate > 0:
            efficiency_metrics['success_efficiency'] = comparison_metrics.success_rate / baseline_metrics.success_rate
        else:
            efficiency_metrics['success_efficiency'] = 1.0 if comparison_metrics.success_rate > 0 else 0.0
        
        # Eficiência geral (média ponderada)
        efficiency_metrics['overall_efficiency'] = (
            efficiency_metrics['time_efficiency'] * 0.3 +
            efficiency_metrics['quality_efficiency'] * 0.4 +
            efficiency_metrics['success_efficiency'] * 0.3
        )
        
        return efficiency_metrics
    
    def calculate_fitness_evaluation_metrics(self, executions: List[ExecutionData]) -> Dict[str, float]:
        """
        Calcula métricas relacionadas ao número de avaliações de fitness.
        
        Args:
            executions: Lista de execuções para analisar
            
        Returns:
            Dict com métricas de avaliações de fitness
        """
        if not executions:
            return {
                'total_evaluations': 0.0,
                'avg_evaluations_per_execution': 0.0,
                'evaluations_per_second': 0.0,
                'efficiency_score': 0.0
            }
        
        successful_executions = [ex for ex in executions if ex.success]
        
        if not successful_executions:
            return {
                'total_evaluations': 0.0,
                'avg_evaluations_per_execution': 0.0,
                'evaluations_per_second': 0.0,
                'efficiency_score': 0.0
            }
        
        total_evaluations = 0.0
        evaluation_rates = []
        
        for ex in successful_executions:
            if ex.iterations_data and ex.execution_time > 0:
                # Estimar avaliações baseado em iterações e tamanho da população
                estimated_population_size = ex.config.get('population_size', 50) if ex.config else 50
                evaluations = len(ex.iterations_data) * estimated_population_size
                total_evaluations += evaluations
                
                # Calcular taxa de avaliações por segundo
                evaluation_rate = evaluations / ex.execution_time
                evaluation_rates.append(evaluation_rate)
        
        avg_evaluations_per_execution = total_evaluations / len(successful_executions) if successful_executions else 0.0
        avg_evaluations_per_second = statistics.mean(evaluation_rates) if evaluation_rates else 0.0
        
        # Calcular score de eficiência (avaliações por segundo normalizado)
        # Valores mais altos indicam melhor eficiência computacional
        efficiency_score = avg_evaluations_per_second / 1000.0  # Normalizar para escala 0-1
        
        return {
            'total_evaluations': total_evaluations,
            'avg_evaluations_per_execution': avg_evaluations_per_execution,
            'evaluations_per_second': avg_evaluations_per_second,
            'efficiency_score': min(efficiency_score, 1.0)  # Cap at 1.0
        }
    
    def compare_fitness_evaluation_efficiency(self, 
                                            config_a_executions: List[ExecutionData],
                                            config_b_executions: List[ExecutionData]) -> Dict[str, Any]:
        """
        Compara a eficiência de avaliações de fitness entre duas configurações.
        
        Args:
            config_a_executions: Execuções da configuração A
            config_b_executions: Execuções da configuração B
            
        Returns:
            Dict com comparação de eficiência de avaliações
        """
        metrics_a = self.calculate_fitness_evaluation_metrics(config_a_executions)
        metrics_b = self.calculate_fitness_evaluation_metrics(config_b_executions)
        
        # Calcular diferenças relativas
        comparison = {
            'config_a_metrics': metrics_a,
            'config_b_metrics': metrics_b,
            'comparison': {
                'evaluations_difference': metrics_b['avg_evaluations_per_execution'] - metrics_a['avg_evaluations_per_execution'],
                'speed_difference': metrics_b['evaluations_per_second'] - metrics_a['evaluations_per_second'],
                'efficiency_ratio': metrics_b['efficiency_score'] / metrics_a['efficiency_score'] if metrics_a['efficiency_score'] > 0 else 1.0,
                'better_config': 'B' if metrics_b['efficiency_score'] > metrics_a['efficiency_score'] else 'A' if metrics_a['efficiency_score'] > metrics_b['efficiency_score'] else 'Empate'
            }
        }
        
        return comparison
    
    def _calculate_descriptive_stats(self, values: List[float], name: str) -> Dict[str, float]:
        """Calcula estatísticas descritivas para uma lista de valores."""
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0,
                'variance': 0.0
            }
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'variance': statistics.variance(values) if len(values) > 1 else 0.0
        }
    
    def _create_metrics_summary(self, metrics: ComparativeMetrics) -> str:
        """Cria um resumo textual das métricas."""
        return f"""
Execuções: {metrics.successful_executions}/{metrics.total_executions} ({metrics.success_rate:.1f}% sucesso)
Tempo médio: {metrics.avg_execution_time:.2f}s (±{metrics.std_execution_time:.2f}s)
Fitness médio: {metrics.avg_best_fitness:.2f} (±{metrics.std_best_fitness:.2f})
Iterações médias: {metrics.avg_convergence_iterations:.1f}
""".strip()
    
    def _determine_better_config(self, metrics_a: ComparativeMetrics, metrics_b: ComparativeMetrics,
                                name_a: str, name_b: str) -> str:
        """Determina qual configuração é melhor baseada nas métricas."""
        score_a = 0
        score_b = 0
        
        # Comparar taxa de sucesso
        if metrics_a.success_rate > metrics_b.success_rate:
            score_a += 1
        elif metrics_b.success_rate > metrics_a.success_rate:
            score_b += 1
        
        # Comparar tempo de execução (menor é melhor)
        if metrics_a.avg_execution_time < metrics_b.avg_execution_time:
            score_a += 1
        elif metrics_b.avg_execution_time < metrics_a.avg_execution_time:
            score_b += 1
        
        # Comparar fitness (menor é melhor)
        if metrics_a.avg_best_fitness < metrics_b.avg_best_fitness:
            score_a += 1
        elif metrics_b.avg_best_fitness < metrics_a.avg_best_fitness:
            score_b += 1
        
        if score_a > score_b:
            return name_a
        elif score_b > score_a:
            return name_b
        else:
            return "Empate"
    
    def _empty_comparative_metrics(self) -> ComparativeMetrics:
        """Retorna métricas comparativas vazias."""
        return ComparativeMetrics(
            total_executions=0,
            successful_executions=0,
            success_rate=0.0,
            avg_execution_time=0.0,
            std_execution_time=0.0,
            avg_best_fitness=0.0,
            std_best_fitness=0.0,
            median_best_fitness=0.0,
            avg_convergence_iterations=0.0,
            relative_efficiency=1.0
        )
    
    def generate_comparison_report(self, comparison_data: Dict[str, Any]) -> str:
        """
        Gera um relatório textual da comparação entre configurações.
        
        Args:
            comparison_data: Dados da comparação gerados por compare_algorithm_configurations
            
        Returns:
            String com relatório formatado
        """
        if not comparison_data:
            return "Nenhum dado de comparação disponível."
        
        config_names = [k for k in comparison_data.keys() if k != 'comparison']
        if len(config_names) < 2:
            return "Dados insuficientes para comparação."
        
        config_a_name = config_names[0]
        config_b_name = config_names[1]
        comparison = comparison_data['comparison']
        
        report = f"""
Relatório de Comparação de Configurações
========================================

{config_a_name}:
{comparison_data[config_a_name]['summary']}

{config_b_name}:
{comparison_data[config_b_name]['summary']}

Análise Comparativa:
-------------------
Diferença na Taxa de Sucesso: {comparison['success_rate_difference']:+.1f}%
Diferença no Tempo Médio: {comparison['avg_time_difference']:+.2f}s
Diferença no Fitness Médio: {comparison['avg_fitness_difference']:+.2f}
Eficiência Relativa: {comparison['relative_efficiency']:.2f}x

Configuração Superior: {comparison['better_config']}
"""
        return report.strip()