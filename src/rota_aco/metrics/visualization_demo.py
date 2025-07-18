#!/usr/bin/env python3
"""
Demonstração das capacidades do motor de visualização do sistema de métricas.

Este script mostra como usar o VisualizationEngine para gerar diferentes tipos
de visualizações para análise de resultados ACO.
"""

import tempfile
import shutil
from pathlib import Path

from .visualization_engine import VisualizationEngine
from .data_models import (
    ConvergenceMetrics, ComparativeMetrics, RouteQualityMetrics,
    ExecutionData, IterationData, Solution, Route
)


def create_sample_data():
    """Cria dados de exemplo para demonstração."""
    
    # Dados de convergência
    convergence_data = ConvergenceMetrics(
        convergence_point=45,
        final_stability=0.001,
        improvement_rate=0.04,
        plateau_detection=True,
        convergence_speed=0.03,
        total_iterations=100,
        best_fitness_evolution=[120.0 - i * 0.6 for i in range(100)],
        avg_fitness_evolution=[130.0 - i * 0.4 for i in range(100)],
        variance_evolution=[15.0 - i * 0.12 for i in range(100)]
    )
    
    # Dados comparativos
    comparative_data = ComparativeMetrics(
        total_executions=10,
        successful_executions=9,
        success_rate=90.0,
        avg_execution_time=52.3,
        std_execution_time=6.8,
        avg_best_fitness=78.5,
        std_best_fitness=4.2,
        median_best_fitness=79.1,
        avg_convergence_iterations=48.7,
        relative_efficiency=1.15
    )
    
    # Dados de qualidade
    quality_data = RouteQualityMetrics(
        valid_routes_percentage=92.5,
        demand_coverage_percentage=87.3,
        vehicle_utilization_efficiency=0.78,
        capacity_violations=3,
        opposite_stops_violations=1,
        average_route_length=14.8,
        route_length_variance=3.2,
        load_balancing_index=0.83
    )
    
    # Dados de execuções múltiplas
    executions_data = []
    for i in range(5):
        iterations = []
        for j in range(80):
            solution = Solution([], 0, 0.0, 0, 110.0 - j - i*2, 0.0, True)
            iter_data = IterationData(j, 110.0 - j - i*2, 115.0 - j - i*1.5, 8.0, solution)
            iterations.append(iter_data)
        
        final_solution = Solution([], 0, 0.0, 0, 30.0 + i, 0.0, True)
        execution = ExecutionData(
            algorithm_type=f"ACS-DEMO-{i+1}",
            iterations_data=iterations,
            final_solution=final_solution
        )
        executions_data.append(execution)
    
    # Dados de rotas para heatmap
    routes = [
        Route([1, 2, 3, 4, 5], [2.1, 1.8, 2.5, 1.9], [15, 20, 18, 12], 8.3, 65, True),
        Route([1, 6, 7, 8], [1.5, 2.2, 1.7], [22, 16, 14], 5.4, 52, True),
        Route([2, 9, 10, 3], [1.9, 1.6, 2.0], [18, 19, 15], 5.5, 52, True),
        Route([4, 11, 12], [2.3, 1.4], [25, 20], 3.7, 45, True)
    ]
    
    stop_coordinates = {
        1: (0.0, 0.0), 2: (1.2, 1.5), 3: (2.8, 2.1), 4: (3.5, 1.2), 5: (2.1, 3.8),
        6: (0.8, 2.9), 7: (1.9, 4.2), 8: (3.2, 3.5), 9: (0.5, 1.8), 10: (1.7, 2.3),
        11: (4.1, 2.8), 12: (3.8, 4.0)
    }
    
    return {
        'convergence': convergence_data,
        'comparative': comparative_data,
        'quality': quality_data,
        'executions': executions_data,
        'routes': routes,
        'coordinates': stop_coordinates
    }


def demonstrate_basic_visualizations():
    """Demonstra visualizações básicas."""
    print("=== Demonstração de Visualizações Básicas ===")
    
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()
    print(f"Salvando visualizações em: {temp_dir}")
    
    try:
        # Inicializa motor de visualização
        viz_engine = VisualizationEngine(output_dir=temp_dir, style="academic")
        
        # Cria dados de exemplo
        data = create_sample_data()
        
        # Gera gráfico de convergência
        print("\n1. Gerando gráfico de convergência...")
        conv_path = viz_engine.generate_convergence_plot(
            data['convergence'], "ACS-DEMO"
        )
        print(f"   Salvo em: {conv_path}")
        
        # Gera gráfico comparativo
        print("\n2. Gerando gráfico comparativo...")
        comp_path = viz_engine.generate_comparison_bar_chart(
            data['comparative'], ["ACS-TIME", "ACS-VEHICLE"]
        )
        print(f"   Salvo em: {comp_path}")
        
        # Gera gráfico de qualidade
        print("\n3. Gerando gráfico de métricas de qualidade...")
        qual_path = viz_engine.generate_quality_metrics_chart(data['quality'])
        print(f"   Salvo em: {qual_path}")
        
        # Gera comparação múltipla
        print("\n4. Gerando comparação de múltiplas execuções...")
        multi_path = viz_engine.generate_multiple_convergence_comparison(data['executions'])
        print(f"   Salvo em: {multi_path}")
        
        print(f"\n✓ Visualizações básicas geradas com sucesso!")
        print(f"  Total de arquivos: {len(viz_engine.get_generated_files())}")
        
    finally:
        # Limpa diretório temporário
        shutil.rmtree(temp_dir)


def demonstrate_advanced_visualizations():
    """Demonstra visualizações avançadas."""
    print("\n=== Demonstração de Visualizações Avançadas ===")
    
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()
    print(f"Salvando visualizações em: {temp_dir}")
    
    try:
        # Inicializa motor de visualização
        viz_engine = VisualizationEngine(output_dir=temp_dir, style="academic")
        
        # Cria dados de exemplo
        data = create_sample_data()
        
        # Gera heatmap de utilização
        print("\n1. Gerando heatmap de utilização de paradas...")
        heatmap_path = viz_engine.generate_stop_utilization_heatmap(
            data['routes'], data['coordinates']
        )
        print(f"   Salvo em: {heatmap_path}")
        
        # Gera histograma de distribuição
        print("\n2. Gerando histograma de distribuição de fitness...")
        hist_path = viz_engine.generate_fitness_distribution_histogram(data['executions'])
        print(f"   Salvo em: {hist_path}")
        
        # Gera gráfico com intervalos de confiança
        print("\n3. Gerando gráfico com intervalos de confiança...")
        conf_path = viz_engine.generate_convergence_confidence_intervals(
            data['executions'], confidence_level=0.95
        )
        print(f"   Salvo em: {conf_path}")
        
        print(f"\n✓ Visualizações avançadas geradas com sucesso!")
        print(f"  Total de arquivos: {len(viz_engine.get_generated_files())}")
        
    finally:
        # Limpa diretório temporário
        shutil.rmtree(temp_dir)


def demonstrate_presentation_formatting():
    """Demonstra formatação para apresentação."""
    print("\n=== Demonstração de Formatação para Apresentação ===")
    
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()
    print(f"Salvando visualizações em: {temp_dir}")
    
    try:
        # Inicializa motor com estilo de apresentação
        viz_engine = VisualizationEngine(output_dir=temp_dir, style="presentation")
        viz_engine.set_export_formats(['png', 'svg'])
        
        # Cria dados de exemplo
        data = create_sample_data()
        
        # Gera gráfico de convergência formatado
        print("\n1. Gerando gráfico de convergência formatado...")
        conv_paths = viz_engine.generate_formatted_convergence_plot(
            data['convergence'], "ACS-PRESENTATION",
            subtitle="Demonstração de Formatação Acadêmica",
            footer="Gerado pelo Sistema de Métricas ACO"
        )
        print(f"   Salvos em: {conv_paths}")
        
        # Gera gráfico comparativo formatado
        print("\n2. Gerando gráfico comparativo formatado...")
        comp_paths = viz_engine.generate_formatted_comparison_chart(
            data['comparative'], ["ACS-TIME", "ACS-VEHICLE", "ACS-HYBRID"],
            subtitle="Comparação de Performance",
            footer="Dados de 10 execuções independentes"
        )
        print(f"   Salvos em: {comp_paths}")
        
        # Gera resumo de apresentação
        print("\n3. Gerando resumo de apresentação...")
        summary_paths = viz_engine.create_presentation_summary(
            {"demo": "data"}, title="Resultados do Sistema ACO"
        )
        print(f"   Salvos em: {summary_paths}")
        
        print(f"\n✓ Visualizações formatadas geradas com sucesso!")
        print(f"  Total de arquivos: {len(viz_engine.get_generated_files())}")
        
    finally:
        # Limpa diretório temporário
        shutil.rmtree(temp_dir)


def main():
    """Função principal da demonstração."""
    print("🎨 DEMONSTRAÇÃO DO MOTOR DE VISUALIZAÇÃO")
    print("=" * 50)
    
    try:
        demonstrate_basic_visualizations()
        demonstrate_advanced_visualizations()
        demonstrate_presentation_formatting()
        
        print("\n" + "=" * 50)
        print("✅ Demonstração concluída com sucesso!")
        print("\nO VisualizationEngine oferece:")
        print("• Gráficos de convergência com análise detalhada")
        print("• Comparações entre algoritmos e configurações")
        print("• Heatmaps de utilização de recursos")
        print("• Histogramas de distribuição de fitness")
        print("• Intervalos de confiança para análise estatística")
        print("• Formatação acadêmica e para apresentações")
        print("• Exportação em múltiplos formatos (PNG, SVG, PDF, EPS)")
        print("• Estilos configuráveis (acadêmico/apresentação)")
        
    except Exception as e:
        print(f"\n❌ Erro durante demonstração: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())