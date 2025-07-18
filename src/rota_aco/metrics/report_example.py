"""
Exemplo de uso do ReportGenerator.

Este script demonstra como usar o sistema de geração de relatórios
para criar relatórios formatados em markdown a partir de dados de métricas ACO.
"""

from datetime import datetime
from .report_generator import ReportGenerator
from .data_models import (
    MetricsReport, ExecutionSummary, RouteQualityMetrics,
    ConvergenceMetrics, ComparativeMetrics, DomainMetrics
)
from .config import MetricsConfig


def create_sample_metrics_report() -> MetricsReport:
    """Cria um relatório de métricas de exemplo para demonstração."""
    
    # Resumo de execução
    execution_summary = ExecutionSummary(
        total_executions=10,
        successful_executions=9,
        success_rate=0.9,
        avg_execution_time=45.5,
        avg_iterations_to_convergence=150.0,
        best_overall_fitness=0.8542,
        algorithm_type="ACS-TIME"
    )
    
    # Métricas de qualidade
    quality_metrics = RouteQualityMetrics(
        valid_routes_percentage=0.95,
        demand_coverage_percentage=0.88,
        vehicle_utilization_efficiency=0.82,
        capacity_violations=2,
        opposite_stops_violations=1,
        average_route_length=12.5,
        route_length_variance=3.2,
        load_balancing_index=0.75
    )
    
    # Métricas de convergência
    convergence_metrics = ConvergenceMetrics(
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
    
    # Métricas comparativas
    comparative_metrics = ComparativeMetrics(
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
    
    # Métricas de domínio
    domain_metrics = DomainMetrics(
        estimated_travel_time=35.5,
        average_transfers=1.2,
        geographic_coverage=25.8,
        load_balancing_index=0.75,
        energy_efficiency=2.3,
        accessibility_index=3.5,
        service_frequency=4.2,
        route_overlap_percentage=0.15
    )
    
    # Relatório completo
    return MetricsReport(
        execution_summary=execution_summary,
        quality_metrics=quality_metrics,
        convergence_analysis=convergence_metrics,
        comparative_analysis=comparative_metrics,
        domain_metrics=domain_metrics,
        visualizations=[
            'output/metrics/visualizations/convergence_acs_time.png',
            'output/metrics/visualizations/quality_comparison.svg'
        ],
        config_used={
            'algorithm': 'ACS-TIME',
            'iterations': 200,
            'population_size': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1
        }
    )


def demonstrate_basic_report_generation():
    """Demonstra geração básica de relatório."""
    print("=== Demonstração: Geração Básica de Relatório ===")
    
    # Configurar sistema
    config = MetricsConfig()
    generator = ReportGenerator(config)
    
    # Criar dados de exemplo
    metrics_report = create_sample_metrics_report()
    
    # Gerar relatório
    output_path = generator.generate_report(metrics_report, "exemplo_basico.md")
    
    print(f"Relatório gerado: {output_path}")
    print("Conteúdo do relatório:")
    print("-" * 50)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Mostrar apenas as primeiras linhas
        lines = content.split('\n')[:20]
        print('\n'.join(lines))
        print("...")
    
    return output_path


def demonstrate_comprehensive_report():
    """Demonstra geração de relatório abrangente comparativo."""
    print("\n=== Demonstração: Relatório Abrangente Comparativo ===")
    
    # Configurar sistema
    config = MetricsConfig()
    generator = ReportGenerator(config)
    
    # Criar primeiro relatório (ACS-TIME)
    report1 = create_sample_metrics_report()
    
    # Criar segundo relatório (ACS-VEHICLE)
    report2 = create_sample_metrics_report()
    report2.execution_summary.algorithm_type = "ACS-VEHICLE"
    report2.execution_summary.total_executions = 8
    report2.execution_summary.successful_executions = 7
    report2.execution_summary.success_rate = 0.875
    report2.execution_summary.avg_execution_time = 52.3
    report2.execution_summary.best_overall_fitness = 0.8234
    report2.quality_metrics.valid_routes_percentage = 0.92
    report2.quality_metrics.capacity_violations = 3
    report2.quality_metrics.opposite_stops_violations = 0
    
    # Gerar relatório comparativo
    output_path = generator.generate_comprehensive_report(
        [report1, report2], 
        "exemplo_comparativo.md"
    )
    
    print(f"Relatório comparativo gerado: {output_path}")
    print("Conteúdo do relatório:")
    print("-" * 50)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Mostrar seções principais
        lines = content.split('\n')
        for i, line in enumerate(lines[:30]):
            if line.startswith('#'):
                print(line)
            elif i < 10:  # Mostrar primeiras linhas
                print(line)
        print("...")
    
    return output_path


def demonstrate_json_export():
    """Demonstra exportação para JSON."""
    print("\n=== Demonstração: Exportação para JSON ===")
    
    # Configurar sistema
    config = MetricsConfig()
    generator = ReportGenerator(config)
    
    # Criar dados de exemplo
    metrics_report = create_sample_metrics_report()
    
    # Exportar para JSON
    output_path = generator.export_to_json(metrics_report, "exemplo_dados.json")
    
    print(f"Dados exportados para JSON: {output_path}")
    print("Estrutura dos dados:")
    print("-" * 50)
    
    import json
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Mostrar estrutura principal
        for key in data.keys():
            print(f"- {key}")
            if isinstance(data[key], dict):
                for subkey in list(data[key].keys())[:3]:  # Primeiras 3 chaves
                    print(f"  - {subkey}: {data[key][subkey]}")
                if len(data[key]) > 3:
                    print(f"  - ... (mais {len(data[key]) - 3} campos)")
    
    return output_path


def demonstrate_summary_table():
    """Demonstra geração de tabela resumo."""
    print("\n=== Demonstração: Tabela Resumo Comparativa ===")
    
    # Configurar sistema
    config = MetricsConfig()
    generator = ReportGenerator(config)
    
    # Criar múltiplos relatórios
    reports = []
    
    # ACS-TIME
    report1 = create_sample_metrics_report()
    reports.append(report1)
    
    # ACS-VEHICLE
    report2 = create_sample_metrics_report()
    report2.execution_summary.algorithm_type = "ACS-VEHICLE"
    report2.execution_summary.avg_execution_time = 52.3
    report2.execution_summary.best_overall_fitness = 0.8234
    report2.quality_metrics.valid_routes_percentage = 0.92
    reports.append(report2)
    
    # Algoritmo customizado
    report3 = create_sample_metrics_report()
    report3.execution_summary.algorithm_type = "CUSTOM-ACO"
    report3.execution_summary.avg_execution_time = 38.1
    report3.execution_summary.best_overall_fitness = 0.8678
    report3.quality_metrics.valid_routes_percentage = 0.97
    reports.append(report3)
    
    # Gerar tabela resumo
    table = generator.generate_summary_table(reports)
    
    print("Tabela resumo gerada:")
    print("-" * 50)
    print(table)
    
    return table


def main():
    """Função principal para executar todas as demonstrações."""
    print("Sistema de Geração de Relatórios - Rota_ACO")
    print("=" * 60)
    
    try:
        # Demonstrações
        demonstrate_basic_report_generation()
        demonstrate_comprehensive_report()
        demonstrate_json_export()
        demonstrate_summary_table()
        
        print("\n" + "=" * 60)
        print("Todas as demonstrações foram executadas com sucesso!")
        print("Verifique os arquivos gerados no diretório output/metrics/reports/")
        
    except Exception as e:
        print(f"\nErro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()