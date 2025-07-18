#!/usr/bin/env python3
"""
Demonstração detalhada de todos os cálculos de métricas.

Este script mostra como cada métrica é calculada, com exemplos
práticos e explicações detalhadas para uso acadêmico.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rota_aco.metrics.data_models import (
    ExecutionData, IterationData, Route, Solution
)
from rota_aco.metrics.quality_metrics import RouteQualityEvaluator
from rota_aco.metrics.convergence_analyzer import ConvergenceAnalyzer
from rota_aco.metrics.statistical_analyzer import StatisticalAnalyzer
from rota_aco.metrics.transportation_metrics import TransportationMetrics
from rota_aco.metrics.config import MetricsConfig


class MetricsCalculationsDemo:
    """Demonstração de cálculos de métricas."""
    
    def __init__(self):
        """Inicializa a demonstração."""
        self.config = MetricsConfig()
        print("DEMONSTRAÇÃO DE CÁLCULOS DE MÉTRICAS")
        print("=" * 60)
        print("Este script demonstra como cada métrica é calculada")
        print("com exemplos práticos e dados realistas.\n")
    
    def create_sample_routes(self):
        """Cria rotas de exemplo para demonstração."""
        print("1. CRIANDO ROTAS DE EXEMPLO")
        print("-" * 30)
        
        # Rota 1: Válida e eficiente
        route1 = Route(
            stops=[1, 2, 3, 4, 5],
            distances=[1200, 800, 1500, 900],  # metros
            passenger_load=[15, 25, 35, 20],   # passageiros por segmento
            total_distance=4400,
            total_passengers=35,  # máximo no veículo
            is_valid=True
        )
        
        # Rota 2: Válida mas menos eficiente
        route2 = Route(
            stops=[1, 6, 7, 8, 5],
            distances=[2000, 1200, 1800, 1100],
            passenger_load=[10, 20, 30, 25],
            total_distance=6100,
            total_passengers=30,
            is_valid=True
        )
        
        # Rota 3: Inválida (excede capacidade)
        route3 = Route(
            stops=[1, 9, 10, 11, 5],
            distances=[1800, 1000, 1400, 1200],
            passenger_load=[40, 60, 75, 50],  # Excede 70 passageiros
            total_distance=5400,
            total_passengers=75,  # Excede capacidade
            is_valid=False
        )
        
        routes = [route1, route2, route3]
        
        print("Rotas criadas:")
        for i, route in enumerate(routes, 1):
            print(f"  Rota {i}: {len(route.stops)} paradas, "
                  f"{route.total_distance}m, {route.total_passengers} pass., "
                  f"{'Válida' if route.is_valid else 'INVÁLIDA'}")
        
        return routes
    
    def demonstrate_quality_metrics(self, routes):
        """Demonstra cálculo de métricas de qualidade."""
        print("\n2. MÉTRICAS DE QUALIDADE DAS ROTAS")
        print("-" * 40)
        
        evaluator = RouteQualityEvaluator(capacity_limit=70)
        
        # Simular dados do grafo
        total_stops = 15  # Total de paradas no sistema
        capacity_limit = 70
        
        # Calcular métricas
        metrics = evaluator.evaluate_routes(routes)
        
        print("Cálculos detalhados:")
        print(f"  Total de rotas: {len(routes)}")
        print(f"  Rotas válidas: {sum(1 for r in routes if r.is_valid)}")
        print(f"  % Rotas válidas: {metrics.valid_routes_percentage:.1f}%")
        print(f"    Cálculo: ({sum(1 for r in routes if r.is_valid)} / {len(routes)}) × 100")
        
        # Cobertura de demanda
        covered_stops = set()
        for route in routes:
            covered_stops.update(route.stops[1:-1])  # Excluir início e fim
        
        print(f"\n  Paradas cobertas: {len(covered_stops)} de {total_stops}")
        print(f"  % Cobertura: {metrics.demand_coverage_percentage:.1f}%")
        print(f"    Cálculo: ({len(covered_stops)} / {total_stops}) × 100")
        
        # Utilização de veículos
        total_capacity = len(routes) * capacity_limit
        total_passengers = sum(r.total_passengers for r in routes)
        
        print(f"\n  Capacidade total: {total_capacity} passageiros")
        print(f"  Passageiros transportados: {total_passengers}")
        print(f"  Eficiência de utilização: {metrics.vehicle_utilization_efficiency:.1f}%")
        print(f"    Cálculo: ({total_passengers} / {total_capacity}) × 100")
        
        # Violações
        capacity_violations = sum(1 for r in routes if r.total_passengers > capacity_limit)
        print(f"\n  Violações de capacidade: {capacity_violations}")
        print(f"  Violações de paradas opostas: {metrics.opposite_stops_violations}")
        
        return metrics
    
    def demonstrate_convergence_analysis(self):
        """Demonstra análise de convergência."""
        print("\n3. ANÁLISE DE CONVERGÊNCIA")
        print("-" * 30)
        
        # Criar dados de convergência simulados
        iterations_data = []
        best_fitness = 1000.0
        
        print("Simulando evolução da convergência:")
        print("Iter\tMelhor\tMédia\tVariância")
        print("-" * 35)
        
        for i in range(50):
            # Simular melhoria gradual com ruído
            if i < 30:
                improvement = max(0, (30 - i) * 3 + np.random.normal(0, 2))
                best_fitness = max(200, best_fitness - improvement)
            else:
                # Convergência (pouca melhoria)
                improvement = max(0, np.random.normal(0, 0.5))
                best_fitness = max(200, best_fitness - improvement)
            
            avg_fitness = best_fitness + np.random.uniform(20, 100)
            variance = max(10, 300 - i * 5 + np.random.normal(0, 10))
            
            # Criar solução mock (sem imprimir)
            mock_routes = [
                Route(stops=[1, 2, 3], distances=[100, 150], passenger_load=[10, 20], 
                      total_distance=250, total_passengers=20, is_valid=True)
            ]
            solution = Solution(
                routes=mock_routes,
                total_vehicles=len(mock_routes),
                total_distance=sum(r.total_distance for r in mock_routes),
                total_passengers_served=sum(r.total_passengers for r in mock_routes),
                fitness_time=best_fitness,
                fitness_vehicle=len(mock_routes),
                is_feasible=True
            )
            
            iteration_data = IterationData(
                iteration=i,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                population_variance=variance,
                best_solution=solution
            )
            iterations_data.append(iteration_data)
            
            if i % 10 == 0 or i < 5 or i > 45:
                print(f"{i:3d}\t{best_fitness:.1f}\t{avg_fitness:.1f}\t{variance:.1f}")
        
        # Analisar convergência
        analyzer = ConvergenceAnalyzer(convergence_threshold=0.001, stability_window=50)
        conv_metrics = analyzer.analyze_convergence(iterations_data)
        
        print(f"\nResultados da análise:")
        print(f"  Ponto de convergência: iteração {conv_metrics.convergence_point}")
        print(f"  Estabilidade final: {conv_metrics.final_stability:.3f}")
        print(f"  Taxa de melhoria: {conv_metrics.improvement_rate:.3f}")
        print(f"  Platô detectado: {'Sim' if conv_metrics.plateau_detection else 'Não'}")
        print(f"  Velocidade de convergência: {conv_metrics.convergence_speed:.3f}")
        
        # Explicar cálculos
        print(f"\nCálculos detalhados:")
        print(f"  Convergência: quando melhoria < {self.config.convergence_threshold}")
        print(f"  Estabilidade: desvio padrão das últimas {self.config.stability_window} iterações")
        print(f"  Taxa de melhoria: (fitness_inicial - fitness_final) / fitness_inicial")
        
        return iterations_data, conv_metrics
    
    def demonstrate_statistical_analysis(self):
        """Demonstra análise estatística comparativa."""
        print("\n4. ANÁLISE ESTATÍSTICA COMPARATIVA")
        print("-" * 40)
        
        # Criar múltiplas execuções simuladas
        executions = []
        
        print("Criando execuções simuladas:")
        for i in range(5):
            # Simular variação nos resultados
            base_fitness = 300 + np.random.normal(0, 50)
            execution_time = 25 + np.random.normal(0, 5)
            
            # Criar dados de iteração simplificados
            iterations_data = []
            for j in range(20):
                fitness = base_fitness + np.random.normal(0, 10)
                iteration_data = IterationData(
                    iteration=j,
                    best_fitness=fitness,
                    avg_fitness=fitness + 20,
                    population_variance=50,
                    best_solution=None
                )
                iterations_data.append(iteration_data)
            
            # Criar rotas mock sem imprimir
            mock_routes = [
                Route(stops=[1, 2, 3], distances=[100, 150], passenger_load=[10, 20], 
                      total_distance=250, total_passengers=20, is_valid=True)
            ]
            
            execution = ExecutionData(
                execution_id=f"exec_{i:03d}",
                algorithm_type="ACS-TIME" if i % 2 == 0 else "ACS-VEHICLE",
                config={'n_ants': 20, 'n_iterations': 20},
                routes=mock_routes,
                iterations_data=iterations_data,
                execution_time=execution_time,
                success=True
            )
            executions.append(execution)
            
            print(f"  Execução {i+1}: fitness={base_fitness:.1f}, tempo={execution_time:.1f}s")
        
        # Análise estatística
        analyzer = StatisticalAnalyzer()
        stats = analyzer.analyze_multiple_executions(executions)
        
        print(f"\nEstatísticas calculadas:")
        print(f"  Fitness final:")
        print(f"    Média: {stats.avg_best_fitness:.2f}")
        print(f"    Mediana: {stats.median_best_fitness:.2f}")
        print(f"    Desvio padrão: {stats.std_best_fitness:.2f}")
        
        print(f"\n  Tempo de execução:")
        print(f"    Média: {stats.avg_execution_time:.2f}s")
        print(f"    Desvio padrão: {stats.std_execution_time:.2f}s")
        
        print(f"\n  Taxa de sucesso: {stats.success_rate:.1f}%")
        print(f"    Cálculo: {stats.successful_executions} / {stats.total_executions}")
        
        print(f"\n  Convergência:")
        print(f"    Iterações médias: {stats.avg_convergence_iterations:.1f}")
        
        # Comparação por algoritmo
        acs_time_results = [e for e in executions if e.algorithm_type == "ACS-TIME"]
        acs_vehicle_results = [e for e in executions if e.algorithm_type == "ACS-VEHICLE"]
        
        if acs_time_results and acs_vehicle_results:
            time_fitness = [e.iterations_data[-1].best_fitness for e in acs_time_results]
            vehicle_fitness = [e.iterations_data[-1].best_fitness for e in acs_vehicle_results]
            
            print(f"\n  Comparação por algoritmo:")
            print(f"    ACS-TIME: {np.mean(time_fitness):.2f} ± {np.std(time_fitness):.2f}")
            print(f"    ACS-VEHICLE: {np.mean(vehicle_fitness):.2f} ± {np.std(vehicle_fitness):.2f}")
        
        return stats
    
    def demonstrate_transportation_metrics(self, routes):
        """Demonstra métricas específicas de transporte."""
        print("\n5. MÉTRICAS DE TRANSPORTE")
        print("-" * 30)
        
        # Simular dados do grafo
        mock_graph_data = {
            'nodes': list(range(1, 16)),  # 15 nós
            'edges': {
                (1, 2): {'time': 120, 'distance': 1200},
                (2, 3): {'time': 100, 'distance': 800},
                (3, 4): {'time': 180, 'distance': 1500},
                # ... mais arestas seriam definidas aqui
            },
            'area_km2': 25.0,  # Área coberta em km²
            'stops_coordinates': {
                1: (-15.7801, -47.9292),
                2: (-15.7851, -47.9342),
                # ... mais coordenadas
            }
        }
        
        calculator = TransportationMetrics(graph_data=mock_graph_data)
        
        print("Calculando métricas de transporte:")
        
        # 1. Tempo total de viagem estimado
        total_time = 0
        for route in routes:
            route_time = 0
            for i in range(len(route.stops) - 1):
                # Simular tempo baseado na distância
                segment_distance = route.distances[i] if i < len(route.distances) else 1000
                # Assumir velocidade média de 30 km/h
                segment_time = (segment_distance / 1000) * (60 / 30)  # minutos
                route_time += segment_time
            total_time += route_time
        
        print(f"  Tempo total de viagem: {total_time:.1f} minutos")
        print(f"    Cálculo: soma dos tempos de todos os segmentos")
        print(f"    Velocidade assumida: 30 km/h")
        
        # 2. Número médio de transferências
        # Simplificado: assumir que passageiros podem precisar trocar de rota
        total_stops_covered = set()
        for route in routes:
            total_stops_covered.update(route.stops[1:-1])
        
        avg_transfers = max(0, len(routes) - 1) if len(routes) > 1 else 0
        print(f"\n  Número médio de transferências: {avg_transfers:.1f}")
        print(f"    Estimativa baseada no número de rotas necessárias")
        
        # 3. Cobertura geográfica
        covered_area_ratio = len(total_stops_covered) / len(mock_graph_data['nodes'])
        geographic_coverage = covered_area_ratio * mock_graph_data['area_km2']
        
        print(f"\n  Cobertura geográfica: {geographic_coverage:.1f} km²")
        print(f"    Cálculo: ({len(total_stops_covered)} / {len(mock_graph_data['nodes'])}) × {mock_graph_data['area_km2']} km²")
        
        # 4. Balanceamento de carga
        if routes:
            loads = [r.total_passengers for r in routes]
            load_std = np.std(loads)
            load_mean = np.mean(loads)
            load_balance_index = 1 - (load_std / load_mean) if load_mean > 0 else 0
            
            print(f"\n  Balanceamento de carga:")
            print(f"    Cargas por rota: {loads}")
            print(f"    Média: {load_mean:.1f} passageiros")
            print(f"    Desvio padrão: {load_std:.1f}")
            print(f"    Índice de balanceamento: {load_balance_index:.3f}")
            print(f"    Cálculo: 1 - (desvio_padrão / média)")
        
        # 5. Eficiência energética
        total_distance = sum(r.total_distance for r in routes)
        total_passengers = sum(r.total_passengers for r in routes)
        energy_efficiency = total_distance / total_passengers if total_passengers > 0 else float('inf')
        
        print(f"\n  Eficiência energética: {energy_efficiency:.1f} metros/passageiro")
        print(f"    Cálculo: {total_distance} metros / {total_passengers} passageiros")
        
        # 6. Índice de acessibilidade
        accessibility_index = len(total_stops_covered) / mock_graph_data['area_km2']
        
        print(f"\n  Índice de acessibilidade: {accessibility_index:.2f} paradas/km²")
        print(f"    Cálculo: {len(total_stops_covered)} paradas / {mock_graph_data['area_km2']} km²")
    
    def create_visualization_example(self, iterations_data):
        """Cria exemplo de visualização."""
        print("\n6. EXEMPLO DE VISUALIZAÇÃO")
        print("-" * 30)
        
        # Extrair dados para plotagem
        iterations = [data.iteration for data in iterations_data]
        best_fitness = [data.best_fitness for data in iterations_data]
        avg_fitness = [data.avg_fitness for data in iterations_data]
        
        # Criar gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_fitness, 'b-', label='Melhor Fitness', linewidth=2)
        plt.plot(iterations, avg_fitness, 'r--', label='Fitness Médio', linewidth=1)
        
        plt.xlabel('Iteração')
        plt.ylabel('Fitness')
        plt.title('Evolução da Convergência - Exemplo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Marcar ponto de convergência
        conv_point = 30  # Exemplo
        plt.axvline(x=conv_point, color='g', linestyle=':', 
                   label=f'Convergência (iter {conv_point})')
        plt.legend()
        
        # Salvar gráfico
        output_file = 'convergence_example.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de convergência salvo em: {output_file}")
        print("Elementos do gráfico:")
        print("  - Linha azul sólida: evolução do melhor fitness")
        print("  - Linha vermelha tracejada: fitness médio da população")
        print("  - Linha verde pontilhada: ponto de convergência detectado")
        print("  - Grade: facilita leitura dos valores")
    
    def generate_summary_report(self):
        """Gera relatório resumo da demonstração."""
        print("\n7. RELATÓRIO RESUMO")
        print("-" * 20)
        
        report = {
            "demonstracao_metricas": {
                "data_execucao": datetime.now().isoformat(),
                "metricas_demonstradas": [
                    "Qualidade das rotas",
                    "Análise de convergência", 
                    "Análise estatística comparativa",
                    "Métricas de transporte",
                    "Visualizações"
                ],
                "calculos_explicados": [
                    "Porcentagem de rotas válidas",
                    "Cobertura de demanda",
                    "Eficiência de utilização de veículos",
                    "Ponto de convergência",
                    "Estabilidade da solução",
                    "Estatísticas descritivas",
                    "Tempo total de viagem",
                    "Balanceamento de carga",
                    "Eficiência energética",
                    "Índice de acessibilidade"
                ],
                "formulas_utilizadas": {
                    "rotas_validas": "(rotas_validas / total_rotas) × 100",
                    "cobertura_demanda": "(paradas_cobertas / total_paradas) × 100",
                    "utilizacao_veiculos": "(passageiros_transportados / capacidade_total) × 100",
                    "estabilidade": "desvio_padrão(últimas_N_iterações)",
                    "taxa_melhoria": "(fitness_inicial - fitness_final) / fitness_inicial",
                    "balanceamento_carga": "1 - (desvio_padrão_cargas / média_cargas)",
                    "eficiencia_energetica": "distância_total / passageiros_totais",
                    "acessibilidade": "paradas_cobertas / área_km²"
                }
            }
        }
        
        # Salvar relatório
        report_file = 'demonstracao_metricas_relatorio.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Relatório detalhado salvo em: {report_file}")
        print("\nResumo das métricas demonstradas:")
        for metrica in report["demonstracao_metricas"]["metricas_demonstradas"]:
            print(f"  ✅ {metrica}")
    
    def run_complete_demo(self):
        """Executa demonstração completa."""
        print("Iniciando demonstração completa dos cálculos de métricas...\n")
        
        # 1. Criar dados de exemplo
        routes = self.create_sample_routes()
        
        # 2. Demonstrar métricas de qualidade
        quality_metrics = self.demonstrate_quality_metrics(routes)
        
        # 3. Demonstrar análise de convergência
        iterations_data, conv_metrics = self.demonstrate_convergence_analysis()
        
        # 4. Demonstrar análise estatística
        stats = self.demonstrate_statistical_analysis()
        
        # 5. Demonstrar métricas de transporte
        self.demonstrate_transportation_metrics(routes)
        
        # 6. Criar visualização exemplo
        self.create_visualization_example(iterations_data)
        
        # 7. Gerar relatório resumo
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("DEMONSTRAÇÃO CONCLUÍDA")
        print("=" * 60)
        print("Todos os cálculos de métricas foram demonstrados com:")
        print("  ✅ Dados de exemplo realistas")
        print("  ✅ Explicações detalhadas dos cálculos")
        print("  ✅ Fórmulas matemáticas utilizadas")
        print("  ✅ Interpretação dos resultados")
        print("  ✅ Exemplo de visualização")
        print("  ✅ Relatório resumo em JSON")
        print("\nEste exemplo pode ser usado como referência para")
        print("entender como o sistema de métricas funciona internamente.")


def main():
    """Função principal."""
    demo = MetricsCalculationsDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()