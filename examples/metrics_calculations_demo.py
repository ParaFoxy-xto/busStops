# examples/metrics_calculations_demo.py

"""
Demonstração de como integrar os algoritmos simples com o sistema de métricas.

Este script mostra como usar os algoritmos de comparação junto com
as métricas de qualidade e análise estatística do sistema principal.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.aco.simple_aco import SimpleACO
from rota_aco.aco.brute_force import BruteForceOptimizer, GreedyOptimizer
from rota_aco.graph.loader import GraphLoader


class AlgorithmMetricsCalculator:
    """
    Calculadora de métricas para comparação de algoritmos.
    """
    
    def __init__(self):
        self.results = {}
        self.execution_history = []
    
    def calculate_route_metrics(self, route: List[Any], meta_edges: Dict, stops_to_visit: List[Any]) -> Dict:
        """
        Calcula métricas detalhadas de uma rota.
        """
        if not route:
            return {
                'total_distance': float('inf'),
                'total_time': float('inf'),
                'coverage': 0.0,
                'efficiency': 0.0,
                'route_length': 0,
                'valid': False
            }
        
        # Métricas básicas
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            edge_data = meta_edges.get(edge, {})
            total_distance += edge_data.get('distance', 0.0)
            total_time += edge_data.get('time', 0.0)
        
        # Cobertura
        visited_stops = set(node for node in route if node in stops_to_visit)
        coverage = len(visited_stops) / len(stops_to_visit) if stops_to_visit else 1.0
        
        # Eficiência (cobertura por unidade de distância)
        efficiency = coverage / total_distance if total_distance > 0 else 0.0
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'coverage': coverage,
            'efficiency': efficiency,
            'route_length': len(route),
            'visited_stops': len(visited_stops),
            'valid': coverage >= 0.99  # Considera válida se visita quase todos os pontos
        }
    
    def run_algorithm_comparison(
        self, 
        graph, 
        meta_edges, 
        stops_to_visit, 
        start_node, 
        exit_node,
        algorithms=['greedy', 'aco', 'brute_force'],
        aco_config=None,
        bf_config=None
    ) -> Dict:
        """
        Executa comparação completa entre algoritmos.
        """
        if aco_config is None:
            aco_config = {'n_ants': 10, 'n_iterations': 50}
        
        if bf_config is None:
            bf_config = {'max_time_seconds': 30.0}
        
        comparison_results = {}
        
        for algorithm in algorithms:
            print(f"\nExecutando {algorithm.upper()}...")
            
            start_time = time.time()
            
            try:
                if algorithm == 'greedy':
                    optimizer = GreedyOptimizer(
                        graph=graph,
                        meta_edges=meta_edges,
                        stops_to_visit=stops_to_visit,
                        start_node=start_node,
                        exit_node=exit_node
                    )
                    route, distance, stats = optimizer.run()
                
                elif algorithm == 'aco':
                    optimizer = SimpleACO(
                        graph=graph,
                        meta_edges=meta_edges,
                        stops_to_visit=stops_to_visit,
                        start_node=start_node,
                        exit_node=exit_node
                    )
                    route, distance, stats = optimizer.run(**aco_config)
                
                elif algorithm == 'brute_force':
                    if len(stops_to_visit) > 7:
                        print(f"  Muitos pontos ({len(stops_to_visit)}) - usando limite de tempo")
                    
                    optimizer = BruteForceOptimizer(
                        graph=graph,
                        meta_edges=meta_edges,
                        stops_to_visit=stops_to_visit,
                        start_node=start_node,
                        exit_node=exit_node
                    )
                    
                    if len(stops_to_visit) > 7:
                        route, distance, stats = optimizer.run_limited(**bf_config)
                    else:
                        route, distance, stats = optimizer.run()
                
                execution_time = time.time() - start_time
                
                # Calcular métricas detalhadas
                route_metrics = self.calculate_route_metrics(route, meta_edges, stops_to_visit)
                
                # Combinar resultados
                comparison_results[algorithm] = {
                    'route': route,
                    'basic_stats': stats,
                    'route_metrics': route_metrics,
                    'execution_time': execution_time,
                    'success': route_metrics['valid']
                }
                
                print(f"  Distância: {route_metrics['total_distance']:.2f}")
                print(f"  Cobertura: {route_metrics['coverage']:.2%}")
                print(f"  Tempo: {execution_time:.4f}s")
                print(f"  Sucesso: {'Sim' if route_metrics['valid'] else 'Não'}")
                
            except Exception as e:
                print(f"  Erro: {e}")
                comparison_results[algorithm] = {
                    'error': str(e),
                    'success': False
                }
        
        return comparison_results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analisa os resultados da comparação.
        """
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            return {'error': 'Nenhum algoritmo executou com sucesso'}
        
        analysis = {
            'summary': {},
            'rankings': {},
            'recommendations': []
        }
        
        # Resumo por algoritmo
        for algo, result in successful_results.items():
            metrics = result['route_metrics']
            analysis['summary'][algo] = {
                'distance': metrics['total_distance'],
                'coverage': metrics['coverage'],
                'efficiency': metrics['efficiency'],
                'execution_time': result['execution_time'],
                'route_length': metrics['route_length']
            }
        
        # Rankings
        if len(successful_results) > 1:
            # Ranking por distância (menor é melhor)
            distance_ranking = sorted(
                successful_results.items(),
                key=lambda x: x[1]['route_metrics']['total_distance']
            )
            analysis['rankings']['distance'] = [algo for algo, _ in distance_ranking]
            
            # Ranking por tempo (menor é melhor)
            time_ranking = sorted(
                successful_results.items(),
                key=lambda x: x[1]['execution_time']
            )
            analysis['rankings']['speed'] = [algo for algo, _ in time_ranking]
            
            # Ranking por eficiência (maior é melhor)
            efficiency_ranking = sorted(
                successful_results.items(),
                key=lambda x: x[1]['route_metrics']['efficiency'],
                reverse=True
            )
            analysis['rankings']['efficiency'] = [algo for algo, _ in efficiency_ranking]
        
        # Recomendações
        best_distance = min(r['route_metrics']['total_distance'] for r in successful_results.values())
        fastest_time = min(r['execution_time'] for r in successful_results.values())
        
        for algo, result in successful_results.items():
            metrics = result['route_metrics']
            
            # Recomendação baseada na performance
            if metrics['total_distance'] <= best_distance * 1.05:  # Dentro de 5% do ótimo
                if result['execution_time'] <= fastest_time * 2:  # Não muito lento
                    analysis['recommendations'].append(f"{algo}: Excelente equilíbrio qualidade/tempo")
                else:
                    analysis['recommendations'].append(f"{algo}: Ótima qualidade, mas lento")
            elif result['execution_time'] <= fastest_time * 1.1:  # Muito rápido
                analysis['recommendations'].append(f"{algo}: Muito rápido, qualidade aceitável")
        
        return analysis
    
    def generate_report(self, results: Dict, analysis: Dict, output_file: str = None) -> str:
        """
        Gera relatório detalhado da comparação.
        """
        report_lines = []
        
        report_lines.append("RELATÓRIO DE COMPARAÇÃO DE ALGORITMOS")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Resumo executivo
        if 'summary' in analysis:
            report_lines.append("RESUMO EXECUTIVO")
            report_lines.append("-" * 20)
            
            for algo, metrics in analysis['summary'].items():
                report_lines.append(f"{algo.upper()}:")
                report_lines.append(f"  Distância: {metrics['distance']:.2f}")
                report_lines.append(f"  Cobertura: {metrics['coverage']:.2%}")
                report_lines.append(f"  Tempo: {metrics['execution_time']:.4f}s")
                report_lines.append(f"  Eficiência: {metrics['efficiency']:.4f}")
                report_lines.append("")
        
        # Rankings
        if 'rankings' in analysis:
            report_lines.append("RANKINGS")
            report_lines.append("-" * 10)
            
            for category, ranking in analysis['rankings'].items():
                report_lines.append(f"{category.title()}: {' > '.join(ranking)}")
            report_lines.append("")
        
        # Recomendações
        if 'recommendations' in analysis:
            report_lines.append("RECOMENDAÇÕES")
            report_lines.append("-" * 15)
            
            for rec in analysis['recommendations']:
                report_lines.append(f"• {rec}")
            report_lines.append("")
        
        # Detalhes técnicos
        report_lines.append("DETALHES TÉCNICOS")
        report_lines.append("-" * 18)
        
        for algo, result in results.items():
            if result.get('success', False):
                report_lines.append(f"{algo.upper()}:")
                
                basic_stats = result.get('basic_stats', {})
                route_metrics = result.get('route_metrics', {})
                
                report_lines.append(f"  Rota válida: {route_metrics.get('valid', False)}")
                report_lines.append(f"  Nós na rota: {route_metrics.get('route_length', 0)}")
                report_lines.append(f"  Pontos visitados: {route_metrics.get('visited_stops', 0)}")
                
                # Estatísticas específicas do algoritmo
                if 'iterations' in basic_stats:
                    report_lines.append(f"  Iterações: {basic_stats['iterations']}")
                if 'routes_tested' in basic_stats:
                    report_lines.append(f"  Rotas testadas: {basic_stats['routes_tested']}")
                
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Salvar em arquivo se especificado
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"Relatório salvo em: {output_path}")
        
        return report_text


def main():
    """
    Demonstração principal.
    """
    print("DEMONSTRAÇÃO - MÉTRICAS DE ALGORITMOS")
    print("=" * 40)
    
    # Criar calculadora de métricas
    calculator = AlgorithmMetricsCalculator()
    
    # Criar grafo de teste
    loader = GraphLoader()
    graph, meta_edges = loader.create_simple_test_graph()
    
    # Configurar cenário
    start_node = 'A'
    exit_node = 'F'
    stops_to_visit = ['B', 'C', 'D', 'E']
    
    print(f"Cenário: {len(graph.nodes())} nós, {len(stops_to_visit)} pontos para visitar")
    
    # Executar comparação
    results = calculator.run_algorithm_comparison(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node,
        algorithms=['greedy', 'aco', 'brute_force'],
        aco_config={'n_ants': 8, 'n_iterations': 30}
    )
    
    # Analisar resultados
    analysis = calculator.analyze_results(results)
    
    # Gerar relatório
    report = calculator.generate_report(
        results, 
        analysis, 
        output_file="output/metrics_comparison_report.txt"
    )
    
    print("\n" + "=" * 50)
    print("RELATÓRIO GERADO")
    print("=" * 50)
    print(report)
    
    # Salvar dados em JSON para análise posterior
    json_output = {
        'results': {
            algo: {
                'success': result.get('success', False),
                'route_metrics': result.get('route_metrics', {}),
                'execution_time': result.get('execution_time', 0),
                'basic_stats': {k: v for k, v in result.get('basic_stats', {}).items() 
                               if isinstance(v, (int, float, str, bool))}
            }
            for algo, result in results.items()
        },
        'analysis': analysis
    }
    
    json_path = Path("output/metrics_comparison_data.json")
    json_path.parent.mkdir(exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nDados salvos em: {json_path}")
    
    return results, analysis


if __name__ == "__main__":
    main()