# examples/benchmark_algorithms.py

"""
Script para executar múltiplas rodadas dos algoritmos simples para comparação estatística.

Este script executa cada algoritmo N vezes no mesmo grafo para coletar
estatísticas que podem ser comparadas com o sistema ACS duplo.
"""

import sys
import time
import json
import statistics
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.aco.simple_aco import SimpleACO
from rota_aco.aco.brute_force import BruteForceOptimizer, GreedyOptimizer
from rota_aco.graph.loader import GraphLoader


class AlgorithmBenchmark:
    """
    Classe para executar benchmark de algoritmos com múltiplas execuções.
    """
    
    def __init__(self, graph_file: str):
        self.graph_file = graph_file
        self.graph = None
        self.meta_edges = None
        self.stops_to_visit = None
        self.start_node = None
        self.exit_node = None
        self.results = {}
        
    def load_graph(self) -> bool:
        """
        Carrega o grafo especificado.
        """
        try:
            loader = GraphLoader()
            graph_path = Path(__file__).parent.parent / self.graph_file
            
            if not graph_path.exists():
                print(f"Arquivo não encontrado: {graph_path}")
                return False
            
            print(f"Carregando grafo: {graph_path}")
            self.graph, self.meta_edges = loader.load_graph(str(graph_path))
            
            # Configurar cenário de teste
            nodes = list(self.graph.nodes())
            print(f"Grafo carregado: {len(nodes)} nós, {len(self.graph.edges())} arestas")
            
            if len(nodes) < 5:
                print("Grafo muito pequeno para teste significativo")
                return False
            
            # Definir pontos de início e fim
            self.start_node = nodes[0]
            self.exit_node = nodes[-1]
            
            # Selecionar pontos para visitar (excluindo início e fim)
            available_stops = [node for node in nodes if node not in [self.start_node, self.exit_node]]
            
            # Limitar a 6 pontos para evitar explosão combinatória na força bruta
            max_stops = min(13, len(available_stops))
            
            # Usar seed fixa para resultados reproduzíveis
            random.seed(42)
            self.stops_to_visit = random.sample(available_stops, max_stops)
            
            print(f"Início: {self.start_node}")
            print(f"Fim: {self.exit_node}")
            print(f"Pontos a visitar ({len(self.stops_to_visit)}): {self.stops_to_visit}")
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar grafo: {e}")
            return False
    
    def run_single_execution(self, algorithm: str, run_id: int, config: Dict) -> Dict:
        """
        Executa uma única rodada de um algoritmo.
        """
        # Usar seed diferente para cada execução para variabilidade
        random.seed(42 + run_id)
        
        start_time = time.time()
        
        try:
            if algorithm == 'aco':
                optimizer = SimpleACO(
                    graph=self.graph,
                    meta_edges=self.meta_edges,
                    stops_to_visit=self.stops_to_visit,
                    start_node=self.start_node,
                    exit_node=self.exit_node,
                    alpha=config.get('alpha', 1.0),
                    beta=config.get('beta', 2.0),
                    rho=config.get('rho', 0.1),
                    q_param=config.get('q_param', 1.0)
                )
                
                route, distance, stats = optimizer.run(
                    n_ants=config.get('n_ants', 10),
                    n_iterations=config.get('n_iterations', 50),
                    verbose=False
                )
                
            elif algorithm == 'brute_force':
                optimizer = BruteForceOptimizer(
                    graph=self.graph,
                    meta_edges=self.meta_edges,
                    stops_to_visit=self.stops_to_visit,
                    start_node=self.start_node,
                    exit_node=self.exit_node,
                    max_permutations=config.get('max_permutations', 10000)
                )
                
                if len(self.stops_to_visit) > 6:
                    route, distance, stats = optimizer.run_limited(
                        max_time_seconds=config.get('max_time_seconds', 60.0),
                        verbose=False
                    )
                else:
                    route, distance, stats = optimizer.run(verbose=False)
                    
            elif algorithm == 'greedy':
                optimizer = GreedyOptimizer(
                    graph=self.graph,
                    meta_edges=self.meta_edges,
                    stops_to_visit=self.stops_to_visit,
                    start_node=self.start_node,
                    exit_node=self.exit_node
                )
                
                route, distance, stats = optimizer.run(verbose=False)
                
            else:
                raise ValueError(f"Algoritmo desconhecido: {algorithm}")
            
            execution_time = time.time() - start_time
            
            # Calcular métricas adicionais
            coverage = self._calculate_coverage(route)
            route_length = len(route) if route else 0
            
            return {
                'run_id': run_id,
                'success': True,
                'distance': distance,
                'coverage': coverage,
                'execution_time': execution_time,
                'route_length': route_length,
                'route': route,
                'algorithm_stats': stats
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _calculate_coverage(self, route: List[Any]) -> float:
        """
        Calcula a cobertura dos pontos de interesse.
        """
        if not self.stops_to_visit or not route:
            return 0.0
        
        visited_stops = set(node for node in route if node in self.stops_to_visit)
        return len(visited_stops) / len(self.stops_to_visit)
    
    def run_benchmark(self, algorithms: List[str], n_runs: int = 10, configs: Dict = None) -> Dict:
        """
        Executa benchmark completo com múltiplas rodadas.
        """
        if configs is None:
            configs = {
                'aco': {
                    'n_ants': 10,
                    'n_iterations': 50,
                    'alpha': 1.0,
                    'beta': 2.0,
                    'rho': 0.1,
                    'q_param': 1.0
                },
                'brute_force': {
                    'max_permutations': 10000,
                    'max_time_seconds': 60.0
                },
                'greedy': {}
            }
        
        print(f"\nIniciando benchmark com {n_runs} execuções por algoritmo...")
        print("=" * 60)
        
        benchmark_results = {}
        
        for algorithm in algorithms:
            print(f"\nExecutando {algorithm.upper()}...")
            
            algorithm_results = []
            successful_runs = 0
            
            for run_id in range(1, n_runs + 1):
                print(f"  Execução {run_id}/{n_runs}...", end=" ")
                
                result = self.run_single_execution(
                    algorithm, 
                    run_id, 
                    configs.get(algorithm, {})
                )
                
                algorithm_results.append(result)
                
                if result['success']:
                    successful_runs += 1
                    print(f"OK (dist: {result['distance']:.2f}, tempo: {result['execution_time']:.3f}s)")
                else:
                    print(f"ERRO: {result.get('error', 'Desconhecido')}")
            
            # Calcular estatísticas
            successful_results = [r for r in algorithm_results if r['success']]
            
            if successful_results:
                distances = [r['distance'] for r in successful_results]
                times = [r['execution_time'] for r in successful_results]
                coverages = [r['coverage'] for r in successful_results]
                
                # Filtrar valores infinitos para estatísticas
                valid_distances = [d for d in distances if d != float('inf') and not (d != d)]  # Remove inf e NaN
                
                if valid_distances:
                    distance_stats = {
                        'mean': statistics.mean(valid_distances),
                        'median': statistics.median(valid_distances),
                        'stdev': statistics.stdev(valid_distances) if len(valid_distances) > 1 else 0.0,
                        'min': min(valid_distances),
                        'max': max(valid_distances),
                        'valid_results': len(valid_distances),
                        'invalid_results': len(distances) - len(valid_distances)
                    }
                else:
                    distance_stats = {
                        'mean': float('inf'),
                        'median': float('inf'),
                        'stdev': 0.0,
                        'min': float('inf'),
                        'max': float('inf'),
                        'valid_results': 0,
                        'invalid_results': len(distances)
                    }
                
                stats = {
                    'total_runs': n_runs,
                    'successful_runs': successful_runs,
                    'success_rate': successful_runs / n_runs,
                    'distance': distance_stats,
                    'execution_time': {
                        'mean': statistics.mean(times),
                        'median': statistics.median(times),
                        'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
                        'min': min(times),
                        'max': max(times)
                    },
                    'coverage': {
                        'mean': statistics.mean(coverages),
                        'median': statistics.median(coverages),
                        'stdev': statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
                        'min': min(coverages),
                        'max': max(coverages)
                    }
                }
                
                print(f"  Sucesso: {successful_runs}/{n_runs} ({stats['success_rate']:.1%})")
                print(f"  Distância média: {stats['distance']['mean']:.2f} ± {stats['distance']['stdev']:.2f}")
                print(f"  Tempo médio: {stats['execution_time']['mean']:.3f}s ± {stats['execution_time']['stdev']:.3f}s")
                print(f"  Cobertura média: {stats['coverage']['mean']:.2%}")
                
            else:
                stats = {
                    'total_runs': n_runs,
                    'successful_runs': 0,
                    'success_rate': 0.0,
                    'error': 'Nenhuma execução bem-sucedida'
                }
                
                print(f"  FALHA: Nenhuma execução bem-sucedida")
            
            benchmark_results[algorithm] = {
                'statistics': stats,
                'raw_results': algorithm_results,
                'config_used': configs.get(algorithm, {})
            }
        
        return benchmark_results
    
    def generate_comparison_report(self, results: Dict, output_file: str = None) -> str:
        """
        Gera relatório de comparação estatística.
        """
        report_lines = []
        
        report_lines.append("RELATÓRIO DE BENCHMARK - COMPARAÇÃO ESTATÍSTICA")
        report_lines.append("=" * 60)
        report_lines.append(f"Grafo: {self.graph_file}")
        report_lines.append(f"Pontos para visitar: {len(self.stops_to_visit)}")
        report_lines.append(f"Início: {self.start_node}, Fim: {self.exit_node}")
        report_lines.append("")
        
        # Resumo por algoritmo
        successful_algorithms = []
        
        for algorithm, data in results.items():
            stats = data['statistics']
            
            report_lines.append(f"{algorithm.upper()}")
            report_lines.append("-" * len(algorithm))
            
            if stats['successful_runs'] > 0:
                successful_algorithms.append(algorithm)
                
                report_lines.append(f"Taxa de sucesso: {stats['success_rate']:.1%} ({stats['successful_runs']}/{stats['total_runs']})")
                report_lines.append("")
                
                # Distância
                dist_stats = stats['distance']
                report_lines.append(f"DISTÂNCIA:")
                report_lines.append(f"  Média: {dist_stats['mean']:.2f}")
                report_lines.append(f"  Mediana: {dist_stats['median']:.2f}")
                report_lines.append(f"  Desvio padrão: {dist_stats['stdev']:.2f}")
                report_lines.append(f"  Min/Max: {dist_stats['min']:.2f} / {dist_stats['max']:.2f}")
                report_lines.append("")
                
                # Tempo
                time_stats = stats['execution_time']
                report_lines.append(f"TEMPO DE EXECUÇÃO:")
                report_lines.append(f"  Média: {time_stats['mean']:.3f}s")
                report_lines.append(f"  Mediana: {time_stats['median']:.3f}s")
                report_lines.append(f"  Desvio padrão: {time_stats['stdev']:.3f}s")
                report_lines.append(f"  Min/Max: {time_stats['min']:.3f}s / {time_stats['max']:.3f}s")
                report_lines.append("")
                
                # Cobertura
                cov_stats = stats['coverage']
                report_lines.append(f"COBERTURA:")
                report_lines.append(f"  Média: {cov_stats['mean']:.2%}")
                report_lines.append(f"  Mediana: {cov_stats['median']:.2%}")
                report_lines.append(f"  Min/Max: {cov_stats['min']:.2%} / {cov_stats['max']:.2%}")
                
            else:
                report_lines.append(f"FALHA: {stats.get('error', 'Nenhuma execução bem-sucedida')}")
            
            report_lines.append("")
            report_lines.append("")
        
        # Comparação entre algoritmos bem-sucedidos
        if len(successful_algorithms) > 1:
            report_lines.append("COMPARAÇÃO ENTRE ALGORITMOS")
            report_lines.append("=" * 30)
            report_lines.append("")
            
            # Tabela comparativa
            report_lines.append(f"{'Algoritmo':<12} {'Dist.Média':<12} {'Tempo Médio':<12} {'Cobertura':<10} {'Consistência'}")
            report_lines.append("-" * 60)
            
            for algorithm in successful_algorithms:
                stats = results[algorithm]['statistics']
                dist_mean = stats['distance']['mean']
                dist_cv = stats['distance']['stdev'] / dist_mean if dist_mean > 0 else 0  # Coeficiente de variação
                time_mean = stats['execution_time']['mean']
                cov_mean = stats['coverage']['mean']
                
                consistency = "Alta" if dist_cv < 0.1 else "Média" if dist_cv < 0.2 else "Baixa"
                
                report_lines.append(f"{algorithm.upper():<12} {dist_mean:<12.2f} {time_mean:<12.3f} {cov_mean:<10.2%} {consistency}")
            
            report_lines.append("")
            
            # Melhor em cada categoria
            best_distance = min(successful_algorithms, key=lambda a: results[a]['statistics']['distance']['mean'])
            fastest = min(successful_algorithms, key=lambda a: results[a]['statistics']['execution_time']['mean'])
            most_consistent = min(successful_algorithms, key=lambda a: results[a]['statistics']['distance']['stdev'])
            
            report_lines.append("MELHORES EM CADA CATEGORIA:")
            report_lines.append(f"  Melhor distância: {best_distance.upper()}")
            report_lines.append(f"  Mais rápido: {fastest.upper()}")
            report_lines.append(f"  Mais consistente: {most_consistent.upper()}")
            report_lines.append("")
        
        # Recomendações para comparação com ACS
        report_lines.append("DADOS PARA COMPARAÇÃO COM ACS DUPLO")
        report_lines.append("=" * 35)
        report_lines.append("")
        
        for algorithm in successful_algorithms:
            stats = results[algorithm]['statistics']
            report_lines.append(f"{algorithm.upper()}:")
            report_lines.append(f"  Distância média: {stats['distance']['mean']:.2f}")
            report_lines.append(f"  Melhor resultado: {stats['distance']['min']:.2f}")
            report_lines.append(f"  Pior resultado: {stats['distance']['max']:.2f}")
            report_lines.append(f"  Desvio padrão: {stats['distance']['stdev']:.2f}")
            report_lines.append(f"  Tempo médio: {stats['execution_time']['mean']:.3f}s")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Salvar em arquivo se especificado
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"\nRelatório salvo em: {output_path}")
        
        return report_text


def main():
    """
    Função principal para executar o benchmark.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark de algoritmos para comparação com ACS')
    parser.add_argument('--graph', '-g', default='graphml/grafo.graphml',
                       help='Arquivo do grafo para teste')
    parser.add_argument('--runs', '-r', type=int, default=10,
                       help='Número de execuções por algoritmo')
    parser.add_argument('--algorithms', '-a', nargs='+',
                       choices=['aco', 'brute_force', 'greedy', 'all'],
                       default=['all'],
                       help='Algoritmos para testar')
    parser.add_argument('--output', '-o', default='output/benchmark_results',
                       help='Prefixo dos arquivos de saída')
    
    args = parser.parse_args()
    
    # Expandir 'all' para todos os algoritmos
    if 'all' in args.algorithms:
        algorithms = ['greedy', 'aco', 'brute_force']
    else:
        algorithms = args.algorithms
    
    print("BENCHMARK DE ALGORITMOS PARA COMPARAÇÃO COM ACS")
    print("=" * 50)
    
    # Criar benchmark
    benchmark = AlgorithmBenchmark(args.graph)
    
    # Carregar grafo
    if not benchmark.load_graph():
        print("Erro ao carregar grafo. Encerrando.")
        return
    
    # Executar benchmark
    results = benchmark.run_benchmark(algorithms, args.runs)
    
    # Gerar relatório
    report = benchmark.generate_comparison_report(
        results,
        output_file=f"{args.output}_report.txt"
    )
    
    # Salvar dados brutos em JSON
    json_output = {
        'graph_file': args.graph,
        'test_config': {
            'n_runs': args.runs,
            'algorithms': algorithms,
            'stops_to_visit': benchmark.stops_to_visit,
            'start_node': benchmark.start_node,
            'exit_node': benchmark.exit_node
        },
        'results': {}
    }
    
    # Preparar dados para JSON (remover objetos não serializáveis)
    for algorithm, data in results.items():
        json_output['results'][algorithm] = {
            'statistics': data['statistics'],
            'config_used': data['config_used'],
            'successful_runs': [
                {k: v for k, v in run.items() if k != 'route'}  # Remover rota para economizar espaço
                for run in data['raw_results'] if run.get('success', False)
            ]
        }
    
    json_file = f"{args.output}_data.json"
    json_path = Path(json_file)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"Dados brutos salvos em: {json_path}")
    
    # Mostrar resumo no terminal
    print("\n" + "=" * 60)
    print("RESUMO DO BENCHMARK")
    print("=" * 60)
    print(report)
    
    print("\nBenchmark concluído! Use os dados gerados para comparar com seu ACS duplo.")


if __name__ == "__main__":
    main()