# examples/acs_benchmark.py

"""
Script para executar o ACS duplo múltiplas vezes e calcular estatísticas.

Este script executa o sistema ACS (ACS-TIME + ACS-VEHICLE) N vezes
para coletar dados estatísticos de performance.
"""

import sys
import os
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.cli.run import main as run_acs
from rota_aco.graph.loader import GraphLoader


class ACSBenchmark:
    """
    Classe para executar benchmark do ACS duplo com múltiplas execuções.
    """
    
    def __init__(self, graph_file: str, output_base_dir: str = "output/acs_benchmark"):
        self.graph_file = graph_file
        self.output_base_dir = output_base_dir
        self.results = []
        
        # Criar diretório base
        os.makedirs(output_base_dir, exist_ok=True)
    
    def run_single_acs_execution(self, run_id: int, config: Dict) -> Dict:
        """
        Executa uma única rodada do ACS.
        
        Args:
            run_id: ID da execução
            config: Configuração do ACS
            
        Returns:
            Dicionário com resultados da execução
        """
        print(f"  Execução {run_id}...", end=" ")
        
        # Criar diretório específico para esta execução
        run_output_dir = os.path.join(self.output_base_dir, f"run_{run_id:02d}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Preparar argumentos para o ACS
            import argparse
            
            # Simular argumentos da linha de comando
            args = argparse.Namespace()
            args.graph = self.graph_file
            args.output = run_output_dir
            args.n_ants = config.get('n_ants', 10)
            args.n_iterations = config.get('n_iterations', 50)
            args.capacity = config.get('capacity', 70)
            args.w_c = config.get('w_c', 10.0)
            args.w_r = config.get('w_r', 1.0)
            args.w_d = config.get('w_d', 0.5)
            args.verbose = False
            args.generate_metrics = True
            args.generate_convergence = True
            # Adicionar coordenadas padrão (você pode ajustar conforme necessário)
            args.start_lat = -15.7801
            args.start_lon = -47.9292
            args.exit_lat = -15.7801
            args.exit_lon = -47.9292
            args.precision = 100
            args.manual_opposites = None
            
            # Executar ACS (capturar stdout para evitar poluição)
            import io
            import contextlib
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Executar o ACS principal
                result = run_acs_with_args(args)
                execution_time = time.time() - start_time
                
                # Restaurar stdout
                sys.stdout = old_stdout
                
                # Extrair métricas dos arquivos gerados
                metrics = self._extract_metrics_from_output(run_output_dir)
                
                result_data = {
                    'run_id': run_id,
                    'success': True,
                    'execution_time': execution_time,
                    'output_dir': run_output_dir,
                    'metrics': metrics,
                    'config_used': config
                }
                
                print(f"OK (tempo: {execution_time:.2f}s, qualidade: {metrics.get('final_quality', 'N/A')})")
                
                return result_data
                
            except Exception as e:
                sys.stdout = old_stdout
                raise e
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            result_data = {
                'run_id': run_id,
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'output_dir': run_output_dir,
                'config_used': config
            }
            
            print(f"ERRO: {str(e)[:50]}...")
            
            return result_data
    
    def _extract_metrics_from_output(self, output_dir: str) -> Dict:
        """
        Extrai métricas dos arquivos de saída gerados pelo ACS.
        """
        metrics = {}
        
        try:
            # Procurar por arquivos de métricas
            metrics_dir = os.path.join(output_dir, "metrics")
            
            if os.path.exists(metrics_dir):
                # Procurar arquivo de convergência
                convergence_file = os.path.join(metrics_dir, "convergence_analysis.json")
                if os.path.exists(convergence_file):
                    with open(convergence_file, 'r', encoding='utf-8') as f:
                        convergence_data = json.load(f)
                        
                    metrics.update({
                        'final_quality': convergence_data.get('convergence_summary', {}).get('final_quality_q', 0),
                        'initial_quality': convergence_data.get('convergence_summary', {}).get('initial_quality_q', 0),
                        'improvement_percentage': convergence_data.get('convergence_summary', {}).get('improvement_percentage', 0),
                        'total_iterations': convergence_data.get('total_iterations', 0),
                        'acs_time_final': convergence_data.get('acs_time_final', {}),
                        'acs_vehicle_final': convergence_data.get('acs_vehicle_final', {}),
                        'invalid_routes_summary': convergence_data.get('invalid_routes_summary', {})
                    })
                
                # Procurar outros arquivos de métricas
                for file in os.listdir(metrics_dir):
                    if file.endswith('.json') and 'report' in file:
                        try:
                            with open(os.path.join(metrics_dir, file), 'r', encoding='utf-8') as f:
                                additional_data = json.load(f)
                                # Adicionar dados relevantes
                                if 'execution_summary' in additional_data:
                                    metrics['execution_summary'] = additional_data['execution_summary']
                        except:
                            continue
            
        except Exception as e:
            print(f"Aviso: Erro ao extrair métricas de {output_dir}: {e}")
        
        return metrics
    
    def run_benchmark(self, n_runs: int = 10, config: Dict = None) -> Dict:
        """
        Executa benchmark completo do ACS.
        
        Args:
            n_runs: Número de execuções
            config: Configuração do ACS
            
        Returns:
            Dicionário com resultados estatísticos
        """
        if config is None:
            config = {
                'n_ants': 10,
                'n_iterations': 50,
                'capacity': 70,
                'w_c': 10.0,
                'w_r': 1.0,
                'w_d': 0.5
            }
        
        print(f"BENCHMARK ACS DUPLO - {n_runs} EXECUÇÕES")
        print("=" * 50)
        print(f"Grafo: {self.graph_file}")
        print(f"Configuração: {config}")
        print()
        
        results = []
        successful_runs = 0
        
        for run_id in range(1, n_runs + 1):
            result = self.run_single_acs_execution(run_id, config)
            results.append(result)
            
            if result['success']:
                successful_runs += 1
        
        # Calcular estatísticas
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            # Extrair dados para análise estatística
            execution_times = [r['execution_time'] for r in successful_results]
            final_qualities = [r['metrics'].get('final_quality', 0) for r in successful_results if r['metrics'].get('final_quality', 0) > 0]
            improvements = [r['metrics'].get('improvement_percentage', 0) for r in successful_results]
            
            # Métricas ACS-TIME
            time_distances = [r['metrics'].get('acs_time_final', {}).get('distance', 0) for r in successful_results]
            time_routes = [r['metrics'].get('acs_time_final', {}).get('routes', 0) for r in successful_results]
            time_coverage = [r['metrics'].get('acs_time_final', {}).get('coverage', 0) for r in successful_results]
            
            # Métricas ACS-VEHICLE
            vehicle_distances = [r['metrics'].get('acs_vehicle_final', {}).get('distance', 0) for r in successful_results]
            vehicle_routes = [r['metrics'].get('acs_vehicle_final', {}).get('routes', 0) for r in successful_results]
            vehicle_coverage = [r['metrics'].get('acs_vehicle_final', {}).get('coverage', 0) for r in successful_results]
            
            # Calcular estatísticas
            stats = {
                'total_runs': n_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / n_runs,
                'execution_time': self._calculate_stats(execution_times),
                'final_quality': self._calculate_stats(final_qualities) if final_qualities else {},
                'improvement_percentage': self._calculate_stats(improvements),
                'acs_time': {
                    'distance': self._calculate_stats([d for d in time_distances if d > 0]),
                    'routes': self._calculate_stats([r for r in time_routes if r > 0]),
                    'coverage': self._calculate_stats([c for c in time_coverage if c > 0])
                },
                'acs_vehicle': {
                    'distance': self._calculate_stats([d for d in vehicle_distances if d > 0]),
                    'routes': self._calculate_stats([r for r in vehicle_routes if r > 0]),
                    'coverage': self._calculate_stats([c for c in vehicle_coverage if c > 0])
                }
            }
            
            print(f"\n" + "=" * 50)
            print("RESULTADOS ESTATÍSTICOS")
            print("=" * 50)
            print(f"Taxa de sucesso: {stats['success_rate']:.1%} ({successful_runs}/{n_runs})")
            print()
            
            print("TEMPO DE EXECUÇÃO:")
            self._print_stats(stats['execution_time'], "s")
            print()
            
            if final_qualities:
                print("QUALIDADE Q FINAL:")
                self._print_stats(stats['final_quality'])
                print()
            
            print("MELHORIA PERCENTUAL:")
            self._print_stats(stats['improvement_percentage'], "%")
            print()
            
            print("ACS-TIME - DISTÂNCIA:")
            self._print_stats(stats['acs_time']['distance'], "m")
            print()
            
            print("ACS-VEHICLE - DISTÂNCIA:")
            self._print_stats(stats['acs_vehicle']['distance'], "m")
            print()
            
        else:
            stats = {
                'total_runs': n_runs,
                'successful_runs': 0,
                'success_rate': 0.0,
                'error': 'Nenhuma execução bem-sucedida'
            }
            
            print(f"\nFALHA: Nenhuma execução bem-sucedida")
        
        # Salvar resultados
        benchmark_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'graph_file': self.graph_file,
                'n_runs': n_runs,
                'config': config
            },
            'statistics': stats,
            'raw_results': results
        }
        
        # Salvar em JSON
        results_file = os.path.join(self.output_base_dir, "benchmark_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados salvos em: {results_file}")
        
        return benchmark_results
    
    def _calculate_stats(self, values: List[float]) -> Dict:
        """
        Calcula estatísticas descritivas para uma lista de valores.
        """
        if not values:
            return {}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'variance': statistics.variance(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    def _print_stats(self, stats_dict: Dict, unit: str = ""):
        """
        Imprime estatísticas de forma formatada.
        """
        if not stats_dict:
            print("  Dados insuficientes")
            return
        
        print(f"  Média: {stats_dict['mean']:.2f}{unit}")
        print(f"  Mediana: {stats_dict['median']:.2f}{unit}")
        print(f"  Desvio padrão: {stats_dict['stdev']:.2f}{unit}")
        print(f"  Variância: {stats_dict['variance']:.2f}{unit}²")
        print(f"  Min/Max: {stats_dict['min']:.2f}{unit} / {stats_dict['max']:.2f}{unit}")


def run_acs_with_args(args):
    """
    Executa o ACS com argumentos específicos.
    """
    # Importar módulos necessários
    import subprocess
    import sys
    
    # Construir comando para executar o ACS
    cmd = [
        sys.executable, 
        "src/rota_aco/cli/run.py",
        "--graph", args.graph,
        "--start-lat", str(args.start_lat),
        "--start-lon", str(args.start_lon),
        "--exit-lat", str(args.exit_lat),
        "--exit-lon", str(args.exit_lon),
        "--ants", str(args.n_ants),
        "--iterations", str(args.n_iterations),
        "--capacity", str(args.capacity),
        "--w-c", str(args.w_c),
        "--w-r", str(args.w_r),
        "--w-d", str(args.w_d),
        "--precision", str(args.precision)
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    if hasattr(args, 'manual_opposites') and args.manual_opposites:
        cmd.extend(["--manual-opposites", args.manual_opposites])
    
    if hasattr(args, 'generate_metrics') and args.generate_metrics:
        cmd.append("--metrics")
    
    if hasattr(args, 'generate_convergence') and args.generate_convergence:
        cmd.append("--convergence-analysis")
    
    # Executar comando
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        raise Exception(f"ACS falhou: {result.stderr}")
    
    return result


def main():
    """
    Função principal para executar o benchmark.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark do ACS duplo')
    parser.add_argument('--graph', '-g', default='graphml/grafo.graphml',
                       help='Arquivo do grafo para teste')
    parser.add_argument('--runs', '-r', type=int, default=10,
                       help='Número de execuções')
    parser.add_argument('--output', '-o', default='output/acs_benchmark',
                       help='Diretório base para saída')
    parser.add_argument('--n-ants', type=int, default=10,
                       help='Número de formigas')
    parser.add_argument('--n-iterations', type=int, default=50,
                       help='Número de iterações')
    parser.add_argument('--capacity', type=int, default=70,
                       help='Capacidade do veículo')
    parser.add_argument('--w-c', type=float, default=10.0,
                       help='Peso da cobertura')
    parser.add_argument('--w-r', type=float, default=1.0,
                       help='Peso do número de rotas')
    parser.add_argument('--w-d', type=float, default=0.5,
                       help='Peso da distância')
    
    args = parser.parse_args()
    
    # Configuração do ACS
    config = {
        'n_ants': args.n_ants,
        'n_iterations': args.n_iterations,
        'capacity': args.capacity,
        'w_c': args.w_c,
        'w_r': args.w_r,
        'w_d': args.w_d
    }
    
    # Criar e executar benchmark
    benchmark = ACSBenchmark(args.graph, args.output)
    results = benchmark.run_benchmark(args.runs, config)
    
    print(f"\nBenchmark concluído!")
    print(f"Resultados disponíveis em: {args.output}")
    
    return results


if __name__ == "__main__":
    main()