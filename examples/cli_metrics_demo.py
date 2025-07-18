# examples/cli_metrics_demo.py

"""
Demonstração via linha de comando para comparar algoritmos com métricas detalhadas.

Este script permite testar os algoritmos com diferentes configurações e
gerar relatórios detalhados de performance.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.aco.simple_aco import SimpleACO
from rota_aco.aco.brute_force import BruteForceOptimizer, GreedyOptimizer
from rota_aco.graph.loader import GraphLoader


def create_complex_test_graph():
    """
    Cria um grafo mais complexo para testes.
    """
    import networkx as nx
    
    graph = nx.DiGraph()
    
    # Criar uma grade 4x4 de nós
    nodes = []
    for i in range(4):
        for j in range(4):
            node = f"N{i}{j}"
            nodes.append(node)
            graph.add_node(node, x=j, y=i)
    
    # Conectar nós adjacentes com pesos variados
    meta_edges = {}
    
    for i in range(4):
        for j in range(4):
            current = f"N{i}{j}"
            
            # Conectar para a direita
            if j < 3:
                right = f"N{i}{j+1}"
                weight = 1.0 + (i * 0.5) + (j * 0.3)  # Pesos variados
                graph.add_edge(current, right, weight=weight)
                graph.add_edge(right, current, weight=weight * 1.2)  # Assimétrico
                meta_edges[(current, right)] = {'time': weight, 'distance': weight}
                meta_edges[(right, current)] = {'time': weight * 1.2, 'distance': weight * 1.2}
            
            # Conectar para baixo
            if i < 3:
                down = f"N{i+1}{j}"
                weight = 1.2 + (i * 0.3) + (j * 0.4)
                graph.add_edge(current, down, weight=weight)
                graph.add_edge(down, current, weight=weight * 1.1)
                meta_edges[(current, down)] = {'time': weight, 'distance': weight}
                meta_edges[(down, current)] = {'time': weight * 1.1, 'distance': weight * 1.1}
            
            # Adicionar algumas conexões diagonais
            if i < 3 and j < 3:
                diag = f"N{i+1}{j+1}"
                weight = 1.8 + (i * 0.2)
                graph.add_edge(current, diag, weight=weight)
                meta_edges[(current, diag)] = {'time': weight, 'distance': weight}
    
    return graph, meta_edges, nodes


def run_algorithm_test(algorithm_name, graph, meta_edges, stops_to_visit, start_node, exit_node, config):
    """
    Executa um algoritmo específico e retorna os resultados.
    """
    print(f"\n=== {algorithm_name.upper()} ===")
    
    start_time = time.time()
    
    if algorithm_name == 'aco':
        algo = SimpleACO(
            graph=graph,
            meta_edges=meta_edges,
            stops_to_visit=stops_to_visit,
            start_node=start_node,
            exit_node=exit_node,
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 2.0),
            rho=config.get('rho', 0.1),
            q_param=config.get('q_param', 1.0)
        )
        
        route, distance, stats = algo.run(
            n_ants=config.get('n_ants', 10),
            n_iterations=config.get('n_iterations', 50),
            verbose=config.get('verbose', False)
        )
        
    elif algorithm_name == 'brute_force':
        algo = BruteForceOptimizer(
            graph=graph,
            meta_edges=meta_edges,
            stops_to_visit=stops_to_visit,
            start_node=start_node,
            exit_node=exit_node,
            max_permutations=config.get('max_permutations', 5000)
        )
        
        if config.get('time_limit'):
            route, distance, stats = algo.run_limited(
                max_time_seconds=config['time_limit'],
                verbose=config.get('verbose', False)
            )
        else:
            route, distance, stats = algo.run(verbose=config.get('verbose', False))
            
    elif algorithm_name == 'greedy':
        algo = GreedyOptimizer(
            graph=graph,
            meta_edges=meta_edges,
            stops_to_visit=stops_to_visit,
            start_node=start_node,
            exit_node=exit_node
        )
        
        route, distance, stats = algo.run(verbose=config.get('verbose', False))
    
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")
    
    execution_time = time.time() - start_time
    
    # Adicionar métricas extras
    stats['execution_time'] = execution_time
    stats['algorithm'] = algorithm_name
    stats['route_nodes'] = len(route) if route else 0
    
    print(f"Distância: {distance:.2f}")
    print(f"Cobertura: {stats.get('coverage', 0):.2%}")
    print(f"Tempo: {execution_time:.4f}s")
    print(f"Nós na rota: {stats['route_nodes']}")
    
    if config.get('show_route', False) and route:
        print(f"Rota: {' -> '.join(map(str, route[:10]))}{'...' if len(route) > 10 else ''}")
    
    return route, distance, stats


def main():
    """
    Função principal com interface de linha de comando.
    """
    parser = argparse.ArgumentParser(description='Comparação de algoritmos de otimização de rotas')
    
    parser.add_argument('--algorithms', '-a', nargs='+', 
                       choices=['aco', 'brute_force', 'greedy', 'all'],
                       default=['all'],
                       help='Algoritmos para executar')
    
    parser.add_argument('--graph-type', '-g', choices=['simple', 'complex'],
                       default='complex',
                       help='Tipo de grafo para teste')
    
    parser.add_argument('--stops', '-s', type=int, default=6,
                       help='Número de pontos para visitar')
    
    parser.add_argument('--aco-ants', type=int, default=10,
                       help='Número de formigas no ACO')
    
    parser.add_argument('--aco-iterations', type=int, default=50,
                       help='Número de iterações do ACO')
    
    parser.add_argument('--bf-time-limit', type=float, default=30.0,
                       help='Limite de tempo para força bruta (segundos)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostrar informações detalhadas')
    
    parser.add_argument('--show-routes', action='store_true',
                       help='Mostrar as rotas encontradas')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Arquivo para salvar resultados JSON')
    
    args = parser.parse_args()
    
    # Expandir 'all' para todos os algoritmos
    if 'all' in args.algorithms:
        algorithms = ['greedy', 'aco', 'brute_force']
    else:
        algorithms = args.algorithms
    
    print("COMPARAÇÃO DE ALGORITMOS - MÉTRICAS DETALHADAS")
    print("=" * 60)
    
    # Criar grafo de teste
    if args.graph_type == 'simple':
        loader = GraphLoader()
        graph, meta_edges = loader.create_simple_test_graph()
        all_nodes = list(graph.nodes())
        start_node = all_nodes[0]
        exit_node = all_nodes[-1]
        available_stops = all_nodes[1:-1]
    else:
        graph, meta_edges, all_nodes = create_complex_test_graph()
        start_node = "N00"  # Canto superior esquerdo
        exit_node = "N33"   # Canto inferior direito
        available_stops = [node for node in all_nodes if node not in [start_node, exit_node]]
    
    # Selecionar pontos para visitar
    import random
    random.seed(42)  # Para resultados reproduzíveis
    stops_to_visit = random.sample(available_stops, min(args.stops, len(available_stops)))
    
    print(f"Grafo: {len(graph.nodes())} nós, {len(graph.edges())} arestas")
    print(f"Início: {start_node}, Fim: {exit_node}")
    print(f"Pontos a visitar ({len(stops_to_visit)}): {stops_to_visit}")
    
    # Configurações dos algoritmos
    configs = {
        'aco': {
            'n_ants': args.aco_ants,
            'n_iterations': args.aco_iterations,
            'verbose': args.verbose,
            'show_route': args.show_routes
        },
        'brute_force': {
            'time_limit': args.bf_time_limit if len(stops_to_visit) > 6 else None,
            'verbose': args.verbose,
            'show_route': args.show_routes
        },
        'greedy': {
            'verbose': args.verbose,
            'show_route': args.show_routes
        }
    }
    
    # Executar algoritmos
    results = {}
    
    for algorithm in algorithms:
        try:
            route, distance, stats = run_algorithm_test(
                algorithm, graph, meta_edges, stops_to_visit, 
                start_node, exit_node, configs[algorithm]
            )
            
            results[algorithm] = {
                'route': route,
                'distance': distance,
                'stats': stats
            }
            
        except Exception as e:
            print(f"Erro no algoritmo {algorithm}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Resumo comparativo
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("RESUMO COMPARATIVO")
        print("=" * 60)
        
        print(f"{'Algoritmo':<12} {'Distância':<12} {'Tempo (s)':<12} {'Cobertura':<12} {'Eficiência'}")
        print("-" * 65)
        
        # Calcular eficiência (distância/tempo)
        for name, result in results.items():
            distance = result['distance']
            exec_time = result['stats']['execution_time']
            coverage = result['stats'].get('coverage', 0)
            efficiency = distance / exec_time if exec_time > 0 else float('inf')
            
            print(f"{name.upper():<12} {distance:<12.2f} {exec_time:<12.4f} {coverage:<12.2%} {efficiency:<12.1f}")
    
    # Salvar resultados se solicitado
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar dados para JSON
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                'distance': result['distance'],
                'stats': {k: v for k, v in result['stats'].items() 
                         if isinstance(v, (int, float, str, bool, list))}
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'graph_type': args.graph_type,
                    'stops_count': len(stops_to_visit),
                    'stops_to_visit': stops_to_visit,
                    'start_node': start_node,
                    'exit_node': exit_node
                },
                'results': json_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResultados salvos em: {output_path}")
    
    print("\nTeste concluído!")
    return results


if __name__ == "__main__":
    main()