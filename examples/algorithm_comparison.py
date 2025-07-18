# examples/algorithm_comparison.py

"""
Script para comparar diferentes algoritmos de otimização de rotas.

Compara ACO simples, força bruta e algoritmo guloso em termos de:
- Qualidade da solução (distância total)
- Tempo de execução
- Cobertura dos pontos de interesse
"""

import sys
import os
import time
import json
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import networkx as nx
from rota_aco.aco.simple_aco import SimpleACO
from rota_aco.aco.brute_force import BruteForceOptimizer, GreedyOptimizer
from rota_aco.graph.loader import GraphLoader


def load_test_graph(graph_file: str = "graphml/pequeno.graphml"):
    """
    Carrega um grafo de teste.
    """
    try:
        loader = GraphLoader()
        graph_path = Path(__file__).parent.parent / graph_file
        
        if not graph_path.exists():
            print(f"Arquivo de grafo não encontrado: {graph_path}")
            return None, None, None, None, None
        
        graph, meta_edges = loader.load_graph(str(graph_path))
        
        # Definir pontos de teste
        nodes = list(graph.nodes())
        if len(nodes) < 5:
            print("Grafo muito pequeno para teste significativo")
            return None, None, None, None, None
        
        start_node = nodes[0]
        exit_node = nodes[-1]
        stops_to_visit = nodes[1:min(6, len(nodes)-1)]  # Máximo 5 pontos para evitar explosão combinatória
        
        print(f"Grafo carregado: {len(nodes)} nós, {len(graph.edges())} arestas")
        print(f"Pontos a visitar: {len(stops_to_visit)}")
        
        return graph, meta_edges, stops_to_visit, start_node, exit_node
        
    except Exception as e:
        print(f"Erro ao carregar grafo: {e}")
        return None, None, None, None, None


def run_simple_aco(graph, meta_edges, stops_to_visit, start_node, exit_node, verbose=False):
    """
    Executa o ACO simples.
    """
    print("\n=== Executando ACO Simples ===")
    
    aco = SimpleACO(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q_param=1.0
    )
    
    start_time = time.time()
    best_route, best_distance, stats = aco.run(
        n_ants=10,
        n_iterations=50,
        verbose=verbose
    )
    execution_time = time.time() - start_time
    
    stats['execution_time'] = execution_time
    stats['algorithm'] = 'ACO Simples'
    
    print(f"Melhor distância: {best_distance:.2f}")
    print(f"Cobertura: {stats['coverage']:.2%}")
    print(f"Tempo de execução: {execution_time:.2f}s")
    
    return best_route, best_distance, stats


def run_brute_force(graph, meta_edges, stops_to_visit, start_node, exit_node, verbose=False):
    """
    Executa força bruta (com limite de tempo para evitar execução muito longa).
    """
    print("\n=== Executando Força Bruta ===")
    
    optimizer = BruteForceOptimizer(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node,
        max_permutations=5000  # Limite para evitar execução muito longa
    )
    
    # Usar versão com limite de tempo se há muitos pontos
    if len(stops_to_visit) > 6:
        best_route, best_distance, stats = optimizer.run_limited(
            max_time_seconds=30.0,
            verbose=verbose
        )
    else:
        best_route, best_distance, stats = optimizer.run(verbose=verbose)
    
    stats['algorithm'] = 'Força Bruta'
    
    print(f"Melhor distância: {best_distance:.2f}")
    print(f"Cobertura: {stats['coverage']:.2%}")
    print(f"Rotas testadas: {stats['routes_tested']}")
    print(f"Tempo de execução: {stats['execution_time']:.2f}s")
    
    return best_route, best_distance, stats


def run_greedy(graph, meta_edges, stops_to_visit, start_node, exit_node, verbose=False):
    """
    Executa algoritmo guloso.
    """
    print("\n=== Executando Algoritmo Guloso ===")
    
    optimizer = GreedyOptimizer(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node
    )
    
    best_route, best_distance, stats = optimizer.run(verbose=verbose)
    
    print(f"Distância total: {best_distance:.2f}")
    print(f"Cobertura: {stats['coverage']:.2%}")
    print(f"Tempo de execução: {stats['execution_time']:.4f}s")
    
    return best_route, best_distance, stats


def compare_algorithms(graph_file: str = "graphml/pequeno.graphml", verbose: bool = False):
    """
    Compara todos os algoritmos implementados.
    """
    print("=== COMPARAÇÃO DE ALGORITMOS DE OTIMIZAÇÃO ===")
    
    # Carregar grafo
    graph, meta_edges, stops_to_visit, start_node, exit_node = load_test_graph(graph_file)
    
    if graph is None:
        print("Não foi possível carregar o grafo de teste.")
        return
    
    results = {}
    
    # Executar algoritmos
    try:
        # ACO Simples
        route_aco, dist_aco, stats_aco = run_simple_aco(
            graph, meta_edges, stops_to_visit, start_node, exit_node, verbose
        )
        results['aco'] = {
            'route': route_aco,
            'distance': dist_aco,
            'stats': stats_aco
        }
    except Exception as e:
        print(f"Erro no ACO: {e}")
        results['aco'] = None
    
    try:
        # Algoritmo Guloso
        route_greedy, dist_greedy, stats_greedy = run_greedy(
            graph, meta_edges, stops_to_visit, start_node, exit_node, verbose
        )
        results['greedy'] = {
            'route': route_greedy,
            'distance': dist_greedy,
            'stats': stats_greedy
        }
    except Exception as e:
        print(f"Erro no algoritmo guloso: {e}")
        results['greedy'] = None
    
    try:
        # Força Bruta (apenas se não há muitos pontos)
        if len(stops_to_visit) <= 7:  # Limite para evitar explosão combinatória
            route_bf, dist_bf, stats_bf = run_brute_force(
                graph, meta_edges, stops_to_visit, start_node, exit_node, verbose
            )
            results['brute_force'] = {
                'route': route_bf,
                'distance': dist_bf,
                'stats': stats_bf
            }
        else:
            print(f"\n=== Força Bruta Pulada ===")
            print(f"Muitos pontos ({len(stops_to_visit)}) - força bruta seria muito lenta")
            results['brute_force'] = None
    except Exception as e:
        print(f"Erro na força bruta: {e}")
        results['brute_force'] = None
    
    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("Nenhum algoritmo executou com sucesso.")
        return
    
    # Tabela de resultados
    print(f"{'Algoritmo':<15} {'Distância':<12} {'Cobertura':<10} {'Tempo (s)':<10}")
    print("-" * 50)
    
    for name, result in valid_results.items():
        stats = result['stats']
        algo_name = stats.get('algorithm', name.upper())
        distance = result['distance']
        coverage = stats['coverage']
        exec_time = stats['execution_time']
        
        print(f"{algo_name:<15} {distance:<12.2f} {coverage:<10.2%} {exec_time:<10.4f}")
    
    # Encontrar melhor solução
    best_algo = min(valid_results.items(), key=lambda x: x[1]['distance'])
    print(f"\nMelhor solução: {best_algo[1]['stats']['algorithm']} "
          f"(distância: {best_algo[1]['distance']:.2f})")
    
    # Salvar resultados
    output_file = Path(__file__).parent.parent / "output" / "algorithm_comparison.json"
    output_file.parent.mkdir(exist_ok=True)
    
    # Preparar dados para JSON (remover objetos não serializáveis)
    json_results = {}
    for name, result in valid_results.items():
        if result:
            json_results[name] = {
                'distance': result['distance'],
                'stats': {k: v for k, v in result['stats'].items() 
                         if isinstance(v, (int, float, str, bool, list))}
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados salvos em: {output_file}")
    
    return results


def main():
    """
    Função principal para execução do script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparar algoritmos de otimização de rotas')
    parser.add_argument('--graph', '-g', default='graphml/pequeno.graphml',
                       help='Arquivo do grafo para teste')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostrar informações detalhadas durante execução')
    
    args = parser.parse_args()
    
    results = compare_algorithms(args.graph, args.verbose)
    
    if results:
        print("\nComparação concluída com sucesso!")
    else:
        print("\nErro na execução da comparação.")


if __name__ == "__main__":
    main()