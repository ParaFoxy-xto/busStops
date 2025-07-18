# examples/simple_comparison_demo.py

"""
Demonstração simples da comparação entre ACO, força bruta e algoritmo guloso.

Este script cria um grafo de teste simples e compara os três algoritmos.
"""

import sys
import time
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.aco.simple_aco import SimpleACO
from rota_aco.aco.brute_force import BruteForceOptimizer, GreedyOptimizer
from rota_aco.graph.loader import GraphLoader


def create_demo_scenario():
    """
    Cria um cenário de demonstração com grafo simples.
    """
    loader = GraphLoader()
    graph, meta_edges = loader.create_simple_test_graph()
    
    # Definir cenário de teste
    start_node = 'A'
    exit_node = 'F'
    stops_to_visit = ['B', 'C', 'D', 'E']  # 4 pontos para visitar
    
    print("=== CENÁRIO DE DEMONSTRAÇÃO ===")
    print(f"Grafo: {len(graph.nodes())} nós, {len(graph.edges())} arestas")
    print(f"Início: {start_node}")
    print(f"Fim: {exit_node}")
    print(f"Pontos a visitar: {stops_to_visit}")
    print(f"Nós do grafo: {list(graph.nodes())}")
    
    return graph, meta_edges, stops_to_visit, start_node, exit_node


def run_comparison_demo():
    """
    Executa a demonstração de comparação dos algoritmos.
    """
    print("COMPARAÇÃO DE ALGORITMOS - DEMONSTRAÇÃO")
    print("=" * 50)
    
    # Criar cenário
    graph, meta_edges, stops_to_visit, start_node, exit_node = create_demo_scenario()
    
    results = {}
    
    # 1. Algoritmo Guloso (mais rápido)
    print("\n1. ALGORITMO GULOSO")
    print("-" * 30)
    
    greedy = GreedyOptimizer(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node
    )
    
    start_time = time.time()
    route_greedy, dist_greedy, stats_greedy = greedy.run(verbose=True)
    greedy_time = time.time() - start_time
    
    results['greedy'] = {
        'route': route_greedy,
        'distance': dist_greedy,
        'time': greedy_time,
        'coverage': stats_greedy['coverage']
    }
    
    print(f"Rota encontrada: {' -> '.join(map(str, route_greedy))}")
    
    # 2. Força Bruta (solução ótima)
    print("\n2. FORÇA BRUTA (SOLUÇÃO ÓTIMA)")
    print("-" * 30)
    
    brute_force = BruteForceOptimizer(
        graph=graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node
    )
    
    start_time = time.time()
    route_bf, dist_bf, stats_bf = brute_force.run(verbose=True)
    bf_time = time.time() - start_time
    
    results['brute_force'] = {
        'route': route_bf,
        'distance': dist_bf,
        'time': bf_time,
        'coverage': stats_bf['coverage']
    }
    
    print(f"Rota ótima: {' -> '.join(map(str, route_bf))}")
    
    # 3. ACO Simples (heurística)
    print("\n3. ACO SIMPLES (HEURÍSTICA)")
    print("-" * 30)
    
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
    route_aco, dist_aco, stats_aco = aco.run(n_ants=10, n_iterations=30, verbose=True)
    aco_time = time.time() - start_time
    
    results['aco'] = {
        'route': route_aco,
        'distance': dist_aco,
        'time': aco_time,
        'coverage': stats_aco['coverage']
    }
    
    print(f"Rota ACO: {' -> '.join(map(str, route_aco))}")
    
    # Resumo comparativo
    print("\n" + "=" * 60)
    print("RESUMO COMPARATIVO")
    print("=" * 60)
    
    print(f"{'Algoritmo':<15} {'Distância':<12} {'Tempo (s)':<12} {'Cobertura':<10} {'Qualidade'}")
    print("-" * 65)
    
    # Calcular qualidade relativa (menor distância = melhor)
    best_distance = min(r['distance'] for r in results.values() if r['distance'] > 0)
    
    for name, result in results.items():
        algo_names = {
            'greedy': 'Guloso',
            'brute_force': 'Força Bruta',
            'aco': 'ACO Simples'
        }
        
        distance = result['distance']
        exec_time = result['time']
        coverage = result['coverage']
        
        # Qualidade: quanto mais próximo de 1.0, melhor
        quality = best_distance / distance if distance > 0 else 0.0
        quality_str = f"{quality:.3f}"
        
        if quality >= 0.99:
            quality_str += " (ÓTIMO)"
        elif quality >= 0.95:
            quality_str += " (MUITO BOM)"
        elif quality >= 0.90:
            quality_str += " (BOM)"
        
        print(f"{algo_names[name]:<15} {distance:<12.2f} {exec_time:<12.4f} {coverage:<10.2%} {quality_str}")
    
    # Análise dos resultados
    print("\n" + "=" * 60)
    print("ANÁLISE DOS RESULTADOS")
    print("=" * 60)
    
    optimal_distance = results['brute_force']['distance']
    
    print(f"Solução ótima (força bruta): {optimal_distance:.2f}")
    print()
    
    for name, result in results.items():
        if name == 'brute_force':
            continue
            
        algo_names = {
            'greedy': 'Algoritmo Guloso',
            'aco': 'ACO Simples'
        }
        
        distance = result['distance']
        gap = ((distance - optimal_distance) / optimal_distance) * 100
        
        print(f"{algo_names[name]}:")
        print(f"  - Distância: {distance:.2f}")
        print(f"  - Gap do ótimo: {gap:.1f}%")
        print(f"  - Tempo: {result['time']:.4f}s")
        
        if gap <= 5:
            print("  - Avaliação: EXCELENTE (muito próximo do ótimo)")
        elif gap <= 15:
            print("  - Avaliação: BOM (próximo do ótimo)")
        elif gap <= 30:
            print("  - Avaliação: ACEITÁVEL")
        else:
            print("  - Avaliação: PODE MELHORAR")
        print()
    
    # Recomendações
    print("RECOMENDAÇÕES:")
    print("- Força bruta: Use apenas para problemas pequenos (< 8 pontos)")
    print("- Algoritmo guloso: Rápido e simples, boa primeira aproximação")
    print("- ACO: Bom equilíbrio entre qualidade e tempo, ajuste parâmetros conforme necessário")
    
    return results


def main():
    """
    Função principal.
    """
    try:
        results = run_comparison_demo()
        print("\nDemonstração concluída com sucesso!")
        return results
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()