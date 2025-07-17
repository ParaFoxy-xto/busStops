# src/rota_aco/cli/run.py

import argparse
import os
import sys
import traceback

# --- Importações Corrigidas e Completas ---
from rota_aco.data.preprocess import load_graph, get_bus_stops
from rota_aco.graph.build_meta import build_meta_graph, expand_meta_route
from rota_aco.graph.utils import find_nearest_node
from rota_aco.aco.controller import ACSController
from rota_aco.viz.matplotlib_viz import plot_multiple_meta_routes, plot_multiple_routes, plot_meta_graph
from rota_aco.viz.folium_viz import visualize_routes_folium
from rota_aco.reporting.report_generator import generate_convergence_plots, generate_final_report, generate_comparison_csv

def setup_arg_parser() -> argparse.ArgumentParser:
    """Configura e retorna o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de otimização de rotas com ACS multi-colônia."
    )
    # ... (o resto da função setup_arg_parser da resposta anterior está correto)
    g_problem = parser.add_argument_group('Parâmetros do Problema e Grafo')
    g_problem.add_argument("--graph", required=True, help="Caminho para o arquivo GraphML do grafo.")
    g_problem.add_argument("--start-lat", type=float, required=True, help="Latitude do ponto de partida.")
    g_problem.add_argument("--start-lon", type=float, required=True, help="Longitude do ponto de partida.")
    g_problem.add_argument("--exit-lat", type=float, required=True, help="Latitude do ponto de chegada.")
    g_problem.add_argument("--exit-lon", type=float, required=True, help="Longitude do ponto de chegada.")
    g_problem.add_argument("--precision", type=int, default=5, help="Precisão geoespacial para agrupar paradas.")
    g_problem.add_argument("--capacity", type=int, default=70, help="Capacidade de passageiros do veículo.")
    g_problem.add_argument("--max-route-length", type=int, default=100, help="Número máximo de nós em uma única rota.")
    g_problem.add_argument("--max-route-attempts", type=int, default=10, help="Tentativas de uma formiga para criar rotas.")

    g_aco = parser.add_argument_group('Parâmetros do Algoritmo ACO')
    g_aco.add_argument("--ants", type=int, default=20, help="Número de formigas por colônia.")
    g_aco.add_argument("--iterations", type=int, default=10, help="Número máximo de iterações.")
    g_aco.add_argument("--alpha", type=float, default=1.0, help="Influência do feromônio (alfa).")
    g_aco.add_argument("--beta", type=float, default=2.0, help="Influência da heurística (beta).")
    g_aco.add_argument("--rho", type=float, default=0.1, help="Taxa de evaporação do feromônio (rho).")
    g_aco.add_argument("--Q", type=float, default=1.0, help="Fator de reforço de feromônio (Q).")

    g_quality = parser.add_argument_group('Pesos da Função de Qualidade')
    g_quality.add_argument("--w-c", type=float, default=10.0, help="Peso da cobertura de paradas (w_c).")
    g_quality.add_argument("--w-r", type=float, default=1.0, help="Peso do número de rotas (w_r).")
    g_quality.add_argument("--w-d", type=float, default=0.5, help="Peso da distância total (w_d).")

    g_output = parser.add_argument_group('Saída e Visualização')
    g_output.add_argument("--output", help="Nome do arquivo para a visualização da rota final (ex: rota_final.png).")
    g_output.add_argument("--meta-output", help="Nome do arquivo para a visualização do meta-grafo (ex: meta_grafo.png).")
    g_output.add_argument("--folium", action="store_true", help="Gerar visualização interativa com Folium.")
    g_output.add_argument("--verbose", action="store_true", help="Ativar logs detalhados durante a execução.")
    g_problem.add_argument("--manual-opposites",help="Caminho para um arquivo JSON opcional definindo pares de opostos manualmente.")
    
    return parser

def main():
    """Função principal que executa o pipeline completo."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- 1. Carregamento e Preparação dos Dados ---
    print("--- 1. Carregando e Preparando Dados ---")
    try:
        graph = load_graph(args.graph)
        bus_stops_nodes = get_bus_stops(graph)
        if not bus_stops_nodes:
            print("[ERRO] Nenhuma parada de ônibus ('bus_stop=true') encontrada no grafo.")
            sys.exit(1)

        start_node = find_nearest_node(graph, (args.start_lat, args.start_lon), list(graph.nodes()))
        exit_node = find_nearest_node(graph, (args.exit_lat, args.exit_lon), list(graph.nodes()))
        print(f"Nó de partida mais próximo: {start_node}")
        print(f"Nó de chegada mais próximo: {exit_node}")
    except Exception as e:
        print(f"[ERRO] Falha na etapa de carregamento de dados: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Construção e Visualização do Meta-Grafo ---
    print("\n--- 2. Construindo o Meta-Grafo ---")
    try:
        meta_graph, meta_edges, representatives, all_opposites, _, _ = build_meta_graph(
            graph=graph,
            bus_stops=bus_stops_nodes,
            start_node=start_node,
            exit_node=exit_node,
            precision=args.precision,
            manual_opposites_path=args.manual_opposites,
            verbose=args.verbose
        )
        if not meta_graph.nodes() or not meta_graph.edges():
            print("[ERRO] A construção resultou em um meta-grafo vazio.")
            sys.exit(1)

        # --- NOVA SEÇÃO: VISUALIZAÇÃO IMEDIATA DO META-GRAFO PARA DEPURAÇÃO ---
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        if args.meta_output:
            meta_graph_path = os.path.join(output_dir, args.meta_output)
            print(f"\n[DEPURAÇÃO] Salvando visualização do meta-grafo em: '{meta_graph_path}'")
            # Usa a função plot_meta_graph, que agora só precisa do meta_graph
            plot_meta_graph(
                meta_graph,
                output_path=meta_graph_path,
                start_node=start_node,
                exit_node=exit_node,
                show_labels=True
            )
        # --- FIM DA NOVA SEÇÃO ---

    except Exception as e:
        print(f"[ERRO] Falha na construção do meta-grafo: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Execução do Otimizador ACS ---
    print("\n--- 3. Executando o Otimizador ACS Multi-Colônia ---")
    
    stops_to_visit = [stop for stop in representatives if stop in meta_graph.nodes() and stop not in {start_node, exit_node}]
    
    aco_params = {'alpha': args.alpha, 'beta': args.beta, 'rho': args.rho, 'Q': args.Q}
    problem_params = {'capacity': args.capacity, 'max_route_length': args.max_route_length, 'max_route_attempts': args.max_route_attempts}
    quality_weights = {'w_c': args.w_c, 'w_r': args.w_r, 'w_d': args.w_d}
    
    try:
        controller = ACSController(
            graph=meta_graph,
            meta_edges=meta_edges,
            stops_to_visit=stops_to_visit,
            start_node=start_node,
            exit_node=exit_node,
            opposites=all_opposites,
            aco_params=aco_params,
            problem_params=problem_params
        )
        
        best_routes, total_dist, num_routes, coverage = controller.run(
            n_ants=args.ants,
            n_iterations=args.iterations,
            quality_weights=quality_weights,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"[ERRO] Falha durante a execução do ACSController: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 4. Relatórios e Visualização Pós-Otimização ---
    print("\n--- 4. Resultados Finais e Relatórios ---")
    if not best_routes:
        print("\nNenhuma rota válida foi encontrada pela otimização.")
        sys.exit(0)

    # Reúne todos os dados para os relatórios
    final_report_data = {
        'params': {
            'ACO_Parameters': aco_params,
            'Problem_Parameters': problem_params,
            'Quality_Weights': quality_weights,
        },
        'problem_setup': {
            'start_node': start_node,
            'exit_node': exit_node,
            'stops_count': len(stops_to_visit),
            'opposites_map': all_opposites
        },
        'solution': {
            'Q_best': controller.history[-1]['best_quality_so_far'],
            'num_routes': num_routes,
            'total_distance': total_dist,
            'coverage': coverage,
            'routes': best_routes
        }
    }

    # Gera todos os relatórios
    if controller.history:
        generate_final_report(final_report_data, output_dir=output_dir)
        generate_convergence_plots(controller.history, output_dir=output_dir)
        generate_comparison_csv(controller.history, output_dir=output_dir)


    print("\nResumo da Melhor Solução:")
    print(f"  - Número de Rotas: {num_routes}")
    print(f"  - Distância Total: {total_dist:.2f}")
    print(f"  - Cobertura: {coverage*100:.2f}%")

    # Geração das visualizações de mapa (código existente)
    expanded_routes = [expand_meta_route(r, meta_graph, meta_edges) for r in best_routes]


    if args.output:
        output_path = os.path.join(output_dir, args.output)
        if args.folium:
            visualize_routes_folium(graph, expanded_routes[0], bus_stops_nodes, output_path, start_node, exit_node)
        else:
            plot_multiple_routes(original_graph=graph, routes=expanded_routes, all_bus_stops=bus_stops_nodes, output_path=output_path, start_node=start_node, exit_node=exit_node)
        print(f"Visualização da rota final salva em: '{output_path}'") 

    if args.meta_output:
        meta_routes_path = os.path.join(output_dir, f"routes_on_{args.meta_output}")
        plot_multiple_meta_routes(meta_graph, best_routes, stops_to_visit, meta_routes_path, start_node, exit_node, show_labels=True)
        print(f"Visualização do meta-grafo com rotas salva em: '{meta_routes_path}'")


if __name__ == "__main__":
    main()