# src/rota_aco/cli/run.py

import argparse
import networkx as nx

from rota_aco.data.preprocess import load_graph, get_bus_stops, pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access
from rota_aco.graph.build_meta import build_meta_graph, expand_meta_route, filter_opposite_meta_route
from rota_aco.graph.dfs_routes import prune_meta_graph, find_k_shortest_paths, filter_paths_remove_opposites
from rota_aco.graph.route_filter import remove_duplicate_paths, filter_paths_by_length, top_n_paths_per_pair
from rota_aco.graph.utils import find_nearest_bus_stop, find_edges_between_opposites
from rota_aco.aco.run import executar_aco
from rota_aco.viz.matplotlib_viz import plot_meta_graph, plot_meta_route
from rota_aco.viz.folium_viz import visualize_route_folium
from rota_aco.aco.controller import ACSController


def cmd_dfs(args):
    """Subcomando DFS: gera grafo de rotas candidatas"""
    G = load_graph(args.graph)
    bus_stops = get_bus_stops(G)

    # Determine start and exit nodes
    extra_nodes = []
    start_node = exit_node = None
    if args.start_lat is not None and args.start_lon is not None:
        start_node = find_nearest_bus_stop(G, list(G.nodes()), (args.start_lat, args.start_lon))
        extra_nodes.append(start_node)
        print(f"Start node: {start_node}")
    if args.exit_lat is not None and args.exit_lon is not None:
        exit_node = find_nearest_bus_stop(G, list(G.nodes()), (args.exit_lat, args.exit_lon))
        extra_nodes.append(exit_node)
        print(f"Exit node: {exit_node}")

    # Preprocess opposites
    opp_prox = pre_process_opposites(bus_stops, G, precision=args.precision)
    opp_acc = find_opposites_by_access(G, bus_stops)

    # Build meta-graph
    meta_G, meta_edges, reps, _, _, mapping, groups = build_meta_graph(
        G, precision=args.precision, extra_nodes=extra_nodes
    )
    print(f"Meta-grafo: {meta_G.number_of_nodes()} nós, {meta_G.number_of_edges()} arestas")

    # Plot meta-graph
    plot_meta_graph(
        G, meta_G,
        start_node=start_node,
        exit_node=exit_node,
        show_labels=args.show_labels,
        output=args.meta_output
    )
    violations_meta = find_edges_between_opposites(meta_G, opp_prox, opp_acc)
    print("✅ meta_G sem opostos" if not violations_meta else f"⚠️ {len(violations_meta)} arestas opostas no meta_G")

    # Prune and generate paths
    pruned = prune_meta_graph(meta_G, k=args.k_prune)
    raw_paths = find_k_shortest_paths(pruned, extra_nodes or reps, reps, k=args.k_paths)
    valid = filter_paths_remove_opposites(raw_paths, opp_prox, opp_acc)
    print(f"{len(valid)} caminhos válidos após DFS+prune+filtro")

    # Initial route filters
    unique = remove_duplicate_paths(valid)
    trimmed = filter_paths_by_length(unique, meta_edges, max_percentile=args.length_percentile)
    if args.top_n:
        trimmed = top_n_paths_per_pair(trimmed, meta_edges, n=args.top_n)
    print(f"{len(trimmed)} caminhos após filtros iniciais")

    # Build candidate graph
    candidate_G = nx.DiGraph()
    candidate_G.add_nodes_from(reps)
    for p in trimmed:
        u, v = p[0], p[-1]
        w = sum(meta_edges[(a, b)]["length"] for a, b in zip(p[:-1], p[1:]))
        candidate_G.add_edge(u, v, weight=w, path=p)
    print(f"candidate_G: {candidate_G.number_of_nodes()} nós, {candidate_G.number_of_edges()} arestas")
    violations_cand = find_edges_between_opposites(candidate_G, opp_prox, opp_acc)
    print("✅ candidate_G sem opostos" if not violations_cand else f"⚠️ {len(violations_cand)} opostos no candidate_G")

    return G, bus_stops, extra_nodes, reps, meta_G, meta_edges, candidate_G, opp_prox, opp_acc


def cmd_meta(args):
    """Subcomando META: pipeline completo com ACO"""
    # 1-9: executa exatamente o fluxo de cmd_dfs, retornando:
    #    G, bus_stops, extra_nodes, reps, meta_edges, candidate_G, opp_prox, opp_acc
    G, bus_stops, extra_nodes, reps,meta_G, meta_edges, candidate_G,opposites_proximity, opposites_access = cmd_dfs(args)

    # Garante que o local_search tenha acesso ao meta_edges
    candidate_G.graph['meta_edges'] = meta_edges
     # start e exit nodes são os nós de entrada e saída do grafo original
    start_node = extra_nodes[0] if extra_nodes else reps[0]
    exit_node  = extra_nodes[1] if len(extra_nodes) > 1 else None
    # Prepara paradas para o ACO (exclui exit_node)
    paradas_aco = set(reps)
    if exit_node is not None:
        paradas_aco.discard(exit_node)
        
    # Combina opostos de proximidade e de acesso num só dict
    combined_opposites = {}
    for u, opps in opposites_proximity.items():
        combined_opposites[u] = set(opps)
    for u, opps in opposites_access.items():
        combined_opposites.setdefault(u, set()).update(opps)
    # garante bidirecionalidade
    for u, opps in list(combined_opposites.items()):
        for v in opps:
            combined_opposites.setdefault(v, set()).add(u)
    # converte para listas
    combined_opposites = {u: list(v) for u, v in combined_opposites.items()}

    best_ant = executar_aco(
    candidate_G,
    paradas_aco,
    start_node,
    combined_opposites,    
    n_formigas=args.ants,
    n_iteracoes=args.iterations,
    max_no_improvement=args.diversify,
    Q=args.pheromone_q,
    evaporacao=args.evaporation
    )


    meta_route = best_ant.caminho
    # filtra saltos entre paradas opostas no meta-grafo
    meta_route = filter_opposite_meta_route(
        meta_route,
        opposites_proximity,
        opposites_access
    )
    if exit_node is not None:
        meta_route.append(exit_node)

    print(f"Melhor rota meta-grafo: {meta_route}")
    # depois de obter meta_G, meta_edges, meta_route...

    # Expande a rota final completa
    final_route = expand_meta_route(meta_route, meta_G, meta_edges)

    print(f"Rota expandida: {final_route}")

    # 7) Visualização final
    if args.folium:
        visualize_route_folium(
            G,
            final_route,
            bus_stops,
            args.output,
            start_node=start_node,
            exit_node=exit_node
        )
    else:
        plot_meta_route(
            G,
            final_route,
            bus_stops,
            args.output,
            start_node=start_node,
            exit_node=exit_node
        )

def cmd_acs(args):


    # 1) Carrega grafo e identifica paradas
    G = load_graph(args.graph)
    bus_stops = get_bus_stops(G)

    # 2) Determina nós de início e fim mais próximos
    start = find_nearest_bus_stop(
        G, list(G.nodes()), (args.start_lat, args.start_lon)
    )
    exit_ = find_nearest_bus_stop(
        G, list(G.nodes()), (args.exit_lat, args.exit_lon)
    )
    extra = [start, exit_]

    # 3) Constrói meta-grafo
    meta_G, meta_edges, reps, opp_p, opp_a, mapping, groups = build_meta_graph(
        G,
        precision=args.precision,
        extra_nodes=extra
    )

    # # 4) Instancia o controlador ACS multicolônia
    # print(">>> reps:", reps, type(reps))
    # print(">>> start:", start, type(start))
    # print(">>> meta_edges is dict?", isinstance(meta_edges, dict))
    ctrl = ACSController(
        meta_G,
        reps,
        start,
        meta_edges,      
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
        Q=args.Q
    )




    # 5) Executa o controller
    meta_routes, best_dist, best_count = ctrl.run(
        n_iterations=args.iterations,
        ants_time=args.ants_time,
        ants_vehicle=args.ants_vehicle,
        lam=args.lam
    )

    # 6) Expande rotas para o grafo original
    final_routes = [
        expand_meta_route(r, meta_G, meta_edges)
        for r in meta_routes
    ]

    # 7) Exibe resultados
    print("Melhor conjunto de rotas (expandido):")
    for i, route in enumerate(final_routes, 1):
        print(f"  Rota {i}: {route}")
    print(f"Distância total: {best_dist}")
    print(f"Número de rotas: {best_count}\n")

def main():
    parser = argparse.ArgumentParser(description="rota_aco CLI")
    sub = parser.add_subparsers(dest='cmd')

    # DFS
    p_dfs = sub.add_parser('dfs', help='Gerar rotas candidatas')
    p_dfs.add_argument('graph')
    p_dfs.add_argument('-p', '--precision', type=int, default=6)
    p_dfs.add_argument('--start-lat', type=float)
    p_dfs.add_argument('--start-lon', type=float)
    p_dfs.add_argument('--exit-lat', type=float)
    p_dfs.add_argument('--exit-lon', type=float)
    p_dfs.add_argument('--k-prune', type=int, default=5)
    p_dfs.add_argument('--k-paths', type=int, default=3)
    p_dfs.add_argument('--length-percentile', type=float, default=0.75)
    p_dfs.add_argument('--top-n', type=int, default=None)
    p_dfs.add_argument('--meta-output', default=None)
    p_dfs.add_argument('--show-labels', action='store_true')
    p_dfs.set_defaults(func=cmd_dfs)

    # META (ACO)
    p_meta = sub.add_parser('meta', help='Pipeline completo com ACO')
    p_meta.add_argument('graph')
    p_meta.add_argument('-p', '--precision', type=int, default=6)
    p_meta.add_argument('--start-lat', type=float)
    p_meta.add_argument('--start-lon', type=float)
    p_meta.add_argument('--exit-lat', type=float)
    p_meta.add_argument('--exit-lon', type=float)
    p_meta.add_argument('--k-prune', type=int, default=5)
    p_meta.add_argument('--k-paths', type=int, default=3)
    p_meta.add_argument('--length-percentile', type=float, default=0.75)
    p_meta.add_argument('--top-n', type=int, default=None)
    p_meta.add_argument('--show-labels', action='store_true')
    p_meta.add_argument('--meta-output', default=None)
    p_meta.add_argument('-a', '--ants', type=int, default=10)
    p_meta.add_argument('-i', '--iterations', type=int, default=300)
    p_meta.add_argument('-d', '--diversify', type=int, default=10)
    p_meta.add_argument('-q', '--pheromone-q', type=float, default=100.0)
    p_meta.add_argument('-e', '--evaporation', type=float, default=0.1)
    p_meta.add_argument('-f', '--folium', action='store_true')
    p_meta.add_argument('-o', '--output', default='route.png')
    p_meta.set_defaults(func=cmd_meta)

        # Subcomando acs: executa o ACS multi-colônia
    p_acs = sub.add_parser(
        "acs",
        help="Execute o ACS-TIME + ACS-VEHICLE para múltiplas rotas"
    )
    p_acs.add_argument("graph", help="Caminho para o GraphML")
    p_acs.add_argument("--precision", "-p", type=int, default=6,
                       help="Precisão para agrupamento")
    p_acs.add_argument("--start-lat",  type=float, required=True,
                       help="Latitude do ponto inicial")
    p_acs.add_argument("--start-lon",  type=float, required=True,
                       help="Longitude do ponto inicial")
    p_acs.add_argument("--exit-lat",   type=float, required=True,
                       help="Latitude do ponto final")
    p_acs.add_argument("--exit-lon",   type=float, required=True,
                       help="Longitude do ponto final")
    p_acs.add_argument("--ants-time",    type=int,   default=10,
                       help="Número de formigas ACS-TIME")
    p_acs.add_argument("--ants-vehicle", type=int,   default=10,
                       help="Número de formigas ACS-VEHICLE")
    p_acs.add_argument("--iterations",   type=int,   default=100,
                       help="Número de iterações do controlador")
    p_acs.add_argument("--lambda",       type=float, dest="lam", default=0.5,
                       help="Peso de combinação dos feromônios (0..1)")
    p_acs.add_argument("--alpha",        type=float, default=1.0,
                       help="Parâmetro α para ACS-TIME")
    p_acs.add_argument("--beta",         type=float, default=2.0,
                       help="Parâmetro β para ACS-TIME")
    p_acs.add_argument("--evaporation",  type=float, default=0.1,
                       help="Taxa de evaporação global")
    p_acs.add_argument("--pheromone-q",  type=float, dest="Q", default=1.0,
                       help="Quantidade de feromônio depositado")
    p_acs.set_defaults(func=cmd_acs)


    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
