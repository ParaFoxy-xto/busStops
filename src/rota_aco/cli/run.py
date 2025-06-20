# src/rota_aco/cli/run.py

import argparse
import networkx as nx

from rota_aco.data.preprocess import load_graph, get_bus_stops, pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access
from rota_aco.graph.build_meta import build_meta_graph, expand_meta_route, filter_opposite_meta_route, prune_meta_graph_edges
from rota_aco.graph.dfs_routes import prune_meta_graph, find_k_shortest_paths, filter_paths_remove_opposites
from rota_aco.graph.route_filter import remove_duplicate_paths, filter_paths_by_length, top_n_paths_per_pair
from rota_aco.graph.utils import find_nearest_bus_stop, find_edges_between_opposites
from rota_aco.aco.run import executar_aco
from rota_aco.viz.matplotlib_viz import plot_meta_graph, plot_meta_route, plot_multiple_routes
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
        G, 
        precision=args.precision, 
        extra_nodes=extra_nodes,
        start_node=start_node,
        exit_node=exit_node
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
    G, bus_stops, extra_nodes, reps,meta_G, meta_edges, candidate_G,opposites_proximity, opposites_access = cmd_dfs(args)

    # Garante que o local_search tenha acesso ao meta_edges
    candidate_G.graph['meta_edges'] = meta_edges
     # start e exit nodes são os nós de entrada e saída do grafo original
    start_node = extra_nodes[0] if extra_nodes else reps[0]
    exit_node  = extra_nodes[1] if len(extra_nodes) > 1 else None
    # Prepara paradas para o ACO (exclui start_node e exit_node)
    paradas_aco = set(reps)
    if start_node is not None:
        paradas_aco.discard(start_node)
    if exit_node is not None:
        paradas_aco.discard(exit_node)
    print(f"[DEBUG] Stops with demand (paradas_aco): {sorted(paradas_aco)}")
        
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

    if best_ant is None:
        print("❌ Nenhuma solução encontrada pelo ACO")
        return
        
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
    uncovered_stops = sorted(set(paradas_aco) - set(final_route))
    print(f"[DEBUG] Stops NOT covered by this route: {uncovered_stops}")

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
    """
    Subcomando ACS: executa o ACS multi-colônia (ACS-TIME + ACS-VEHICLE) para múltiplas rotas.
    Utiliza o ACSController para combinar as duas heurísticas, diferente do ACO clássico (subcomando 'meta').
    """
    # 1) Carrega grafo e identifica paradas
    try:
        G = load_graph(args.graph)
    except Exception as e:
        print(f"Erro ao carregar o grafo: {e}")
        return
    bus_stops = get_bus_stops(G)
    if not bus_stops:
        print("Nenhuma parada de ônibus encontrada no grafo.")
        return

    # 2) Determina nós de início e fim mais próximos
    try:
        start = find_nearest_bus_stop(
            G, list(G.nodes()), (args.start_lat, args.start_lon)
        )
        exit_ = find_nearest_bus_stop(
            G, list(G.nodes()), (args.exit_lat, args.exit_lon)
        )
    except Exception as e:
        print(f"Erro ao determinar nós de início/fim: {e}")
        return
    extra = [start, exit_]

    # 3) Constrói meta-grafo
    try:
        meta_G, meta_edges, reps, opposites_proximity, opposites_access, mapping, groups = build_meta_graph(
            G,
            precision=args.precision,
            extra_nodes=extra,
            start_node=start,
            exit_node=exit_
        )
        # Prune meta-graph edges to remove long detours
        meta_G, meta_edges = prune_meta_graph_edges(meta_G, meta_edges, factor=2.0)
    except Exception as e:
        print(f"Erro ao construir o meta-grafo: {e}")
        return

    if meta_G.number_of_nodes() == 0 or meta_G.number_of_edges() == 0:
        print("Meta-grafo vazio. Verifique os parâmetros de entrada.")
        return

    # Plot meta-graph if requested
    if args.meta_output:
        try:
            plot_meta_graph(
                G, meta_G,
                start_node=start,
                exit_node=exit_,
                show_labels=args.show_labels,
                output=args.meta_output
            )
            print(f"Meta-grafo salvo em: {args.meta_output}")
        except Exception as e:
            print(f"Erro ao plotar meta-grafo: {e}")

    meta_G.graph['meta_edges'] = meta_edges

    # 3.2) Prepara o conjunto de meta-nós a cobrir pelo ACO
    paradas_aco = set(reps)
    if start is not None:
        paradas_aco.discard(start)
    if exit_ is not None:
        paradas_aco.discard(exit_)
    print(f"[DEBUG] Stops with demand (paradas_aco): {sorted(paradas_aco)}")

    # 3.3) Combina opostos de proximidade e acesso
    combined_opposites = {**opposites_proximity}
    for u, opps in opposites_access.items():
        combined_opposites.setdefault(u, set()).update(opps)
    # garante bidirecionalidade
    for u, opps in list(combined_opposites.items()):
        for v in opps:
            combined_opposites.setdefault(v, set()).add(u)
    combined_opposites = {u: list(v) for u, v in combined_opposites.items()}

    # 4) Executa o ACS multi-colônia
    try:
        controller = ACSController(
            graph=meta_G,
            meta_edges=meta_edges,
            stops=list(paradas_aco),
            start_node=start,
            alpha=args.alpha,
            beta=args.beta,
            evaporation=args.evaporation,
            Q=args.Q,
            opposites=combined_opposites,
            exit_node=exit_
        )
        best_routes, best_dist, best_count, route_directions, coverage_percentage, uncovered_stops = controller.run(
            n_iterations=args.iterations,
            n_ants=args.ants,
            lam=args.lam,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Erro ao executar o ACSController: {e}")
        return

    print(f"\nMelhor conjunto de rotas encontradas pelo ACS multi-colônia:")
    all_covered = set()
    for i, (route, direction) in enumerate(zip(best_routes, route_directions), 1):
        print(f"  Rota {i} ({direction}): {route}")
        all_covered.update(stop for stop in route if stop in paradas_aco)
    not_covered = sorted(set(paradas_aco) - all_covered)
    print(f"[DEBUG] Stops NOT covered by any route: {not_covered}")
    print(f"Distância total: {best_dist:.2f}")
    print(f"Número de rotas: {best_count}")
    print(f"Cobertura: {coverage_percentage:.1f}% dos passageiros atendidos")
    if uncovered_stops:
        print(f"Paradas não atendidas ({len(uncovered_stops)}): {uncovered_stops}")

    # 5) Expande cada meta-rota em sequência completa de nós reais
    final_routes = [
        expand_meta_route(r, meta_G, meta_edges)
        for r in best_routes
    ]

    bus_stop_set = set(bus_stops)
    # Filter out routes that serve 0 bus stops
    filtered_routes = []
    filtered_directions = []
    for route, direction in zip(final_routes, route_directions):
        stops_in_route = set(route) & bus_stop_set
        # Remove start and exit nodes if present
        stops_in_route.discard(start)
        stops_in_route.discard(exit_)
        if len(stops_in_route) > 0:
            filtered_routes.append(route)
            filtered_directions.append(direction)

    # Print number of unique bus stops accessed by each route
    for idx, (rota, direction) in enumerate(zip(filtered_routes, filtered_directions), start=1):
        stops_in_route = set(rota) & bus_stop_set
        stops_in_route.discard(start)
        stops_in_route.discard(exit_)
        print(f"Rota {idx} ({direction}): acessa {len(stops_in_route)} paradas de ônibus")

    # Warn if more than 2 routes are generated
    if len(filtered_routes) > 2:
        print(f"[WARNING] {len(filtered_routes)} rotas geradas. Isso pode indicar um problema com a lógica do ACS.")

    # 6) Plota cada rota final (gera um PNG para cada)
    if args.output and filtered_routes:
        for idx, rota in enumerate(filtered_routes, start=1):
            out = args.output.replace(".png", f"_{idx}.png")
            try:
                plot_meta_route(
                    G,
                    rota,
                    bus_stops,
                    out,
                    start_node=start,
                    exit_node=exit_,
                    color=f"C{idx % 10}"
                )
            except Exception as e:
                print(f"Erro ao plotar rota {idx}: {e}")

    # 7) Visualização interativa (opcional)
    if args.output and args.folium and filtered_routes:
        try:
            visualize_route_folium(
                G,
                filtered_routes,  # Only valid routes
                bus_stops,
                args.output.replace(".png", ".html"),
                start_node=start,
                exit_node=exit_
            )
        except Exception as e:
            print(f"Erro ao gerar visualização folium: {e}")


def main():
    parser = argparse.ArgumentParser(description="rota_aco CLI")
    sub = parser.add_subparsers(dest='cmd')

    # Subcomando DFS: executa o DFS
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

    # Subcomando META: executa o (ACO)
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
        help="Execute o ACS multi-rota para múltiplas rotas"
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
    p_acs.add_argument("-a", "--ants", type=int, default=10,
                       help="Número de formigas por iteração")
    p_acs.add_argument("--iterations",   type=int,   default=100,
                       help="Número de iterações do controlador")
    p_acs.add_argument("--lambda",       type=float, dest="lam", default=0.5,
                       help="Peso de combinação dos feromônios (0..1)")
    p_acs.add_argument("--alpha",        type=float, default=1.0,
                       help="Parâmetro α para ACS")
    p_acs.add_argument("--beta",         type=float, default=4.0,
                       help="Parâmetro β para ACS (heurística, use 4.0 ou mais para rotas mais diretas)")
    p_acs.add_argument("--evaporation",  type=float, default=0.1,
                       help="Taxa de evaporação global")
    p_acs.add_argument("--pheromone-q",  type=float, dest="Q", default=1.0,
                       help="Quantidade de feromônio depositado")
    p_acs.add_argument("--verbose", action="store_true",
                       help="Exibe logs detalhados de cada iteração do ACS")
    p_acs.add_argument("--output", "-o",
        help="Arquivo PNG para salvar a rota ACS", default=None)
    p_acs.add_argument('--folium', action='store_true', help='Gera visualização folium interativa')
    p_acs.add_argument('--meta-output', help='Arquivo PNG para salvar o meta-grafo', default=None)
    p_acs.add_argument('--show-labels', action='store_true', help='Mostra labels dos nós no meta-grafo')
    p_acs.set_defaults(func=cmd_acs)


    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
