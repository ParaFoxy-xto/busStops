# src/rota_aco/aco/run.py

import random
import networkx as nx
from typing import Set
from rota_aco.aco.ant import Ant
from rota_aco.aco.local_search import local_search_2opt

def compute_route_distance(route, grafo):
    """
    Calcula distância total de uma rota de nós em um grafo de rotas candidatas.
    Suporta arestas dict, float e fallback via grafo.graph['meta_edges'].
    """
    total = 0.0
    meta_edges = grafo.graph.get('meta_edges', {})
    for u, v in zip(route, route[1:]):
        if grafo.has_edge(u, v):
            data = grafo[u][v]
        elif grafo.has_edge(v, u):
            data = grafo[v][u]
        elif (u, v) in meta_edges:
            total += float(meta_edges[(u, v)]['length'])
            continue
        elif (v, u) in meta_edges:
            total += float(meta_edges[(v, u)]['length'])
            continue
        else:
            raise KeyError(f"Aresta não encontrada: {u}->{v}")
        if isinstance(data, dict):
            total += float(data.get('length', data.get('weight', 0)))
        else:
            total += float(data)
    return total

def executar_aco(
    graph,
    stops,
    start_node,
    combined_opposites,   # agora o 4º argumento
    n_formigas=10,
    n_iteracoes=100,
    max_no_improvement=20,
    Q=1.0,
    evaporacao=0.1
):
    # inicializa feromônios uniformes
    pheromones = {
        (u, v): 1.0
        for u, v in graph.edges()
    }

    # loop de iterações
    best_ant = None
    best_dist = float('inf')
    no_improve = 0

    for _ in range(n_iteracoes):
        # cria formigas, passando combined_opposites
        ants = [
            Ant(start_node, stops, combined_opposites, combined_opposites)
            for _ in range(n_formigas)
        ]
        # cada formiga constrói uma rota heurística
        for ant in ants:
            while True:
                nxt = ant._choose_next(graph, pheromones)
                if nxt is None:
                    break
                ant.caminho.append(nxt)
                ant.distancia_total = compute_route_distance(ant.caminho, graph)
                ant.current = nxt
            # opcional: local search
            ant.caminho = local_search_2opt(ant.caminho, graph)
            ant.distancia_total = compute_route_distance(ant.caminho, graph)

        # atualiza melhor solução
        for ant in ants:
            if ant.distancia_total < best_dist:
                best_dist = ant.distancia_total
                best_ant = ant
                no_improve = 0
        else:
            no_improve += 1
        if no_improve >= max_no_improvement:
            break

        # evaporação
        for key in pheromones:
            pheromones[key] *= (1.0 - evaporacao)

        # reforço de feromônio nas melhores formigas
        for ant in ants:
            delta = Q / ant.distancia_total if ant.distancia_total > 0 else 0
            for u, v in zip(ant.caminho, ant.caminho[1:]):
                pheromones[(u, v)] = pheromones.get((u, v), 0) + delta

    return best_ant