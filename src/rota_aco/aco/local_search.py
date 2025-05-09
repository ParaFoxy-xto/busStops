# src/rota_aco/aco/local_search.py

def local_search_2opt(route, grafo):
    """
    Aplica busca local 2-opt na rota (lista de nós) sobre o grafo de rotas.
    """
    improved = True
    melhor = route[:]
    while improved:
        improved = False
        best_dist = _route_distance(melhor, grafo)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue
                nova = melhor[:]
                nova[i:j+1] = reversed(nova[i:j+1])
                dist_nova = _route_distance(nova, grafo)
                if dist_nova < best_dist:
                    melhor = nova
                    best_dist = dist_nova
                    improved = True
        route = melhor
    return melhor


def _route_distance(route, grafo):
    """
    Calcula distância total de uma rota, suportando:
      - arestas dict com 'length' ou 'weight'
      - arestas peso direto (float)
      - fallback via grafo.graph['meta_edges']
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