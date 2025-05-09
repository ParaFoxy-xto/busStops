# src/rota_aco/graph/route_filter.py

from statistics import quantiles

def remove_duplicate_paths(paths: list[list]) -> list[list]:
    """
    Remove caminhos duplicados ou reversos.
    Ex.: [A,B,C] e [C,B,A] são considerados a mesma rota.
    """
    seen = set()
    unique = []
    for p in paths:
        key = tuple(p)
        rev = tuple(p[::-1])
        if key in seen or rev in seen:
            continue
        seen.add(key)
        seen.add(rev)
        unique.append(p)
    return unique

def filter_paths_by_length(
    paths: list[list],
    meta_edges: dict,
    max_percentile: float = 0.75
) -> list[list]:
    """
    Calcula o comprimento de cada rota via meta_edges e descarta
    aquelas acima do percentil especificado (ex.: 75%).
    """
    # 1) Calcular distâncias
    dists = []
    for p in paths:
        length = sum(meta_edges[(u, v)]["length"] 
                     for u, v in zip(p[:-1], p[1:]))
        dists.append((p, length))
    # 2) Determinar o limiar
    thresholds = quantiles([d for _, d in dists], n=4)
    cutoff = thresholds[int(max_percentile * 4) - 1]
    # 3) Filtrar
    filtered = [p for p, d in dists if d <= cutoff]
    return filtered

def top_n_paths_per_pair(
    paths: list[list],
    meta_edges: dict,
    n: int = 2
) -> list[list]:
    """
    Para cada par (u,v), mantém apenas as n rotas de menor distância.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in paths:
        u, v = p[0], p[-1]
        dist = sum(meta_edges[(a, b)]["length"] 
                   for a, b in zip(p[:-1], p[1:]))
        buckets[(u, v)].append((dist, p))
    result = []
    for (u, v), lst in buckets.items():
        lst.sort(key=lambda x: x[0])
        result.extend(p for _, p in lst[:n])
    return result
