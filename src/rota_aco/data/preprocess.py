import osmnx as ox
import networkx as nx


def load_graph(graph_path: str) -> nx.MultiDiGraph:
    """
    Carrega o grafo original a partir de um arquivo GraphML.
    """
    G = ox.load_graphml(graph_path)
    return G


def get_bus_stops(G: nx.MultiDiGraph) -> list:
    """
    Retorna a lista de nós que representam paradas de ônibus, com base no atributo 'bus_stop'.
    """
    return [n for n, data in G.nodes(data=True)
            if data.get('bus_stop', '').strip().lower() == 'true']


def pre_process_opposites(bus_stops: list, G: nx.MultiDiGraph, precision: int = 4) -> dict:
    """
    Agrupa as paradas por coordenadas arredondadas e identifica opostos (mesma localização).
    Retorna um dicionário mapping de cada parada para lista de suas opostas.
    """
    groups = {}
    for stop in bus_stops:
        data = G.nodes[stop]
        key = (round(float(data['x']), precision), round(float(data['y']), precision))
        groups.setdefault(key, []).append(stop)

    opposites = {}
    for stops in groups.values():
        if len(stops) > 1:
            for s in stops:
                opposites[s] = [o for o in stops if o != s]
    return opposites
