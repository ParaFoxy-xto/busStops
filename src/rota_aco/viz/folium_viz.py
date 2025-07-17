# src/rota_aco/viz/folium_viz.py

"""
Funções para gerar visualizações de rotas interativas usando a biblioteca Folium.
"""

from typing import Any, List, Optional, Union

import folium
import networkx as nx
from folium import Map, PolyLine, CircleMarker

def visualize_routes_folium(
    graph: nx.MultiDiGraph,
    routes: Union[List[Any], List[List[Any]]],
    all_bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None
) -> None:
    """
    Gera um mapa HTML interativo com uma ou mais rotas.

    Args:
        graph: O grafo original com dados de coordenadas dos nós.
        routes: Uma única rota (lista de nós) ou uma lista de rotas.
        all_bus_stops: Lista de todos os nós de parada de ônibus para marcar no mapa.
        output_path: Caminho para salvar o arquivo HTML do mapa.
        start_node: O nó de partida a ser destacado.
        exit_node: O nó de chegada a ser destacado.
    """
    # Garante que 'routes' seja sempre uma lista de listas
    if not routes or not isinstance(routes[0], list):
        routes = [routes]

    # Centraliza o mapa com base na primeira rota
    try:
        first_route_nodes = routes[0]
        lats = [float(graph.nodes[n]['y']) for n in first_route_nodes]
        lons = [float(graph.nodes[n]['x']) for n in first_route_nodes]
        map_center = (sum(lats) / len(lats), sum(lons) / len(lons))
    except (IndexError, KeyError, ZeroDivisionError):
        # Fallback se a rota estiver vazia ou os nós não tiverem coordenadas
        map_center = (-15.77, -47.87) # Coordenadas de fallback (ex: Brasília)

    # Cria o mapa base
    m = Map(location=map_center, zoom_start=14, tiles='CartoDB positron')

    # Define um ciclo de cores para as rotas
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # 1. Desenha cada rota
    for i, route in enumerate(routes):
        if not route: continue
        route_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in route if n in graph.nodes]
        PolyLine(
            locations=route_coords,
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.8,
            tooltip=f"Rota {i+1}"
        ).add_to(m)

    # 2. Marca todas as paradas de ônibus
    for stop in all_bus_stops:
        if stop in graph.nodes:
            lat, lon = graph.nodes[stop]['y'], graph.nodes[stop]['x']
            CircleMarker(
                location=(lat, lon),
                radius=4,
                color='#0033cc', # Azul escuro
                fill=True,
                fill_color='#aaccff',
                fill_opacity=0.7,
                tooltip=f"Parada: {stop}"
            ).add_to(m)

    # 3. Destaca o nó de partida
    if start_node and start_node in graph.nodes:
        lat, lon = graph.nodes[start_node]['y'], graph.nodes[start_node]['x']
        folium.Marker(
            location=(lat, lon),
            tooltip=f"Partida: {start_node}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

    # 4. Destaca o nó de chegada
    if exit_node and exit_node in graph.nodes:
        lat, lon = graph.nodes[exit_node]['y'], graph.nodes[exit_node]['x']
        folium.Marker(
            location=(lat, lon),
            tooltip=f"Chegada: {exit_node}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

    m.save(output_path)
    print(f"Mapa interativo salvo em: '{output_path}'")