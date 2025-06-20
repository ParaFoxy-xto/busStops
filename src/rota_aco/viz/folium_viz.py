# viz/folium_viz.py

import folium
from folium import Map, PolyLine, CircleMarker, Popup, Icon, DivIcon, Marker
from typing import Any, List, Optional
import itertools


def visualize_route_folium(
    G,
    routes,
    bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None
) -> None:
    """
    Gera um mapa interativo com Folium para uma ou várias rotas:
    - marca paradas, traça todas as rotas e destaca start/exit em verde
    - mostra IDs das paradas de ônibus
    """
    # Permitir tanto uma rota única quanto uma lista de rotas
    if isinstance(routes[0], (str, int)):
        routes = [routes]
    # Centro do mapa baseado na primeira rota
    lats = [float(G.nodes[n]['y']) for n in routes[0]]
    lngs = [float(G.nodes[n]['x']) for n in routes[0]]
    center = (sum(lats)/len(lats), sum(lngs)/len(lngs))

    m = Map(location=center, zoom_start=14, tiles='Cartodb Positron')

    # Cores para múltiplas rotas
    color_cycle = itertools.cycle([
        'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
        'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
        'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'
    ])

    # Desenha todas as rotas
    for route in routes:
        lats = [float(G.nodes[n]['y']) for n in route]
        lngs = [float(G.nodes[n]['x']) for n in route]
        coords = list(zip(lats, lngs))
        color = next(color_cycle)
        PolyLine(locations=coords, color=color, weight=5, opacity=0.8).add_to(m)

    # Marca paradas de ônibus com IDs
    for n in bus_stops:
        lat, lon = float(G.nodes[n]['y']), float(G.nodes[n]['x'])
        # Add bus stop marker with ID popup (hover only)
        CircleMarker(
            location=(lat, lon), 
            radius=3, 
            color='blue', 
            fill=True,
            popup=f'Bus Stop ID: {n}'
        ).add_to(m)

    # Destaca start em verde com ID
    if start_node is not None:
        lat0, lon0 = float(G.nodes[start_node]['y']), float(G.nodes[start_node]['x'])
        CircleMarker(
            location=(lat0, lon0), 
            radius=6, 
            color='green', 
            fill=True, 
            popup=f'Start - ID: {start_node}'
        ).add_to(m)
        
    # Destaca exit em verde com ID
    if exit_node is not None:
        lat1, lon1 = float(G.nodes[exit_node]['y']), float(G.nodes[exit_node]['x'])
        CircleMarker(
            location=(lat1, lon1), 
            radius=6, 
            color='green', 
            fill=True, 
            popup=f'Exit - ID: {exit_node}'
        ).add_to(m)

    m.save(output_path)
