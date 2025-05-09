# viz/folium_viz.py

import folium
from folium import Map, PolyLine, CircleMarker
from typing import Any, List, Optional


def visualize_route_folium(
    G,
    route: List[Any],
    bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None
) -> None:
    """
    Gera um mapa interativo com Folium para a rota:
    - marca paradas, traça a rota e destaca start/exit em verde
    """
    # Centro do mapa baseado na rota
    lats = [float(G.nodes[n]['y']) for n in route]
    lngs = [float(G.nodes[n]['x']) for n in route]
    center = (sum(lats)/len(lats), sum(lngs)/len(lngs))

    m = Map(location=center, zoom_start=14)

    # Desenha rota
    coords = list(zip(lats, lngs))
    PolyLine(locations=coords, color='red', weight=5, opacity=0.8).add_to(m)

    # Marca paradas de ônibus
    for n in bus_stops:
        lat, lon = float(G.nodes[n]['y']), float(G.nodes[n]['x'])
        CircleMarker(location=(lat, lon), radius=3, color='blue', fill=True).add_to(m)

    # Destaca start em verde
    if start_node is not None:
        lat0, lon0 = float(G.nodes[start_node]['y']), float(G.nodes[start_node]['x'])
        CircleMarker(location=(lat0, lon0), radius=6, color='green', fill=True, popup='Start').add_to(m)
    # Destaca exit em verde
    if exit_node is not None:
        lat1, lon1 = float(G.nodes[exit_node]['y']), float(G.nodes[exit_node]['x'])
        CircleMarker(location=(lat1, lon1), radius=6, color='green', fill=True, popup='Exit', icon=None).add_to(m)

    m.save(output_path)
