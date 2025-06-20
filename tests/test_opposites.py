#!/usr/bin/env python3
"""
Script de teste: Lista pares de paradas opostas por proximidade e por acesso.
Uso: python test_opposites.py <graphml_file>
"""
import sys
from rota_aco.data.preprocess import load_graph, get_bus_stops, pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def print_colored_pairs(pairs, title):
    print(title)
    color_codes = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]
    used = set()
    color_map = {}
    color_idx = 0
    for a, b in pairs:
        if (b, a) in used:
            continue
        used.add((a, b))
        color = color_codes[color_idx % len(color_codes)]
        color_map[(a, b)] = color
        color_map[(b, a)] = color
        color_idx += 1
        print(f"  \033[{color}m{a} <-> {b}\033[0m")

def plot_opposites(G, bus_stops, pairs, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = [float(G.nodes[n]['x']) for n in bus_stops]
    ys = [float(G.nodes[n]['y']) for n in bus_stops]
    ax.scatter(xs, ys, c='blue', s=30, label='Paradas')
    # Assign a color to each pair
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    used = set()
    color_idx = 0
    for a, b in pairs:
        if (b, a) in used:
            continue
        used.add((a, b))
        color = color_list[color_idx % len(color_list)]
        color_idx += 1
        x1, y1 = float(G.nodes[a]['x']), float(G.nodes[a]['y'])
        x2, y2 = float(G.nodes[b]['x']), float(G.nodes[b]['y'])
        ax.plot([x1, x2], [y1, y2], c=color, linewidth=2)
        ax.scatter([x1, x2], [y1, y2], c=color, s=80)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

def main(graphml_path: str, precision: int = 5):
    # Carrega grafo e identifica paradas de ônibus
    G = load_graph(graphml_path)
    bus_stops = get_bus_stops(G)

    # Calcula opostos por proximidade
    opposites_proximity = pre_process_opposites(bus_stops, G, precision=precision)
    # Calcula opostos por acesso (bus_access)
    opposites_access = find_opposites_by_access(G, bus_stops)

    # Build unique pairs for proximity
    prox_pairs = set()
    for stop, opps in opposites_proximity.items():
        for opp in opps:
            if stop != opp:
                prox_pairs.add(tuple(sorted((stop, opp))))
    print_colored_pairs(prox_pairs, "Paradas opostas por proximidade geográfica (coloridas):")
    plot_opposites(G, bus_stops, prox_pairs, "Opostos por proximidade geográfica")

    # Build unique pairs for access
    access_pairs = set()
    for stop, opps in opposites_access.items():
        for opp in opps:
            if stop != opp:
                access_pairs.add(tuple(sorted((stop, opp))))
    print_colored_pairs(access_pairs, "\nParadas opostas por acesso via bus_access (coloridas):")
    plot_opposites(G, bus_stops, access_pairs, "Opostos por acesso bus_access")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python test_opposites.py <arquivo.graphml>")
        sys.exit(1)
    graphml = sys.argv[1]
    main(graphml)
