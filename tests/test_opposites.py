#!/usr/bin/env python3
"""
Script de teste: Lista pares de paradas opostas por proximidade e por acesso.
Uso: python test_opposites.py <graphml_file>
"""
import sys
from rota_aco.data.preprocess import load_graph, get_bus_stops, pre_process_opposites
from rota_aco.data.opposites import find_opposites_by_access

def main(graphml_path: str, precision: int = 5):
    # Carrega grafo e identifica paradas de ônibus
    G = load_graph(graphml_path)
    bus_stops = get_bus_stops(G)

    # Calcula opostos por proximidade
    opposites_proximity = pre_process_opposites(bus_stops, G, precision=precision)
    # Calcula opostos por acesso (bus_access)
    opposites_access = find_opposites_by_access(G, bus_stops)

    # Exibe resultados
    print("Paradas opostas por proximidade geográfica:")
    for stop, opps in opposites_proximity.items():
        print(f"  {stop} -> {opps}")

    print("\nParadas opostas por acesso via bus_access:")
    for stop, opps in opposites_access.items():
        print(f"  {stop} -> {opps}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python test_opposites.py <arquivo.graphml>")
        sys.exit(1)
    graphml = sys.argv[1]
    main(graphml)
