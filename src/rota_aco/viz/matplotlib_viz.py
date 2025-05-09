# viz/matplotlib_viz.py

import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, List, Optional


def plot_meta_route(
    G: nx.MultiDiGraph,
    route: List[Any],
    bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None
) -> None:
    """
    Plota a rota expandida sobre o grafo original usando Matplotlib.
    - G: grafo original
    - route: lista de nós no caminho final
    - bus_stops: lista de nós de parada para destacar
    - output_path: arquivo de saída (PNG)
    - start_node, exit_node: nós de parada inicial e final para destacar em verde
    """
    # Coordenadas de todos os nós (ruas)
    xs_all = [float(data['x']) for _, data in G.nodes(data=True)]
    ys_all = [float(data['y']) for _, data in G.nodes(data=True)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(xs_all, ys_all, c='lightgray', s=2, label='Rua')

    # Paradas de ônibus
    xs_stops = [float(G.nodes[n]['x']) for n in bus_stops]
    ys_stops = [float(G.nodes[n]['y']) for n in bus_stops]
    ax.scatter(xs_stops, ys_stops, c='blue', s=20, label='Paradas')

    # Rota
    xs_route = [float(G.nodes[n]['x']) for n in route]
    ys_route = [float(G.nodes[n]['y']) for n in route]
    ax.plot(xs_route, ys_route, c='red', linewidth=2, label='Rota')

    # Destaque start e exit em verde
    if start_node is not None:
        x0, y0 = float(G.nodes[start_node]['x']), float(G.nodes[start_node]['y'])
        ax.scatter([x0], [y0], c='green', s=100, marker='o', label='Start')
    if exit_node is not None:
        x1, y1 = float(G.nodes[exit_node]['x']), float(G.nodes[exit_node]['y'])
        ax.scatter([x1], [y1], c='green', s=100, marker='X', label='Exit')

    ax.legend()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Rota Ótima no Grafo Original')
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# src/rota_aco/viz/matplotlib_viz.py

# src/rota_aco/viz/matplotlib_viz.py

import matplotlib.pyplot as plt

def plot_meta_graph(G, meta_G, start_node=None, exit_node=None, show_labels=True, output=None):
    """
    Plota o grafo original em cinza e sobrepõe as arestas do meta-grafo em azul,
    destacando nós de meta-Grafo em vermelho. Se fornecido, marca start/exit em verde.
    Se show_labels=True, escreve o ID de cada nó ao lado do ponto.
    Se 'output' for informado, salva em arquivo; caso contrário, mostra na tela.
    """
    fig, ax = plt.subplots(figsize=(10,10))

    # 1) Todos os nós do grafo original em cinza
    xs_all = [float(data['x']) for _, data in G.nodes(data=True)]
    ys_all = [float(data['y']) for _, data in G.nodes(data=True)]
    ax.scatter(xs_all, ys_all, c='lightgray', s=5, label='Grafo original')

    # 2) Todas as arestas do meta-grafo (linhas azuis)
    for u, v, data in meta_G.edges(data=True):
        x1, y1 = float(G.nodes[u]['x']), float(G.nodes[u]['y'])
        x2, y2 = float(G.nodes[v]['x']), float(G.nodes[v]['y'])
        ax.plot([x1, x2], [y1, y2], c='blue', linewidth=1)

    # 3) Nós do meta-grafo em vermelho
    xs_meta = [float(G.nodes[n]['x']) for n in meta_G.nodes()]
    ys_meta = [float(G.nodes[n]['y']) for n in meta_G.nodes()]
    ax.scatter(xs_meta, ys_meta, c='red', s=30, label='Nós meta-grafo')

    # 4) Opcional: labels dos nós
    if show_labels:
        for n in meta_G.nodes():
            x, y = float(G.nodes[n]['x']), float(G.nodes[n]['y'])
            ax.text(x, y, str(n), fontsize=6, color='black',
                    verticalalignment='bottom', horizontalalignment='right')

    # 5) Destacar start/exit em verde, se existirem
    if start_node is not None:
        x, y = float(G.nodes[start_node]['x']), float(G.nodes[start_node]['y'])
        ax.scatter(x, y, c='green', s=80, marker='*', label='Start')
        if show_labels:
            ax.text(x, y, str(start_node), fontsize=8, color='green',
                    verticalalignment='bottom', horizontalalignment='left')
    if exit_node is not None:
        x, y = float(G.nodes[exit_node]['x']), float(G.nodes[exit_node]['y'])
        ax.scatter(x, y, c='green', s=80, marker='X', label='Exit')
        if show_labels:
            ax.text(x, y, str(exit_node), fontsize=8, color='green',
                    verticalalignment='top', horizontalalignment='left')

    ax.legend(loc='upper right')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Meta-Grafo: nós e arestas")

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Salvo em {output}")
    else:
        plt.show()
