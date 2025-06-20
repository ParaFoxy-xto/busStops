# viz/matplotlib_viz.py

import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, List, Optional
import pandas as pd


def plot_meta_route(
    G: nx.MultiDiGraph,
    route: List[Any],
    bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None,
    color: str = "red",
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
    ax.plot(xs_route, ys_route, linewidth=2, label=f"Rota {color}", c=color)

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

def plot_multiple_routes(
    G: nx.MultiDiGraph,
    routes: List[List[Any]],
    bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """
    Plota múltiplas rotas expandidas sobre o grafo original usando Matplotlib.
    - G: grafo original
    - routes: lista de rotas (cada rota é uma lista de nós)
    - bus_stops: lista de nós de parada para destacar
    - output_path: arquivo de saída (PNG)
    - start_node, exit_node: nós de parada inicial e final para destacar em verde
    - colors: lista de cores para cada rota (opcional)
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Coordenadas de todos os nós (ruas)
    xs_all = [float(data['x']) for _, data in G.nodes(data=True)]
    ys_all = [float(data['y']) for _, data in G.nodes(data=True)]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(xs_all, ys_all, c='lightgray', s=2, label='Rua')

    # Paradas de ônibus
    xs_stops = [float(G.nodes[n]['x']) for n in bus_stops]
    ys_stops = [float(G.nodes[n]['y']) for n in bus_stops]
    ax.scatter(xs_stops, ys_stops, c='blue', s=20, label='Paradas')

    # Plot each route with a different color
    for i, route in enumerate(routes):
        if len(route) < 2:  # Skip invalid routes
            continue
        color = colors[i % len(colors)]
        xs_route = [float(G.nodes[n]['x']) for n in route]
        ys_route = [float(G.nodes[n]['y']) for n in route]
        ax.plot(xs_route, ys_route, linewidth=2, label=f"Rota {i+1}", c=color, alpha=0.8)

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
    ax.set_title(f'Rotas Ótimas no Grafo Original ({len(routes)} rotas)')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_convergence(csv_path: str = 'output/controller_convergence.csv', output_path: str = 'output/convergence_plot.png'):
    """
    Plota a convergência do controlador a partir do CSV gerado (controller_convergence.csv).
    Mostra best_total_time, best_coverage e best_count em função de v (número de veículos/rotas).
    Salva o gráfico em output/convergence_plot.png.
    """
    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Número de veículos/rotas (v)')
    ax1.set_ylabel('Tempo total (best_total_time)', color=color)
    ax1.plot(df['v'], df['best_total_time'], marker='o', color=color, label='Tempo total')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Cobertura (%)', color=color2)
    ax2.plot(df['v'], df['best_coverage'], marker='s', color=color2, label='Cobertura')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Plot best_count as a bar plot
    ax3 = ax1.twinx()
    color3 = 'tab:red'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Número de rotas (best_count)', color=color3)
    ax3.bar(df['v'], df['best_count'], alpha=0.2, color=color3, label='Nº de rotas')
    ax3.tick_params(axis='y', labelcolor=color3)

    fig.tight_layout()
    plt.title('Convergência do Controlador ACO')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Convergência plotada e salva em {output_path}")
