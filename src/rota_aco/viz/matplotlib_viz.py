# src/rota_aco/viz/matplotlib_viz.py

"""
Funções para gerar visualizações de grafos e rotas estáticas usando Matplotlib.
"""

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx

# --- Funções Auxiliares de Plotagem (Lógica Comum) ---

def _setup_plot(title: str) -> tuple:
    """Configura a figura e os eixos básicos para uma plotagem."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    return fig, ax

def _draw_nodes(ax, graph: nx.Graph, nodes: List[Any], **kwargs):
    """Desenha um conjunto de nós no eixo com coordenadas do grafo."""
    if not nodes: return
    coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in nodes if n in graph.nodes]
    if not coords: return
    xs, ys = zip(*coords)
    ax.scatter(xs, ys, **kwargs)

def _draw_path(ax, graph: nx.Graph, path: List[Any], **kwargs):
    """Desenha um caminho (rota) no eixo."""
    if not path or len(path) < 2: return
    coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in path if n in graph.nodes]
    if len(coords) < 2: return
    xs, ys = zip(*coords)
    ax.plot(xs, ys, **kwargs)

def _highlight_start_exit(ax, graph: nx.Graph, start_node: Any, exit_node: Any):
    """Destaca os nós de início e fim no mapa."""
    if start_node and start_node in graph.nodes:
        _draw_nodes(ax, graph, [start_node], c='green', s=150, marker='o', label='Partida', zorder=5)
    if exit_node and exit_node in graph.nodes:
        _draw_nodes(ax, graph, [exit_node], c='red', s=150, marker='X', label='Chegada', zorder=5)

def _save_plot(fig, output_path: Optional[str]):
    """Salva a figura em um arquivo ou a exibe na tela."""
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: '{output_path}'")
    else:
        plt.show()
    plt.close(fig)

# --- Funções Principais de Plotagem ---

def plot_multiple_routes(
    original_graph: nx.MultiDiGraph,
    routes: List[List[Any]],
    all_bus_stops: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None,
) -> None:
    """
    Plota múltiplas rotas expandidas sobre o grafo original.
    """
    fig, ax = _setup_plot(f'Solução Final com {len(routes)} Rota(s)')
    
    # 1. Desenha o fundo do grafo de ruas
    _draw_nodes(ax, original_graph, list(original_graph.nodes()), c='lightgray', s=2, label='Ruas')
    
    # 2. Desenha as paradas de ônibus
    _draw_nodes(ax, original_graph, all_bus_stops, c='darkblue', s=25, label='Paradas', zorder=3)

    # 3. Desenha cada rota
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    for i, route in enumerate(routes):
        _draw_path(ax, original_graph, route, c=colors[i % len(colors)], linewidth=2.5, alpha=0.9, label=f"Rota {i+1}", zorder=4)

    # 4. Destaca início e fim
    _highlight_start_exit(ax, original_graph, start_node, exit_node)

    ax.legend()
    _save_plot(fig, output_path)

def plot_multiple_meta_routes(
    meta_graph: nx.DiGraph,
    meta_routes: List[List[Any]],
    stops_to_visit: List[Any],
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None,
    show_labels: bool = False,
) -> None:
    """
    Plota múltiplas meta-rotas sobre o meta-grafo.
    """
    fig, ax = _setup_plot(f'Meta-Rotas no Meta-Grafo ({len(meta_routes)} rotas)')

    # 1. Desenha as arestas do meta-grafo
    for u, v in meta_graph.edges():
        _draw_path(ax, meta_graph, [u, v], c='lightblue', linewidth=1, zorder=1)
        
    # 2. Desenha os nós do meta-grafo (representantes)
    _draw_nodes(ax, meta_graph, list(meta_graph.nodes()), c='gray', s=30, label='Nós do Meta-Grafo', zorder=2)
    
    # 3. Desenha as paradas que devem ser visitadas
    _draw_nodes(ax, meta_graph, stops_to_visit, c='darkblue', s=40, label='Paradas com Demanda', zorder=3)

    # 4. Desenha cada meta-rota
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    for i, route in enumerate(meta_routes):
        _draw_path(ax, meta_graph, route, c=colors[i % len(colors)], linewidth=2.5, alpha=0.9, label=f"Meta-Rota {i+1}", zorder=4)
        
    # 5. Destaca início e fim
    _highlight_start_exit(ax, meta_graph, start_node, exit_node)
    
    # 6. Adiciona labels se solicitado
    if show_labels:
        for node in meta_graph.nodes():
            x, y = meta_graph.nodes[node]['x'], meta_graph.nodes[node]['y']
            ax.text(x, y + 0.0001, str(node), fontsize=7, ha='center', color='black')

    ax.legend()
    _save_plot(fig, output_path)


def plot_meta_graph(
    meta_graph: nx.DiGraph,
    output_path: str,
    start_node: Optional[Any] = None,
    exit_node: Optional[Any] = None,
    show_labels: bool = False
) -> None:
    """
    Plota a estrutura do meta-grafo: seus nós e arestas.
    """
    fig, ax = _setup_plot('Estrutura do Meta-Grafo Gerado')

    # 1. Desenha as arestas do meta-grafo
    for u, v in meta_graph.edges():
        _draw_path(ax, meta_graph, [u, v], c='lightblue', linewidth=1, zorder=1)
        
    # 2. Desenha os nós do meta-grafo (representantes)
    _draw_nodes(ax, meta_graph, list(meta_graph.nodes()), c='darkblue', s=40, label='Nós do Meta-Grafo', zorder=3)
    
    # 3. Destaca início e fim
    _highlight_start_exit(ax, meta_graph, start_node, exit_node)
    
    # 4. Adiciona labels se solicitado
    if show_labels:
        for node in meta_graph.nodes():
            x, y = meta_graph.nodes[node]['x'], meta_graph.nodes[node]['y']
            ax.text(x, y + 0.0001, str(node), fontsize=7, ha='center', color='black')

    ax.legend()
    _save_plot(fig, output_path)

def plot_path_debug(
    graph: nx.MultiDiGraph,
    path_to_plot: List[Any],
    u_node: Any,
    v_node: Any,
    u_opposites: List[Any],
    output_path: str
):
    """
    Plota um único caminho para depuração, destacando o início, fim e opostos.
    """
    fig, ax = _setup_plot(f'Análise do Caminho: {u_node} -> {v_node}')
    
    # Desenha todos os nós do grafo de fundo
    _draw_nodes(ax, graph, list(graph.nodes()), c='lightgray', s=5, zorder=1)
    
    # Destaca o caminho
    _draw_path(ax, graph, path_to_plot, c='blue', linewidth=3, alpha=0.8, label='Caminho Analisado', zorder=2)
    
    # Destaca os nós no caminho
    _draw_nodes(ax, graph, path_to_plot, c='blue', s=20, zorder=3)
    
    # Destaca início (u) e fim (v) do caminho
    _draw_nodes(ax, graph, [u_node], c='green', s=150, marker='o', label=f'Início (u): {u_node}', zorder=4)
    _draw_nodes(ax, graph, [v_node], c='red', s=150, marker='X', label=f'Fim (v): {v_node}', zorder=4)
    
    # Destaca os opostos do nó de início (u)
    if u_opposites:
        _draw_nodes(ax, graph, u_opposites, c='orange', s=100, marker='s', label=f'Opostos de u', zorder=4)

    ax.legend()
    _save_plot(fig, output_path)