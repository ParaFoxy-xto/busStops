# src/rota_aco/data/preprocess.py

"""
Funções para carregamento e pré-processamento de dados do grafo.

Este módulo é responsável por interagir com os arquivos de dados brutos
(como GraphML) e extrair informações básicas necessárias para as etapas
seguintes do pipeline, como a lista de nós que são paradas de ônibus.
"""

from typing import Any, List

import networkx as nx
import osmnx as ox

def load_graph(graph_path: str) -> nx.MultiDiGraph:
    """
    Carrega um grafo a partir de um arquivo GraphML.

    Args:
        graph_path: O caminho para o arquivo .graphml.

    Returns:
        Um objeto de grafo NetworkX (MultiDiGraph).
    
    Raises:
        FileNotFoundError: Se o arquivo especificado não for encontrado.
    """
    try:
        graph = ox.load_graphml(graph_path)
        print(f"Grafo '{graph_path}' carregado com sucesso. Nós: {len(graph.nodes)}, Arestas: {len(graph.edges)}")
        return graph
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de grafo não encontrado em: '{graph_path}'")
        raise

def get_bus_stops(graph: nx.MultiDiGraph) -> List[Any]:
    """
    Extrai todos os nós do grafo que são designados como paradas de ônibus.

    A verificação é feita pelo atributo 'bus_stop' com valor 'true'.

    Args:
        graph: O grafo da rede de ruas.

    Returns:
        Uma lista contendo os IDs dos nós das paradas de ônibus.
    """
    bus_stops = [
        node for node, data in graph.nodes(data=True)
        if str(data.get('bus_stop', '')).strip().lower() == 'true'
    ]
    
    if not bus_stops:
        print("[AVISO] Nenhuma parada de ônibus com o atributo 'bus_stop=true' foi encontrada no grafo.")
    
    return bus_stops