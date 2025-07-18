# src/rota_aco/graph/loader.py

"""
Carregador de grafos para diferentes formatos.
"""

import networkx as nx
from typing import Dict, Tuple, Any
import xml.etree.ElementTree as ET


class GraphLoader:
    """
    Classe para carregar grafos de diferentes formatos.
    """
    
    def load_graph(self, file_path: str) -> Tuple[nx.DiGraph, Dict[Tuple[Any, Any], dict]]:
        """
        Carrega um grafo de um arquivo.
        
        Args:
            file_path: Caminho para o arquivo do grafo
            
        Returns:
            Tupla contendo (grafo, meta_edges)
        """
        if file_path.endswith('.graphml'):
            return self._load_graphml(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {file_path}")
    
    def _load_graphml(self, file_path: str) -> Tuple[nx.DiGraph, Dict[Tuple[Any, Any], dict]]:
        """
        Carrega um grafo do formato GraphML.
        """
        try:
            # Carregar grafo usando NetworkX
            graph = nx.read_graphml(file_path)
            
            # Converter para DiGraph se necessário
            if not isinstance(graph, nx.DiGraph):
                graph = graph.to_directed()
            
            # Construir meta_edges
            meta_edges = {}
            
            for u, v, data in graph.edges(data=True):
                edge_key = (u, v)
                
                # Extrair informações da aresta
                edge_info = {
                    'time': float(data.get('weight', data.get('time', data.get('length', 1.0)))),
                    'distance': float(data.get('distance', data.get('length', 1.0))),
                }
                
                # Adicionar outros atributos se existirem
                for key, value in data.items():
                    if key not in ['time', 'distance', 'weight', 'length']:
                        try:
                            edge_info[key] = float(value)
                        except (ValueError, TypeError):
                            edge_info[key] = value
                
                meta_edges[edge_key] = edge_info
            
            return graph, meta_edges
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar GraphML {file_path}: {e}")
    
    def create_simple_test_graph(self) -> Tuple[nx.DiGraph, Dict[Tuple[Any, Any], dict]]:
        """
        Cria um grafo simples para testes.
        """
        graph = nx.DiGraph()
        
        # Adicionar nós
        nodes = ['A', 'B', 'C', 'D', 'E', 'F']
        for node in nodes:
            graph.add_node(node)
        
        # Adicionar arestas com pesos
        edges = [
            ('A', 'B', 5.0),
            ('A', 'C', 3.0),
            ('B', 'C', 2.0),
            ('B', 'D', 4.0),
            ('C', 'D', 6.0),
            ('C', 'E', 3.0),
            ('D', 'E', 2.0),
            ('D', 'F', 4.0),
            ('E', 'F', 1.0),
            # Adicionar algumas arestas reversas
            ('B', 'A', 5.0),
            ('C', 'A', 3.0),
            ('C', 'B', 2.0),
            ('D', 'B', 4.0),
            ('E', 'C', 3.0),
            ('E', 'D', 2.0),
            ('F', 'E', 1.0),
        ]
        
        meta_edges = {}
        
        for u, v, weight in edges:
            graph.add_edge(u, v, weight=weight, time=weight)
            meta_edges[(u, v)] = {
                'time': weight,
                'distance': weight,
                'weight': weight
            }
        
        return graph, meta_edges