# src/rota_aco/aco/ant.py

import random
import networkx as nx
from typing import Any, Dict, Set, List

class Ant:
    def __init__(
        self,
        start_node,
        stops,
        opposites_dict,
        combined_opposites,
        alpha=1.0,
        beta=2.0
    ):
        """
        start_node: nó inicial
        stops: lista de nós a cobrir
        opposites_dict: filtro de opostos (proximidade/acesso) usado em LS, etc.
        combined_opposites: dicionário u -> [v1, v2…] de saltos proibidos
        alpha, beta: parâmetros de feromônio e heurística
        """
        self.start_node = start_node
        self.current = start_node
        self.stops = list(stops)
        self.opposites = opposites_dict
        self.combined_opposites = combined_opposites
        self.alpha = alpha
        self.beta = beta
        self.caminho = [start_node]
        self.distancia_total = 0.0

    def construct_solution(
        self,
        graph: nx.DiGraph,
        pheromones: Dict[tuple, float]
    ) -> None:
        """
        Constrói a solução no grafo compacto de rotas candidatas.
        Usa fórmula de probabilidade baseada em feromônio e heurística (1/distância).
        """
        current = self.start
        remaining = set(self.paradas) - {current}
        while remaining:
            next_node = self._choose_next(graph, pheromones, current, remaining)
            if next_node is None:
                break
            self.caminho.append(next_node)
            remaining.remove(next_node)
            current = next_node

    def _choose_next(self, graph, pheromones):
        # vizinhos possíveis
        neighbors = list(graph.successors(self.current))
        # remove saltos proibidos por opostos
        neighbors = [
            n for n in neighbors
            if n not in self.combined_opposites.get(self.current, [])
        ]
        if not neighbors:
            return None

        # calcula probabilidade combinada de feromônio^alpha * (1/dist)^beta
        probs = []
        for n in neighbors:
            tau = pheromones.get((self.current, n), 1e-6) ** self.alpha
            try:
                eta = (1.0 / graph[self.current][n]['weight']) ** self.beta
            except KeyError:
                eta = 0.0
            probs.append(tau * eta)

        total = sum(probs)
        if total == 0:
            return random.choice(neighbors)
        # roleta
        r = random.random()
        cum = 0.0
        for n, p in zip(neighbors, probs):
            cum += p / total
            if r <= cum:
                return n
        return neighbors[-1]