import osmnx as ox
import networkx as nx

# Altere o caminho para o seu arquivo .graphml
graph_path = r"tcc_estagio\pequeno.graphml"

# Carrega o grafo
grafo = ox.load_graphml(graph_path)

print(f"Grafo carregado com {len(grafo.nodes)} nós e {len(grafo.edges)} arestas.")

# Verifica se o grafo é fortemente conexo (para grafos dirigidos)
if nx.is_strongly_connected(grafo):
    print("O grafo é fortemente conexo.")
else:
    num_strong = nx.number_strongly_connected_components(grafo)
    print(f"O grafo NÃO é fortemente conexo. Número de componentes fortemente conexos: {num_strong}")

# Verifica a conectividade fraca (ignorando a direção)
if nx.is_weakly_connected(grafo):
    print("O grafo é fracamente conexo.")
else:
    num_weak = nx.number_weakly_connected_components(grafo)
    print(f"O grafo NÃO é fracamente conexo. Número de componentes fracamente conexos: {num_weak}")

# Listar os tamanhos dos componentes fortemente conexos
strong_components = list(nx.strongly_connected_components(grafo))
print("Tamanhos dos componentes fortemente conexos:")
for comp in strong_components:
    print(len(comp))
# Extrair a maior componente fortemente conexa
componentes = list(nx.strongly_connected_components(grafo))
maior_componente = max(componentes, key=len)
grafo_maior = grafo.subgraph(maior_componente).copy()
print(f"Grafo da maior componente possui {len(grafo_maior.nodes)} nós e {len(grafo_maior.edges)} arestas.")
