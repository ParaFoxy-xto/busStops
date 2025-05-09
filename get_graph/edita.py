import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import os

class GraphEditor:
    def __init__(self, graph):
        """Inicializa o editor de grafo."""
        self.graph = graph
        self.pos = {node: (float(data['x']), float(data['y'])) for node, data in graph.nodes(data=True) if 'x' in data and 'y' in data}
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.rect = None
        self.start_point = None
        self.zoom_level = 1.0
        self.base_limits = None

        
    def plot_graph(self):
        """Plota o grafo preservando a posição geográfica."""
        self.ax.clear()
        valid_edges = [(u, v) for u, v in self.graph.edges() if u in self.pos and v in self.pos]
        node_colors = ['red' if self.graph.nodes[n].get('bus_stop') else 'blue' for n in self.graph.nodes()]
        ox.plot_graph(self.graph, ax=self.ax, node_color=node_colors, edge_color='gray', node_size=10, bgcolor='white', show=False, close=False)
        self.ax.set_title("Use Shift + Seleção para deletar; Scroll para zoom")
        

        # Obter os valores de x e y das posições dos nós
        x_vals = [pos[0] for pos in self.pos.values()]
        y_vals = [pos[1] for pos in self.pos.values()]

        if x_vals and y_vals:  # Certifica-se de que há nós no grafo antes de definir os limites
            margin = 0.001  # Adiciona uma pequena margem ao redor do grafo
            self.ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
            self.ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

            # Adicionar um contorno ao redor do grafo
            border = patches.Rectangle(
                (min(x_vals) - margin, min(y_vals) - margin),
                (max(x_vals) - min(x_vals)) + 2 * margin,
                (max(y_vals) - min(y_vals)) + 2 * margin,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            self.ax.add_patch(border)

        # Ativar a grade para referência visual
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # Alterar a cor de fundo para melhorar a visibilidade
        self.ax.set_facecolor("lightgray")
        plt.draw()


    def on_press(self, event):
        """Inicia a seleção ao pressionar o botão do mouse."""
        if event.xdata is None or event.ydata is None:
            return
        self.start_point = (event.xdata, event.ydata)
        self.rect = Rectangle(self.start_point, 0, 0, linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
        self.ax.add_patch(self.rect)
        plt.draw()

    def on_motion(self, event):
        """Atualiza o retângulo de seleção ao mover o mouse."""
        if self.start_point is None or event.xdata is None or event.ydata is None:
            return
        width = event.xdata - self.start_point[0]
        height = event.ydata - self.start_point[1]
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.rect.set_xy(self.start_point)
        plt.draw()

    def on_release(self, event):
        """Finaliza a seleção ao soltar o botão do mouse."""
        if self.start_point is None or self.rect is None:
            return

        if not event.key or event.key.lower() != "shift":
            print("Nenhum nó removido. Pressione Shift para deletar.")
            self.start_point, self.rect = None, None
            return

        x_min, x_max = sorted([self.start_point[0], event.xdata])
        y_min, y_max = sorted([self.start_point[1], event.ydata])

        nodes_to_remove = [
            node for node, (x, y) in self.pos.items()
            if x_min <= x <= x_max and y_min <= y <= y_max
        ]
        self.graph.remove_nodes_from(nodes_to_remove)

        print(f"Nós removidos: {nodes_to_remove}")

        self.start_point, self.rect = None, None
        self.plot_graph()

    def on_scroll(self, event):
        """Controla o zoom usando a rolagem do mouse."""
        if event.button == 'up':  # Zoom in
            self.zoom_level /= 1.2
        elif event.button == 'down':  # Zoom out
            self.zoom_level *= 1.2
        self.plot_graph()

    def start(self):
        """Inicia o editor interativo."""
        self.plot_graph()
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()


def main():
    input_graphml = 'graphml\grafo.graphml'
    output_graphml = "graphml\ normal.graphml"

    if not os.path.exists(input_graphml):
        print(f"Erro: Arquivo '{input_graphml}' não encontrado.")
        return

    graph = ox.load_graphml(input_graphml)
    editor = GraphEditor(graph)
    editor.start()
    ox.save_graphml(graph, output_graphml)
    print(f"Grafo salvo em '{output_graphml}'.")


if __name__ == "__main__":
    main()