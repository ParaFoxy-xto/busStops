import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import deque

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def get_nearest_node(G, coord):
    nearest_node = None
    min_dist = float('inf')
    for node, data in G.nodes(data=True):
        try:
            x = float(data.get('x', data.get('lon')))
            y = float(data.get('y', data.get('lat')))
        except (TypeError, ValueError):
            continue
        node_coord = (y, x)  # (lat, lon)
        dist = euclidean_distance(coord, node_coord)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def get_node_coord(node, data):
    try:
        x = float(data.get('x', data.get('lon')))
        y = float(data.get('y', data.get('lat')))
        return (x, y)
    except (TypeError, ValueError):
        return (0, 0)

# ----- Preprocessing & Setup -----

# Set the start coordinate.
start_coords = (-15.77614, -47.871507)
# (The exit coordinate is not used in this traversal.)
# Load the graph from file.
G = nx.read_graphml("bus_network.graphml")

# Preprocess: For every directed edge with highway=="bus_access" (d11 attribute) that leads to a bus stop,
# add the reverse edge if not present—making these edges effectively undirected.
if G.is_directed():
    for u, v, data in list(G.edges(data=True)):
        if data.get('highway', "").lower() == "bus_access" and \
           G.nodes[v].get('bus_stop', "false").lower() == "true":
            if not G.has_edge(v, u):
                G.add_edge(v, u, **data)

# Determine the start node using the coordinate.
start_node = get_nearest_node(G, start_coords)
if start_node is None:
    print("No start node found.")
    exit()

# ----- BFS Traversal with Bus_Access Priority -----
# We will perform a BFS starting from start_node.
# For each visited node we record the "access type":
#   - "bus_access" if the edge used to reach it had highway=="bus_access"
#   - "normal" otherwise.
# The start node is labeled as "normal".
access_type = {start_node: "normal"}
visited = set([start_node])
queue = deque([start_node])

while queue:
    current = queue.popleft()
    for neighbor in G.neighbors(current):
        # If already visited, skip.
        if neighbor in visited:
            continue

        # Check if the edge from current to neighbor is a bus_access edge.
        data = G.get_edge_data(current, neighbor)
        # We assume the attribute is stored with key "highway".
        if data and data.get('highway', "").lower() == "bus_access" and \
           G.nodes[neighbor].get('bus_stop', "false").lower() == "true":
            # HARD CONSTRAINT: If a bus_access edge is found, force access.
            access_type[neighbor] = "bus_access"
        else:
            access_type[neighbor] = "normal"
        visited.add(neighbor)
        queue.append(neighbor)

# ----- Assign Colors for Plotting -----
# Nodes not visited will be colored red.
node_color_map = {}
for node in G.nodes():
    if node not in visited:
        node_color_map[node] = 'red'
    else:
        # If accessed via a bus_access edge, color green; otherwise blue.
        if access_type.get(node, "normal") == "bus_access":
            node_color_map[node] = 'green'
        else:
            node_color_map[node] = 'blue'

# For edges, we can color them based on the colors of their endpoints.
# Here we will color an edge red if either endpoint was not visited; otherwise gray.
edge_color_map = {}
for u, v in G.edges():
    if u not in visited or v not in visited:
        edge_color_map[(u, v)] = 'red'
    else:
        edge_color_map[(u, v)] = 'gray'

# ----- Build Position Dictionary using Real Map Coordinates -----
pos = {}
for node, data in G.nodes(data=True):
    pos[node] = get_node_coord(node, data)

# ----- Plotting the Graph -----
plt.figure(figsize=(12,8))
# Nodes: color by node_color_map.
nx.draw_networkx_nodes(G, pos, node_color=[node_color_map[node] for node in G.nodes()], node_size=50)
# Edges: color by edge_color_map.
nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color=[edge_color_map[edge] for edge in G.edges()], alpha=0.5)
plt.title("Graph Traversal Starting at {}:\nGreen = accessed via bus_access; Blue = normal; Red = not accessed".format(start_node))
plt.axis("off")
plt.show()
