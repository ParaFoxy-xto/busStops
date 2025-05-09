import networkx as nx
import math
import random
import matplotlib.pyplot as plt

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def get_nearest_node(G, coord):
    nearest_node = None
    min_dist = float('inf')
    for node, data in G.nodes(data=True):
        try:
            x = float(data.get('x', data.get('lon')))
            y = float(data.get('y', data.get('lat')))
        except:
            continue
        node_coord = (y, x)  # (lat, lon)
        dist = euclidean_distance(coord, node_coord)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def get_node_coord(node):
    data = G.nodes[node]
    try:
        x = float(data.get('x', data.get('lon')))
        y = float(data.get('y', data.get('lat')))
        return (x, y)
    except:
        return None

def branch_leads_to_exit(node, visited=None, depth=0, max_depth=10):
    if node == exit_node:
        return True
    if depth >= max_depth:
        return False
    if visited is None:
        visited = set()
    visited.add(node)
    for nb in G.neighbors(node):
        if nb in visited:
            continue
        if branch_leads_to_exit(nb, visited.copy(), depth+1, max_depth):
            return True
    return False

def is_dead_end(node, max_depth=10):
    if node == exit_node or node == start_node:
        return False
    return not branch_leads_to_exit(node, max_depth=max_depth)

# PARAMETERS
start_coords = (-15.77614, -47.871507)
exit_coords  = (-15.760521, -47.8741429)

# Load graph from file
G = nx.read_graphml("pequeno.graphml")
start_node = get_nearest_node(G, start_coords)
exit_node  = get_nearest_node(G, exit_coords)
#print("Start node:", start_node)
#print("Exit node:", exit_node)

# --- Pre-process: For every edge with highway=="bus_access" that leads to a bus stop, make it undirected.
if G.is_directed():
    for u, v, data in list(G.edges(data=True)):
        if data.get('highway', "").lower() == "bus_access" and \
           G.nodes[v].get('bus_stop', "false").lower() == "true":
            if not G.has_edge(v, u):
                G.add_edge(v, u, **data)
# --------------------------------------------------------------

# ACO PARAMETERS
num_ants = 1
max_iterations = 10
alpha = 1.0         # pheromone importance
beta = 2.0          # heuristic importance (inverse distance)
evaporation_rate = 0.5
pheromone_initial = 1.0
bonus_factor = 2.0  # bonus multiplier for passenger pickup

# BUS CAPACITY PARAMETERS
bus_capacity = 75
stop_passengers = 10  # each bus stop always offers 10 passengers

# Initialize pheromones on each edge.
pheromones = {}
for u, v, data in G.edges(data=True):
    initial_val = pheromone_initial
    if G.nodes[v].get('bus_stop', "false").lower() == "true":
        initial_val *= 2.0
    pheromones[(u, v)] = initial_val
    if not G.is_directed():
        initial_val_rev = pheromone_initial
        if G.nodes[u].get('bus_stop', "false").lower() == "true":
            initial_val_rev *= 2.0
        pheromones[(v, u)] = initial_val_rev

def heuristic(u, v):
    coord_u = get_node_coord(u)
    coord_v = get_node_coord(v)
    if coord_u is None or coord_v is None:
        return 1.0
    return 1.0 / (euclidean_distance(coord_u, coord_v) + 1e-6)

def exit_heuristic(current, candidate):
    coord_current = get_node_coord(current)
    coord_candidate = get_node_coord(candidate)
    coord_exit = get_node_coord(exit_node)
    if not (coord_current and coord_candidate and coord_exit):
        return 1.0
    d_current = euclidean_distance(coord_current, coord_exit)
    d_candidate = euclidean_distance(coord_candidate, coord_exit)
    if d_candidate < d_current:
        return 1 + (d_current - d_candidate)/(d_current + 1e-6)
    else:
        return 0.5

def construct_route():
    current_node = start_node
    route = [current_node]
    bus_load = 0
    stop_access_direction = {}  # to record bus_access edge usage

    while current_node != exit_node:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        if exit_node in neighbors:
            route.append(exit_node)
            break

        # HARD CONSTRAINT: Check for bus_access candidates.
        bus_access_candidates = []
        for neighbor in neighbors:
            data = G.get_edge_data(current_node, neighbor)
            if data and data.get('highway', "").lower() == "bus_access" and \
               G.nodes[neighbor].get('bus_stop', "false").lower() == "true":
                bus_access_candidates.append(neighbor)
        if bus_access_candidates:
            # Force selection: choose candidate with highest pheromone value.
            next_node = max(bus_access_candidates, key=lambda n: pheromones.get((current_node, n), pheromone_initial))
            stop_access_direction[next_node] = (current_node, next_node)
            route.append(next_node)
            current_node = next_node
            if G.nodes[next_node].get('bus_stop', "false").lower() == "true":
                pickup = min(stop_passengers, bus_capacity - bus_load)
                if pickup > 0:
                    bonus_deposit = bonus_factor * pickup
                    edge = (route[-2], next_node)
                    pheromones[edge] = pheromones.get(edge, pheromone_initial) + bonus_deposit
                    bus_load += pickup
            continue

        # Otherwise, compute probabilities for all neighbors.
        probabilities = []
        for neighbor in neighbors:
            edge = (current_node, neighbor)
            pheromone = pheromones.get(edge, pheromone_initial)
            h_val = heuristic(current_node, neighbor)
            data = G.get_edge_data(current_node, neighbor)
            if data and data.get('highway', "").lower() == "bus_access":
                pheromone *= 1.5
            if len(route) >= 2:
                prev_node = route[-2]
                coord_prev = get_node_coord(prev_node)
                coord_current = get_node_coord(current_node)
                coord_candidate = get_node_coord(neighbor)
                if coord_prev and coord_current and coord_candidate:
                    vec1 = (coord_current[0]-coord_prev[0], coord_current[1]-coord_prev[1])
                    vec2 = (coord_candidate[0]-coord_current[0], coord_candidate[1]-coord_current[1])
                    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                    oneway_val = data.get('oneway', "true") if data else "true"
                    if oneway_val.lower() == "true" and dot < 0:
                        continue
            if neighbor in route:
                dead_end_factor = 0.001
                local_factor = 0.1
            else:
                available_neighbors = [n for n in G.neighbors(neighbor) if n not in route]
                if neighbor != exit_node and len(available_neighbors) == 0:
                    dead_end_factor = 0.001
                else:
                    dead_end_factor = 1.0
                local_factor = 1.0
            exit_fact = exit_heuristic(current_node, neighbor)
            prob = (pheromone**alpha) * (h_val**beta) * exit_fact * local_factor * dead_end_factor
            probabilities.append((neighbor, prob))
        if not probabilities:
            break
        total = sum(p for n, p in probabilities)
        if total == 0:
            probabilities = [(n, 1.0/len(probabilities)) for n, p in probabilities]
        else:
            probabilities = [(n, p/total) for n, p in probabilities]
        r = random.random()
        cumulative = 0.0
        next_node = None
        for n, p in probabilities:
            cumulative += p
            if r <= cumulative:
                next_node = n
                break
        if next_node is None:
            break
        if G.nodes[next_node].get('bus_stop', "false").lower() == "true":
            pickup = min(stop_passengers, bus_capacity - bus_load)
            if pickup > 0:
                bonus_deposit = bonus_factor * pickup
                edge = (current_node, next_node)
                pheromones[edge] = pheromones.get(edge, pheromone_initial) + bonus_deposit
                bus_load += pickup
            stop_access_direction[next_node] = (current_node, next_node)
        route.append(next_node)
        current_node = next_node
        if len(route) > len(G.nodes()):
            break
    return route

def validate_route(route, stop_access_direction):
    if not route or route[0]!=start_node or route[-1]!=exit_node:
        return False
    if len(route) != len(set(route)):
        return False
    for i in range(1, len(route)-1):
        if route[i-1]==route[i+1]:
            return False
    return True

def update_pheromones_intelligent(routes_found):
    for edge in pheromones:
        pheromones[edge] *= (1-evaporation_rate)
    for route, cost in routes_found:
        if len(route) < 2:
            continue
        deposit = 1.0/(cost+1e-6)
        if route[-1] == exit_node and validate_route(route, {}):
            for i in range(len(route)-1):
                edge = (route[i], route[i+1])
                pheromones[edge] += deposit
        else:
            for i in range(len(route)-1):
                edge = (route[i], route[i+1])
                pheromones[edge] *= 0.001

def calculate_cost(route):
    cost = 0.0
    for i in range(len(route)-1):
        cost += 1.0/heuristic(route[i], route[i+1])
    return cost

all_attempted_routes = []
best_route = None
best_cost = float('inf')

for iteration in range(max_iterations):
    routes_found = []
    route = construct_route()
    all_attempted_routes.append(route)
    cost = calculate_cost(route)
    routes_found.append((route, cost))
    if validate_route(route, {}) and cost < best_cost:
        best_cost = cost
        best_route = route
    update_pheromones_intelligent(routes_found)

if best_route:
    print("\nBest VALID route found:", best_route)
    attended = [node for node in best_route if G.nodes[node].get('bus_stop','false').lower()=="true"]
    num_attended = len(attended)
    all_bus_stops = [node for node, data in G.nodes(data=True) if data.get('bus_stop','false').lower()=="true"]
    num_total = len(all_bus_stops)
    not_attended = num_total - num_attended
    print("Bus stops accessed in best route:", num_attended)
    print("Bus stops not accessed:", not_attended)
else:
    print("\nNo VALID route found.")

def plot_graph(G, attempted_routes, valid_route=None):
    pos = {}
    for node, data in G.nodes(data=True):
        try:
            x = float(data.get('x', data.get('lon')))
            y = float(data.get('y', data.get('lat')))
            pos[node] = (x, y)
        except:
            continue
    plt.figure(figsize=(12,8))
    node_colors = []
    for node in G.nodes():
        if node == start_node or node == exit_node:
            node_colors.append('green')
        elif G.nodes[node].get('bus_stop','false').lower()=="true":
            node_colors.append('red')
        else:
            node_colors.append('blue')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    color_list = ['red','blue','orange','purple','cyan','magenta','yellow','brown','pink','olive']
    for i, route in enumerate(all_attempted_routes):
        if len(route)>1:
            route_edges = [(route[j], route[j+1]) for j in range(len(route)-1)]
            color = color_list[i % len(color_list)]
            nx.draw_networkx_nodes(G, pos, nodelist=route, node_color=color, node_size=40, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=color, width=2, alpha=0.8)
    if valid_route and len(valid_route)>1:
        valid_edges = [(valid_route[i], valid_route[i+1]) for i in range(len(valid_route)-1)]
        nx.draw_networkx_nodes(G, pos, nodelist=valid_route, node_color='orange', node_size=40)
        nx.draw_networkx_edges(G, pos, edgelist=valid_edges, edge_color='orange', width=4)
    plt.title("Bus Network:\nStart/Exit in Green, Bus Stops in Red, Others in Blue\nEach Attempted Route in a Different Color")
    plt.axis('off')
    plt.show()

plot_graph(G, all_attempted_routes, best_route)
