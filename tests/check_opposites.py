import networkx as nx
from rota_aco.data.opposites import find_opposites_by_access

# Load the graph
G = nx.read_graphml('graphml/pequeno.graphml')
bus_stops = list(G.nodes())
opposites = find_opposites_by_access(G, bus_stops)

# The routes from the last run
route1 = [2705153069, 12529762096, 12529762065, 12529762100, 12529762087, 12529762085, 4826126847]
route2 = [2705153069, 12529762064, 12529762099, 12529762088, 12529762102, 12529762086, 4826126847]

print('Route 1 stops:', route1)
print('Route 2 stops:', route2)
print('Checking for opposite violations...')

violations = []
for stop1 in route1:
    for stop2 in route2:
        if stop2 in opposites.get(stop1, []):
            violations.append((stop1, stop2))
            print(f'OPPOSITE VIOLATION: {stop1} and {stop2} are opposites!')

if not violations:
    print('No opposite violations found!')
else:
    print(f'Found {len(violations)} opposite violations')

# Show some sample opposites
print('\nSample opposites in the graph:')
count = 0
for stop, opps in opposites.items():
    if opps and count < 5:
        print(f'{stop} -> {opps}')
        count += 1 