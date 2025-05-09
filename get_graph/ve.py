import osmnx as ox
import folium
import numpy as np

# Load the saved graph
graph = ox.load_graphml("bus_routes.graphml")

# Identify bus stop nodes
bus_stop_nodes = [node for node, data in graph.nodes(data=True) if data.get("bus_stop", False)]

# Identify edges connected to bus stops
bus_stop_edges = []
for u, v, k, data in graph.edges(keys=True, data=True):
    if u in bus_stop_nodes or v in bus_stop_nodes:
        bus_stop_edges.append((u, v, k))

# Calculate center coordinates
y_coords = [data['y'] for _, data in graph.nodes(data=True)]
x_coords = [data['x'] for _, data in graph.nodes(data=True)]
center_y = np.mean(y_coords)
center_x = np.mean(x_coords)

# Create a folium map
m = folium.Map(location=[center_y, center_x], zoom_start=15, tiles="cartodbpositron")

# Define colors for different road types
road_colors = {
    "motorway": "darkred",
    "primary": "red",
    "secondary": "orange",
    "tertiary": "yellow",
    "residential": "green",
    "service": "lightblue",
    "footway": "purple",
    "cycleway": "blue",
    "path": "gray",
    "busway": "pink",  # Bus-only roads
    "bus_access": "pink",  # Acesso às paradas
    "default": "black"  # Default for unknown types
}

# Plot all edges with color-coded road types
for u, v, k, data in graph.edges(keys=True, data=True):
    # Get road type and assign color
    highway_type = data.get("highway", "default")
    if isinstance(highway_type, list):  
        highway_type = highway_type[0]  # Handle lists

    color = road_colors.get(highway_type, road_colors["default"])

    # Get coordinates
    if 'geometry' in data:
        coords = [(lat, lon) for lon, lat in data['geometry'].coords]
    else:
        coords = [(graph.nodes[u]['y'], graph.nodes[u]['x']), 
                  (graph.nodes[v]['y'], graph.nodes[v]['x'])]

    # Add line to map
    folium.PolyLine(
        coords,
        color=color,
        weight=2 if color == "black" else 3,
        opacity=0.8,
        tooltip=f"Edge {u} → {v} | Type: {highway_type}"
    ).add_to(m)

# Highlight bus stop connected edges in red (on top of colored roads)
for u, v, k in bus_stop_edges:
    if 'geometry' in graph.edges[u, v, k]:
        coords = [(lat, lon) for lon, lat in graph.edges[u, v, k]['geometry'].coords]
    else:
        coords = [(graph.nodes[u]['y'], graph.nodes[u]['x']), 
                  (graph.nodes[v]['y'], graph.nodes[v]['x'])]

    # Overlay red bus stop connections
    folium.PolyLine(
        coords,
        color='pink',
        weight=3,
        opacity=0.9,
        tooltip=f"Bus access: {u} to {v}"
    ).add_to(m)

# Plot regular nodes as small blue circles
for node, data in graph.nodes(data=True):
    if node not in bus_stop_nodes:
        folium.CircleMarker(
            location=[data['y'], data['x']],
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7
        ).add_to(m)

# Plot bus stops as larger red markers
for node in bus_stop_nodes:
    data = graph.nodes[node]
    stop_name = data.get("name", f"Bus Stop {node}")
    
    folium.Marker(
        location=[data['y'], data['x']],
        popup=stop_name,
        tooltip=stop_name,
        icon=folium.Icon(color="red", icon="bus", prefix='fa')
    ).add_to(m)

# Add a legend (as a control)
#legend_html = '''
# <div style="position: fixed; 
#             bottom: 50px; right: 50px; 
#             border:2px solid grey; z-index:9999; 
#             background-color:white;
#             padding: 10px;
#             border-radius: 5px;">
#     <p><i class="fa fa-bus" style="color:red"></i> Bus Stops</p>
#     <p><hr style="border: 2px solid red; width: 50px;"> Bus Routes</p>
#     <p><i style="background-color:blue;border-radius:50%;width:10px;height:10px;display:inline-block;"></i> Street Nodes</p>
#     <p><hr style="border: 1px solid lightgray; width: 50px;"> Streets</p>
#     <p><hr style="border: 2px solid black; width: 50px;"> Default Road</p>
#     <p><hr style="border: 2px solid darkred; width: 50px;"> Motorway</p>
#     <p><hr style="border: 2px solid red; width: 50px;"> Primary Road</p>
#     <p><hr style="border: 2px solid orange; width: 50px;"> Secondary Road</p>
#     <p><hr style="border: 2px solid yellow; width: 50px;"> Tertiary Road</p>
#     <p><hr style="border: 2px solid green; width: 50px;"> Residential Road</p>
#     <p><hr style="border: 2px solid lightblue; width: 50px;"> Service Road</p>
#     <p><hr style="border: 2px solid purple; width: 50px;"> Footway</p>
#     <p><hr style="border: 2px solid blue; width: 50px;"> Cycleway</p>
#     <p><hr style="border: 2px solid gray; width: 50px;"> Path</p>
#     <p><hr style="border: 2px solid pink; width: 50px;"> Bus-Only Roads</p>
# </div>
# '''
#m.get_root().html.add_child(folium.Element(legend_html))

# Save the map
output_file = 'bus_routes.html'
m.save(output_file)
print(f"✅ Map saved to {output_file} - Open it in a browser!")
