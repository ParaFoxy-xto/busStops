import osmnx as ox
import networkx as nx
import folium
from shapely.geometry import Point, LineString

def insert_bus_stop(G, bus_point, stop_name, new_node_id):
    """
    Inserts a bus stop as a separate node at its original position.
    Connects it to the nearest street edge without moving the stop.
    """
    # Ensure all nodes have coordinates
    for node in list(G.nodes):
        if "x" not in G.nodes[node] or "y" not in G.nodes[node]:
            continue  # Skip nodes without coordinates

    # Find the nearest edge
    u, v, key = ox.nearest_edges(G, bus_point.x, bus_point.y)

    # Add the bus stop at its original coordinates
    G.add_node(new_node_id, x=bus_point.x, y=bus_point.y, bus_stop=True, name=stop_name)

    # Compute the direct distance to the nearest edge's midpoint
    midpoint_x = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    midpoint_y = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    dist_to_edge = ox.distance.great_circle(bus_point.y, bus_point.x, midpoint_y, midpoint_x)

    # Connect bus stop to nearest edge nodes (keeping original street directionality)
    G.add_edge(new_node_id, u, length=dist_to_edge)
    G.add_edge(new_node_id, v, length=dist_to_edge)


def main():
    # Define bounding box coordinates
    bbox = (-47.881497, -15.776747, -47.847927, -15.743726)
    #bbox= (-47.881597, -15.776847, -47.847827, -15.743626) #bbox expandida
    bboxCO=( -47.863784,-15.766731,-47.855523,-15.759917)
    # Get the street network as a directed graph
    # Get the main street network as a directed graph (drivable roads only)
    G = ox.graph_from_bbox(bbox, network_type="drive", simplify=False)
    
    # Get service roads from bboxCO
    G_service = ox.graph_from_bbox(bboxCO, network_type="all", simplify=False)

    # Step 1: Identify only service roads and their connected nodes
    service_edges = [(u, v, key, data) for u, v, key, data in G_service.edges(keys=True, data=True)
                     if data.get("highway") == "service"]

    # Step 2: Collect the nodes connected to service roads
    service_nodes = set()
    for u, v, key, data in service_edges:
        service_nodes.add(u)
        service_nodes.add(v)

    # Step 3: Add only the nodes that are part of service roads
    for node, data in G_service.nodes(data=True):
        if node in service_nodes and "x" in data and "y" in data:
            G.add_node(node, **data)  # Add only necessary nodes

    # Step 4: Add only the service road edges
    for u, v, key, data in service_edges:
        G.add_edge(u, v, key, **data)

    # Get bus stops from OpenStreetMap
    tags = {'highway': 'bus_stop'}
    bus_stops = ox.features_from_bbox(bbox, tags)
    bus_stops = bus_stops[bus_stops.geometry.type == 'Point']
    
    # Start bus stop node IDs from a new unique value
    new_node_id = max(G.nodes) + 1

    # Process each bus stop
    for idx, row in bus_stops.iterrows():
        bus_point = row.geometry
        stop_name = row.get("name", f"Stop {idx}")  # Get stop name or assign default
        
        # Insert bus stop without moving its position
        insert_bus_stop(G, bus_point, stop_name, new_node_id)

        new_node_id += 1
        
    # Step 1: Identify edges connected to bus stops, ensure they are two-way, and remove loops
    for node, data in G.nodes(data=True):
        if data.get("bus_stop", False):  # If the node is a bus stop
            for neighbor in list(G.neighbors(node)):  # Use list() to avoid modification issues
                for key in list(G[node][neighbor]):  # Iterate over all possible keys
                    # Remove self-loop edges (bus stop connected to itself)
                    if node == neighbor:
                        G.remove_edge(node, neighbor, key)
                        continue  # Skip to next edge

                    # Set bus stop roads to "bus_access" and make them two-way
                    G.edges[node, neighbor, key]["highway"] = "bus_access"
                    G.edges[node, neighbor, key]["oneway"] = False  # Ensure it is undirected
                    
                    # Check if reverse edge exists, if not, add it
                    if not G.has_edge(neighbor, node, key):  
                        G.add_edge(neighbor, node, key, **G.edges[node, neighbor, key])




    # ** Folium Interactive Map Visualization **
    
    # Get the center of the map
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2

    # Create a Folium map centered on the area
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")

    # Add bus stops as red markers with names
    for node, data in G.nodes(data=True):
        location = [data["y"], data["x"]]
        if data.get("bus_stop", False):
            stop_name = data.get("name", f"Bus Stop {node}")
            folium.Marker(
                location=[data["y"], data["x"]],
                popup=stop_name,
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)
        else:
            # Regular street node (blue)
            folium.CircleMarker(
                location=location,
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
                popup=f"Street Node {node}",
            ).add_to(m)

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
        "bus_access": "pink",#Acesso as paradas
        "default": "black"  # Default for unknown types
    }

    # Add streets as lines with color-coded types
    for u, v, key, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway", "default")  # Get highway type
        
        # Some roads have multiple tags, take the first one
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        color = road_colors.get(highway_type, road_colors["default"])  # Assign color

        # If edge has geometry, use it; otherwise, create a simple line
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]

        # Add the edge with a tooltip showing its type
        folium.PolyLine(
            coords, color=color, weight=4, opacity=0.8,
            tooltip=f"Edge {u} → {v} | Type: {highway_type}"
        ).add_to(m)
    # Modify highway type for bus-access roads
    for u, v, key, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway", "default")  # Get the current type
    
    # If it's an unknown type or bus-related, rename it to "bus_access"
    if highway_type == "default" or "bus" in str(highway_type).lower():
        G.edges[u, v, key]["highway"] = "bus_access"


    # Save and display the map
    m.save("streets.html")    
    ox.save_graphml(G, filepath="streets.graphml")
    print("✅ Interactive map saved as 'streets.html' - Open it in a browser! and graph saved as 'streets.graphml'")


if __name__ == '__main__':
    main()
