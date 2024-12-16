# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

### NOTE: The following imports are required for analysis of algorithms
# import time
# import tracemalloc
# import random

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df
   
    # Create trip_id to route_id mapping
    trip_to_route = df_trips.groupby('trip_id')['route_id'].first().to_dict()

    # NOTE: uncomment the next few lines to store the dictionary in a text file
    # with open("trip_to_route.txt", "w") as file:
    #     for trip_id, route_ids in trip_to_route.items():
    #         file.write(f"{trip_id}: {route_ids}\n")
    # print("Output has been written to trip_to_route.txt")

    # Map route_id to a list of stops in order of their sequence

    # Ensure each route only has unique stops
    df_route_to_stops = pd.merge(df_trips[['route_id', 'trip_id']], df_stop_times[['trip_id', 'stop_id']], on='trip_id')
    route_to_stops = df_route_to_stops.groupby('route_id')['stop_id'].unique().apply(list).to_dict()
    
    # NOTE: uncomment the next few lines to store the dictionary in a text file
    # with open("route_to_stops.txt", "w") as file:
    #     for route_id, stop_ids in route_to_stops.items():
    #         file.write(f"{route_id}: {stop_ids}\n")
    # print("Output has been written to route_to_stops.txt")

    # Count trips per stop
    stop_trip_count = df_stop_times.groupby('stop_id')['trip_id'].count().to_dict()

    # NOTE: uncomment the next few lines to store the dictionary in a text file
    # with open("stop_trip_count.txt", "w") as file:
    #     for stop_id, num_trips in stop_trip_count.items():
    #         file.write(f"{stop_id}: {num_trips}\n")
    # print("Output has been written to stop_trip_count.txt")

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    # Implementation here
    route_trip_count = {}
    for route in trip_to_route.values():
        if route in route_trip_count:
            route_trip_count[route] += 1
        else:
            route_trip_count[route] = 1
    print(f'Top busiest routes: {sorted(route_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]}')
    return sorted(route_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]
    

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # Implementation here
    print(f'Most frequent stops: {sorted(stop_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]}')
    return sorted(stop_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]


# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # Implementation here
    stop_route_count = {}
    
    for values in route_to_stops.values():
        for num in values:
            if num in stop_route_count:
                stop_route_count[num] += 1
            else:
                stop_route_count[num] = 1

    print(f'Top busiest stops: {sorted(stop_route_count.items(), key=lambda item: item[1], reverse=True)[:5]}')
    return sorted(stop_route_count.items(), key=lambda item: item[1], reverse=True)[:5]


# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # Implementation here
    stop_pairs = {}
    for route, stops in route_to_stops.items():
        for i in range(len(stops)-1):
            pair = (stops[i], stops[i+1])
            if pair in stop_pairs:
                stop_pairs[pair] = (-1, 0)
            else:
                stop_pairs[pair] = (stop_trip_count[stops[i]] + stop_trip_count[stops[i + 1]], route)
    
    stop_pairs_to_route = sorted(stop_pairs.items(), key=lambda item: item[1], reverse=True)[:5]
    return [(item[0], item[1][1]) for item in stop_pairs_to_route]


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Implementation here
    print("Visualizing . . .")
    G = nx.Graph()

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    pos = nx.spring_layout(G, seed=42)

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(size=10, color='skyblue', line_width=2)
    )

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Stop-Route Network Graph",
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.show()
    # fig.write_html("Routes.html")
    # fig.write_image("Routes.png")

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # Implementation here
    direct_routes = []
    for route, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            direct_routes.append(route)
    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Implementation here
    for route, stops in route_to_stops.items():
        for stop in stops:
            +RouteHasStop(route, stop)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # Implementation here
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    return [route[0] for route in DirectRoute(R, start, end)]

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Implementation here
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include) <= DirectRoute(R1, start_stop_id, stop_id_to_include) & DirectRoute(R2, stop_id_to_include, end_stop_id)
    data = (OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include))
    return [(item[0], stop_id_to_include, item[1]) for item in data]

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Implementation here
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    OptimalRoute(R1, R2, X, Y, Z) <= DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y)
    data = (OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include))
    return [(item[1], stop_id_to_include, item[0]) for item in data]

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Implementation here
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    OptimalRoute(R1, R2, X, Y, Z) <= DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y)
    data = (OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include))
    return [(item[0], stop_id_to_include, item[1]) for item in data]

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here

###NOTE: The following code is a part of visualization and testing.

# create_kb()
# add_route_data(route_to_stops)
# stop_list = list(stop_trip_count.keys())

# def get_execution_time_direct_route(func):
#     start_time = time.time()
#     for i in range(500):
#         for j in range(500):
#             start = stop_list[i]
#             end = stop_list[j]
#             func(start, end)
#             print(start, end)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     return execution_time

# def get_memory_usage_direct_route(func):
#     tracemalloc.start()
#     for i in range(500):
#         for j in range(500):
#             start = stop_list[i]
#             end = stop_list[j]
#             func(start, end)
#             print(start, end)
#     _, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return peak

# def get_execution_time_optimal_route(func):
#     start_time = time.time()
#     for i in range(100):
#         for j in range(100):
#             start = stop_list[i]
#             end = stop_list[j]
#             random_include_stop = 300
#             func(start, end, random_include_stop, 1)
#             print(start, end)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     return execution_time

# def get_memory_usage_optimal_route(func):
#     tracemalloc.start()
#     for i in range(100):
#         for j in range(100):
#             start = stop_list[i]
#             end = stop_list[j]
#             random_include_stop = 300
#             func(start, end, random_include_stop, 1)
#             print(start, end)
#     _, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return peak

# execution_time = get_execution_time_direct_route(direct_route_brute_force)
# print(f"Execution time for direct_route_brute_force: {execution_time} seconds")

# execution_time = get_execution_time_direct_route(query_direct_routes)
# print(f"Execution time for query_direct_routes: {execution_time} seconds")

# memory_usage = get_memory_usage_direct_route(direct_route_brute_force)
# print(f"Peak memory usage for direct_route_brute_force: {memory_usage / 10**6} MB")

# memory_usage = get_memory_usage_direct_route(query_direct_routes)
# print(f"Peak memory usage for query_direct_routes: {memory_usage / 10**6} MB")

# execution_time = get_execution_time_optimal_route(forward_chaining)
# print(f"Execution time for forward_chaining: {execution_time} seconds")

# execution_time = get_execution_time_optimal_route(backward_chaining)
# print(f"Execution time for backward_chaining: {execution_time} seconds")

# memory_usage = get_memory_usage_optimal_route(forward_chaining)
# print(f"Peak memory usage for forward_chaining: {memory_usage / 10**6} MB")

# memory_usage = get_memory_usage_optimal_route(backward_chaining)
# print(f"Peak memory usage for backward_chaining: {memory_usage / 10**6} MB")

### NOTE: the function to implement visualization may contain similarity with certain internet sources and documentations due to unfamilairty with the pythongraph libraries
### NOTE: Uncomment create_kb() to obtain the correct visualization
# visualize_stop_route_graph_interactive(route_to_stops)