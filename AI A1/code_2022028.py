import numpy as np
import pickle
import queue
import heapq
import time
import tracemalloc
import matplotlib.pyplot as plt

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def check_DFS(adj_matrix, node, goal_node, explored):
  if node == goal_node:
    return True
  for adj_node in range(len(adj_matrix[node])):
    if adj_node not in explored and adj_matrix[node][adj_node] != 0:
      explored.append(adj_node)
      if check_DFS(adj_matrix, adj_node, goal_node, explored):
        return True
  return False

def get_ids_path(adj_matrix, start_node, goal_node):

  if(not check_DFS(adj_matrix, start_node, goal_node, [start_node])):
    # return None, 10000
    return None
  
  def recursive_DFS(adj_matrix, node, goal_node, visited, depth):
    if node == goal_node:
      return [node]
    
    if depth == 0:
      return False
    
    visited[node] = True
    for adj_node in range(len(adj_matrix[node])):
      if not visited[adj_node] and adj_matrix[node][adj_node] != 0:
        path = recursive_DFS(adj_matrix, adj_node, goal_node, visited, depth-1)
        if path:
          path.insert(0, node)
          return path
    visited[node] = False
    return None

  num_nodes = len(adj_matrix)
  depth = 0
  visited = [False]*num_nodes
  path = None
  for depth in range(num_nodes):
    path = recursive_DFS(adj_matrix, start_node, goal_node, visited, depth)
    if path:
      break
  cost = 0
  for i in range(len(path)-1):
    cost += adj_matrix[path[i]][path[i+1]]
  # return path, cost
  return path

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
  num_nodes = len(adj_matrix)
  q1 = queue.Queue()
  q2 = queue.Queue()
  q1.put(start_node)
  q2.put(goal_node)
  intersection_node = -1

  parent_f = [-1]*num_nodes
  parent_b = [-1]*num_nodes
  explored_f = [start_node]
  explored_b = [goal_node]

  explored_b = [goal_node]
  rev_adj_matrix = np.transpose(adj_matrix)
  
  while(not q1.empty() and not q2.empty()):
    node_f = q1.get()
    if(node_f in explored_b):
      intersection_node = node_f
      break

    for adj_node in range(len(adj_matrix[node_f])):
      if adj_node not in explored_f and adj_matrix[node_f][adj_node] != 0:
        parent_f[adj_node] = node_f
        q1.put(adj_node)
        explored_f.append(adj_node)
    
    node_b = q2.get()
    if(node_b in explored_f):
      intersection_node = node_b
      break

    for adj_node in range(len(rev_adj_matrix[node_b])):
      if adj_node not in explored_b and rev_adj_matrix[node_b][adj_node] != 0:
        parent_b[adj_node] = node_b
        q2.put(adj_node)
        explored_b.append(adj_node)

  if(intersection_node == -1):
    # return None, 10000
    return None
  path_f = [intersection_node]
  node = intersection_node
  while(node != start_node):
    path_f.append(parent_f[node])
    node = parent_f[node]
  path_f.reverse()
  path_b = []
  node = intersection_node
  while(node != goal_node):
    path_b.append(parent_b[node])
    node = parent_b[node]
  path = path_f + path_b
  cost = 0
  for i in range(len(path)-1):
    cost += adj_matrix[path[i]][path[i+1]]
  # return path, cost
  return path

# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def dist(x1, y1, x2, y2):
  return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def heuristic(node, start_node, goal_node, node_attributes):
  x_start, y_start = node_attributes[start_node]['x'], node_attributes[start_node]['y']
  x_goal, y_goal = node_attributes[goal_node]['x'], node_attributes[goal_node]['y']
  x_node, y_node = node_attributes[node]['x'], node_attributes[node]['y']
  return dist(x_start, y_start, x_node, y_node) + dist(x_node, y_node, x_goal, y_goal)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
  num_nodes = len(adj_matrix)
  optimal_cost_list = [np.inf]*num_nodes
  parent = [-1]*num_nodes
  priority_queue = []
  g = 0
  h = heuristic(start_node, start_node, goal_node, node_attributes)
  heapq.heappush(priority_queue, (g+h, start_node))
  optimal_cost_list[start_node] = g+h
  while priority_queue:
    cost, node = heapq.heappop(priority_queue)
    h_node = heuristic(node, start_node, goal_node, node_attributes)
    g_node = cost - h_node
    for adj_node in range(len(adj_matrix[node])):
      if adj_matrix[node][adj_node] != 0:
        g_adj_node = g_node + adj_matrix[node][adj_node]
        h_adj_node = heuristic(adj_node, start_node, goal_node, node_attributes)
        if g_adj_node + h_adj_node < optimal_cost_list[adj_node]:
          optimal_cost_list[adj_node] = g_adj_node + h_adj_node
          parent[adj_node] = node
          heapq.heappush(priority_queue, (g_adj_node + h_adj_node, adj_node))

  if(optimal_cost_list[goal_node] == np.inf):
    # return None, 10000
    return None
  path = [goal_node]
  node = goal_node
  while(node != start_node):
    path.append(parent[node])
    node = parent[node]
  path.reverse()
  # return path, optimal_cost_list[goal_node]
  return path


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  rev_adj_matrix = np.transpose(adj_matrix)
  num_nodes = len(adj_matrix)
  optimal_cost_list_f = [np.inf]*num_nodes
  optimal_cost_list_b = [np.inf]*num_nodes
  parent_f = [-1]*num_nodes
  parent_b = [-1]*num_nodes
  priority_queue_f = []
  priority_queue_b = []
  h_f = heuristic(start_node, start_node, goal_node, node_attributes)
  h_b = heuristic(goal_node, start_node, goal_node, node_attributes)
  heapq.heappush(priority_queue_f, (h_f, start_node))
  heapq.heappush(priority_queue_b, (h_b, goal_node))
  optimal_cost_list_f[start_node] = h_f
  optimal_cost_list_b[goal_node] = h_b
  dir = True
  min_cost = np.inf
  intersection_node = -1
  while priority_queue_f or priority_queue_b:
    cost_node, node = np.inf, -1
    matrix = adj_matrix
    optimal_cost_list = optimal_cost_list_f
    optimal_cost_list2 = optimal_cost_list_b
    priority_queue = priority_queue_f
    parent = parent_f
    dir = not priority_queue_b or (priority_queue_f and priority_queue_f[0] < priority_queue_b[0])
    if dir:
      cost_node, node = heapq.heappop(priority_queue_f)
    else:
      cost_node, node = heapq.heappop(priority_queue_b)
      matrix = rev_adj_matrix
      optimal_cost_list = optimal_cost_list_b
      optimal_cost_list2 = optimal_cost_list_f
      priority_queue = priority_queue_b
      parent = parent_b

    if(dir):
      h_node = heuristic(node, start_node, goal_node, node_attributes)
      g_node = cost_node - h_node
      for adj_node in range(len(matrix[node])):
        if matrix[node][adj_node] != 0:
          g_adj_node = g_node + matrix[node][adj_node]
          h_adj_node = heuristic(adj_node, start_node, goal_node, node_attributes)
          if g_adj_node + h_adj_node < optimal_cost_list[adj_node]:
            optimal_cost_list[adj_node] = g_adj_node + h_adj_node
            if optimal_cost_list[adj_node] + optimal_cost_list2[adj_node] < min_cost:
              min_cost = optimal_cost_list[adj_node] + optimal_cost_list2[adj_node]
              intersection_node = adj_node
            parent[adj_node] = node
            heapq.heappush(priority_queue, (g_adj_node + h_adj_node, adj_node))

  if(intersection_node == -1):
    # return None, 10000
    return None
  path_f = [intersection_node]
  node = intersection_node
  while(node != start_node):
    path_f.append(parent_f[node])
    node = parent_f[node]
  path_f.reverse()
  path_b = []
  node = intersection_node
  while(node != goal_node):
    path_b.append(parent_b[node])
    node = parent_b[node]
  path = path_f + path_b

  # return path, min_cost
  return path

# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def plotter(adj_matrix, node_attributes):
  time_consumed = []
  memory_consumed = []
  cost = []
  limit = len(adj_matrix)
  limit1 = 50
  limit2 = 50
  for node1 in range(limit1):
    for node2 in range(limit2):
      print("IDS", node1, node2)
      tracemalloc.start()
      start_time = time.time()
      _, cost_i = get_ids_path(adj_matrix, node1, node2)
      cost.append(cost_i)
      end_time = time.time()
      time_consumed.append(end_time-start_time)
      _, peak = tracemalloc.get_traced_memory()
      peak = peak / 1024 / 1024
      tracemalloc.stop()
      memory_consumed.append(peak)
  return time_consumed, memory_consumed, cost

def bonus_problem(adj_matrix):

  return []


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
  # time_consumed, memory_consumed, cost = plotter(adj_matrix, node_attributes)

# x = time_consumed
# y = memory_consumed
# z = cost
# plt.scatter(x, y)
# plt.title('time vs memory')
# plt.xlabel('Time')
# plt.ylabel('Memory')
# plt.grid(True)
# plt.show()

# plt.scatter(x, z)
# plt.title('time vs cost')
# plt.xlabel('Time')
# plt.ylabel('cost')
# plt.grid(True)
# plt.show()

# plt.scatter(y, z)
# plt.title('Memory vs cost')
# plt.xlabel('memory')
# plt.ylabel('cost')
# plt.grid(True)
# plt.show()