from enum import Enum
import heapq
import itertools
from math import sqrt
from typing import List
import time
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
GRID_SIZE = 8
results = {}
FRONTIER_MAX = GRID_SIZE * GRID_SIZE   # bounded frontier capacity
 
# ─── Tee: write stdout to both console and output file ───────────────────────
class Tee:
    """Writes output to both the console and a file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()
 
# ─── Bounded Priority Queue with proper empty/full error messages ─────────────
class BoundedPriorityQueue:
    """
    A bounded priority queue (min-heap) wrapping heapq.
    Raises RuntimeError with a descriptive message when:
    - push() is called on a full queue (capacity = FRONTIER_MAX)
    - pop() is called on an empty queue
    This satisfies the assignment requirement for explicit insert/delete
    error handling on the frontier data structure.
    """
    def __init__(self, maxsize: int = FRONTIER_MAX):
        self._heap = []
        self.maxsize = maxsize
 
    def push(self, item):
        if len(self._heap) >= self.maxsize:
            raise RuntimeError(
                f"ERROR: Frontier is FULL (capacity={self.maxsize}). "
                f"Cannot insert node ({item[2].x},{item[2].y})."
            )
        heapq.heappush(self._heap, item)
 
    def pop(self):
        if not self._heap:
            raise RuntimeError("ERROR: Frontier is EMPTY. Cannot remove a node.")
        return heapq.heappop(self._heap)
 
    def empty(self):
        return len(self._heap) == 0
 
    def qsize(self):
        return len(self._heap)
 
    def snapshot(self):
        return list(self._heap)
 
class NodeType(Enum):
    """
    Represents different cell types in the environment.
    Each cell type is associated with:
    1. A visual symbol
    2. A traversal cost
    Symbols:
    S : Start Node
    E : Goal Node
    . : Passable Airspace
    W : Weather Hazard
    N : No-Fly Zone
    """
    PASSABLE_AIRSPACE = (".", 1)
    WEATHER_HAZARD = ("W", 4)
    NO_FLY_ZONE = ("N", 8)
    START = ("S", 2)
    END = ("E", 2)
 
class MovementModel(Enum):
    """
    Represents the allowed movement directions for the drone.
    The drone can move only in four orthogonal directions:
    North, South, East and West. Diagonal movement is prohibited.
    """
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
 
class Node:
    """
    Represents a single cell in the grid.
    Attributes:
    x      : Row index
    y      : Column index
    type   : Cell type symbol
    score  : Traversal cost of the cell
    """
    def __init__(self, x: int, y: int, type: NodeType) -> None:
        self.x = x
        self.y = y
        self.type = type[0]
        self.score = type[1]
    def __hash__(self):
        return hash((self.x, self.y))
    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y
 
class HeuristicType(Enum):
    """
    Enumeration of the supported heuristics.
    h1 : Euclidean Distance
    h2 : Bounding Box Risk Weighted Heuristic
    """
    EUCLIDEAN_DISTANCE = "h1"
    BOUNDING_BOX_RISK_WEIGHTED = "h2"
 
def euclidean_distance(current_state: Node, goal_state: Node):
    """
    Computes the Euclidean distance between the current node and the goal node.
    Formula: h1(n) = sqrt((xg - xn)^2 + (yg - yn)^2)
    """
    return sqrt(((goal_state.x - current_state.x) ** 2 +
                 (goal_state.y - current_state.y) ** 2))
 
def manhattan_distance(current_state: Node, goal_state: Node):
    """
    Computes the Manhattan distance between the current node and goal node.
    Used internally by the Bounding Box Risk Weighted heuristic.
    """
    return abs(goal_state.x - current_state.x) + abs(goal_state.y - current_state.y)
 
def bounding_box_risk_weighted(grid: List[List[Node]], current_state: Node, goal_state: Node):
    """
    Computes the Bounding Box Risk Weighted heuristic (h2).
    Steps:
    1. Calculate Manhattan distance.
    2. Construct the bounding box between current node and goal node.
    3. Sum traversal costs of all cells inside the bounding box.
    4. Compute average risk.
    5. Multiply Manhattan distance by average risk.
    Formula: h2 = ManhattanDistance × AverageRisk
    """
    manhattan = manhattan_distance(current_state, goal_state)
    k = (abs(goal_state.x - current_state.x) + 1) * (abs(goal_state.y - current_state.y) + 1)
    x_min = min(current_state.x, goal_state.x)
    x_max = max(current_state.x, goal_state.x)
    y_min = min(current_state.y, goal_state.y)
    y_max = max(current_state.y, goal_state.y)
    partial_score = sum(grid[r][c].score for r in range(x_min, x_max + 1) for c in range(y_min, y_max + 1))
    return manhattan * (partial_score / k)
 
def heuristic(grid, current_state: Node, goal_state: Node, type: HeuristicType):
    """Dispatcher: selects heuristic h1 (Euclidean) or h2 (Bounding Box Risk Weighted)."""
    if type == HeuristicType.EUCLIDEAN_DISTANCE:
        return euclidean_distance(current_state, goal_state)
    elif type == HeuristicType.BOUNDING_BOX_RISK_WEIGHTED:
        return bounding_box_risk_weighted(grid, current_state, goal_state)
    else:
        raise ValueError(f"Unsupported heuristic: {type}")
 
# ─── Grid loading from file (Req: no hardcoded values) ────────────────────────
def load_grid_from_file(file_path: str):
    """
    Loads the 8x8 grid from inputPSXX.txt.
    Each row must contain exactly 8 symbols from {S, E, ., W, N}.
    Returns (grid, start_node, end_node) or None if file is invalid.
    """
    if not os.path.exists(file_path):
        return None
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = [ch for ch in line.strip() if ch in "SE.WN"]
            if len(tokens) == GRID_SIZE:
                rows.append(tokens)
    if len(rows) != GRID_SIZE:
        return None
    grid = []
    start_node = end_node = None
    symbol_map = {"S": NodeType.START, "E": NodeType.END,
                  "W": NodeType.WEATHER_HAZARD, "N": NodeType.NO_FLY_ZONE,
                  ".": NodeType.PASSABLE_AIRSPACE}
    for x, row in enumerate(rows):
        grid_row = []
        for y, sym in enumerate(row):
            node = Node(x, y, symbol_map[sym].value)
            grid_row.append(node)
            if sym == "S": start_node = node
            if sym == "E": end_node = node
        grid.append(grid_row)
    if start_node is None or end_node is None:
        return None
    return grid, start_node, end_node
 
def build_grid(size: int, start_x, start_y, end_x, end_y):
    """
    Fallback: builds the fixed 8×8 assignment grid if no input file is found.
    """
    grid = [[Node(x, y, NodeType.PASSABLE_AIRSPACE.value) for y in range(size)] for x in range(size)]
    grid[start_x][start_y] = Node(start_x, start_y, NodeType.START.value)
    grid[end_x][end_y] = Node(end_x, end_y, NodeType.END.value)
    for x, y in [(1,1),(2,3),(2,6),(4,3),(6,2),(7,3),(7,5)]:
        grid[x][y] = Node(x, y, NodeType.WEATHER_HAZARD.value)
    for x, y in [(0,4),(1,4),(2,4),(3,0),(3,1),(3,6),(5,5),(5,6)]:
        grid[x][y] = Node(x, y, NodeType.NO_FLY_ZONE.value)
    return grid
 
def show_grid(grid, path=[]):
    """
    Displays the environment in a visual grid format.
    Path cells are highlighted with '*'.
    """
    path_set = set(path)
    size = len(grid)
    boundary = "+---" * size + "+"
    print(boundary)
    for row in grid:
        print("|", end="")
        for node in row:
            print(f" {'*' if node in path_set else node.type} |", end="")
        print()
        print(boundary)
    print("\n")
 
def print_complexity_analysis():
    """
    Displays full complexity and performance metrics for all algorithms.
    Columns: Nodes Expanded, Runtime(ms), Memory, Cost, Path Length, Heuristic, Penalty.
    Also prints theoretical Big-O complexity for GBFS and A*.
    """
    print("\n" + "=" * 100)
    print("COMPLEXITY ANALYSIS")
    print("=" * 100)
    print(f"{'Algorithm':<15}{'Nodes':<10}{'Runtime(ms)':<15}{'Memory':<10}"
          f"{'Cost':<10}{'Length':<10}{'Heuristic':<12}{'Penalty':<10}")
    for algo, data in results.items():
        print(f"{algo:<15}{data['nodes_expanded']:<10}{data['runtime']:<15.3f}"
              f"{data['memory']:<10}{data['cost']:<10}{data['path_length']:<10}"
              f"{data['heuristic']:<12}{data['penalty_points']:<10}")
    print("\nTHEORETICAL COMPLEXITY")
    print("GBFS Time Complexity : O(V log V)")
    print("GBFS Space Complexity: O(V)")
    print("A* Time Complexity   : O(V log V)")
    print("A* Space Complexity  : O(V)")
 
def compare_heuristics_gbfs():
    """
    Compares GBFS heuristics h1 (Euclidean) vs h2 (Bounding Box Risk Weighted).
    Metrics: Nodes Expanded, Runtime, Path Cost, Penalty.
    Describes which heuristic is better and why (Req 5a).
    """
    print("\n" + "=" * 100)
    print("GBFS HEURISTIC COMPARISON")
    print("=" * 100)
    h1 = results.get("GBFS-h1")
    h2 = results.get("GBFS-h2")
    if not h1 or not h2:
        return
    print(f"{'Metric':<20}{'h1':<15}{'h2':<15}")
    print(f"{'Nodes Expanded':<20}{h1['nodes_expanded']:<15}{h2['nodes_expanded']:<15}")
    print(f"{'Runtime(ms)':<20}{round(h1['runtime'],3):<15}{round(h2['runtime'],3):<15}")
    print(f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    print(f"{'Penalty':<20}{h1['penalty_points']:<15}{h2['penalty_points']:<15}")
    # Req 5a: written heuristic analysis
    print("\nHEURISTIC ANALYSIS (Req 5a):")
    if h1["cost"] < h2["cost"]:
        print("  h1 (Euclidean) produced a lower cost path.")
        print("  h1 is better here: it guides GBFS to avoid costly hazard cells.")
    elif h2["cost"] < h1["cost"]:
        print("  h2 (Bounding Box Risk Weighted) produced a lower cost path.")
        print("  h2 is better here: its future-aware risk weighting avoids hazard regions more effectively.")
    else:
        print("  Both heuristics produced the same path cost.")
    print("  h2 is generally more informed as it considers the average terrain risk")
    print("  across the bounding box, not just straight-line distance.")
 
def compare_heuristics_Astar():
    """
    Compares A* heuristics h1 vs h2.
    Metrics: Nodes Expanded, Runtime, Path Cost, Penalty.
    """
    print("\n" + "=" * 100)
    print("A* HEURISTIC COMPARISON")
    print("=" * 100)
    h1 = results.get(f"A*-{HeuristicType.EUCLIDEAN_DISTANCE.value}")
    h2 = results.get(f"A*-{HeuristicType.BOUNDING_BOX_RISK_WEIGHTED.value}")
    if not h1 or not h2:
        print("check why heuristics are not found")
        return
    print(f"{'Metric':<20}{'h1':<15}{'h2':<15}")
    print(f"{'Nodes Expanded':<20}{h1['nodes_expanded']:<15}{h2['nodes_expanded']:<15}")
    print(f"{'Runtime(ms)':<20}{round(h1['runtime'],3):<15}{round(h2['runtime'],3):<15}")
    print(f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    print(f"{'Penalty':<20}{h1['penalty_points']:<15}{h2['penalty_points']:<15}")
    if h1["cost"] < h2["cost"]:
        print("\nh1 produced a lower cost path.")
    elif h2["cost"] < h1["cost"]:
        print("\nh2 produced a lower cost path.")
    else:
        print("\nBoth heuristics produced the same cost.")
 
def compare_algorithms():
    """
    Compares GBFS and A* on: Nodes Expanded, Runtime, Cost, Penalty.
    Also reports completeness, fastest algorithm, and lowest cost path (Req 7).
    """
    print("\n" + "=" * 100)
    print("GBFS vs A* COMPARISON")
    print("=" * 100)
    print(f"{'Algorithm':<15}{'Nodes':<15}{'Runtime(ms)':<15}{'Cost':<15}{'Penalty':<12}")
    for algo, data in results.items():
        print(f"{algo:<15}{data['nodes_expanded']:<15}{round(data['runtime'],3):<15}"
              f"{data['cost']:<15}{data['penalty_points']:<12}")
    print("\nCOMPLETENESS ANALYSIS")
    print("---------------------")
    print("A*   : Complete")
    print("GBFS : Not guaranteed complete in general")
    fastest = min(results.items(), key=lambda x: x[1]["runtime"])
    cheapest = min(results.items(), key=lambda x: x[1]["cost"])
    print(f"\nFASTEST ALGORITHM\n-----------------\n"
          f"{fastest[0]} was fastest with {fastest[1]['runtime']:.3f} ms")
    print(f"\nLOWEST COST PATH\n----------------\n"
          f"{cheapest[0]} produced the lowest path cost of {cheapest[1]['cost']}")
 
def get_neighbours(grid, node: Node):
    """
    Returns valid neighbours in tie-breaking order: North, East, South, West.
    """
    neighbours = []
    if node.x >= 1:            neighbours.append(grid[node.x - 1][node.y])  # NORTH
    if node.y < GRID_SIZE - 1: neighbours.append(grid[node.x][node.y + 1])  # EAST
    if node.x < GRID_SIZE - 1: neighbours.append(grid[node.x + 1][node.y])  # SOUTH
    if node.y >= 1:            neighbours.append(grid[node.x][node.y - 1])  # WEST
    return neighbours
 
# ─── Trap analysis (Req 6) ────────────────────────────────────────────────────
def print_trap_analysis():
    """
    Req 6: Tabular display of all trap situations encountered.
    A trap occurs when no new children can be added from a node (dead-end).
    Columns: Algorithm, Heuristic, Heuristic Value, Trapped At, Next Iteration.
    """
    print("\n" + "=" * 100)
    print("TRAP / INEFFICIENT REGION ANALYSIS (Req 6)")
    print("=" * 100)
    print(f"{'Algorithm':<14}{'Heuristic':<10}{'Heuristic Value':<18}{'Trapped At':<16}{'Next Iteration':<16}")
    any_trap = False
    for algo, data in results.items():
        for trap in data.get("traps", []):
            any_trap = True
            trapped_at = f"({trap['node'][0]},{trap['node'][1]})"
            next_iter = trap['escape'] if trap['escape'] is not None else 'N/A'
            print(f"{algo:<14}{trap['heuristic']:<10}{trap['value']:<18}{trapped_at:<16}{next_iter:<16}")
    if not any_trap:
        print("No trap-like dead-end situations were encountered in this run.")
 
# ─── Conclusion (Req 12) ──────────────────────────────────────────────────────
def print_conclusion():
    """
    Req 12: Concluding analysis of GBFS performance
    in terms of optimality and completeness.
    """
    print("\n" + "=" * 100)
    print("CONCLUSION (Req 12)")
    print("=" * 100)
    gbfs_h1 = results.get("GBFS-h1")
    gbfs_h2 = results.get("GBFS-h2")
    astar_h1 = results.get("A*-h1")
    print("OPTIMALITY:")
    print("  GBFS is NOT optimal. It selects nodes based purely on heuristic value h(n),")
    print("  ignoring the actual path cost g(n). This greedy strategy can lead to")
    print("  suboptimal paths, especially in grids with heterogeneous terrain costs.")
    if gbfs_h1 and astar_h1:
        diff = gbfs_h1["cost"] - astar_h1["cost"]
        if diff > 0:
            print(f"  In this run, GBFS-h1 cost={gbfs_h1['cost']} vs A*-h1 cost={astar_h1['cost']}"
                  f" (GBFS overspent by {diff}).")
        else:
            print(f"  In this run, GBFS-h1 and A*-h1 found equally costed paths (cost={astar_h1['cost']}).")
    print("\nCOMPLETENESS:")
    print("  GBFS is NOT guaranteed to be complete in general graphs.")
    print("  However, on a finite grid with no-fly zones (hard obstacles), GBFS will")
    print("  always terminate — either finding the goal or exhausting the frontier.")
    print("  A* is complete and optimal when using an admissible heuristic.")
    print("\nHEURISTIC PERFORMANCE:")
    if gbfs_h1 and gbfs_h2:
        print(f"  GBFS-h1 expanded {gbfs_h1['nodes_expanded']} nodes, cost={gbfs_h1['cost']}.")
        print(f"  GBFS-h2 expanded {gbfs_h2['nodes_expanded']} nodes, cost={gbfs_h2['cost']}.")
        print("  h2's risk-weighted bounding box makes it more terrain-aware,")
        print("  often producing lower penalty paths at the cost of more computation.")
    print("\nOVERALL: A* with h1 is the recommended algorithm for this problem when")
    print("optimality is required. GBFS with h2 is preferable when speed matters more.")
 
def astar(grid, start_node: Node, end_node: Node,
          heuristic_fx: HeuristicType = HeuristicType.EUCLIDEAN_DISTANCE):
    """
    Implements A* Search.
    Evaluation Function: f(n) = g(n) + h(n)
    Tracks traps (Req 6): nodes from which no new children can be added.
    Stores penalty_points (weather * 4 + no-fly * 8) in results.
    """
    goal_found = False
    start_time = time.perf_counter()
    pq = BoundedPriorityQueue()
    counter = itertools.count()
    visited = set()
    parent = {}
    g_score = {}
    traps = []                        # Req 6: trap list
    time_history = []                 # Req 5c: elapsed ms at each expansion
    g_score[start_node] = 0
    parent[start_node] = None
    h_val = heuristic(grid, start_node, end_node, heuristic_fx)
    initial_h = round(h_val, 2)       # store start-node heuristic for metrics table
    try:
        pq.push((g_score[start_node] + h_val, next(counter), start_node))
    except RuntimeError as e:
        print(e); return
    while not pq.empty():
        print("\nFRONTIER:")
        for item in sorted(pq.snapshot()):
            print(f"({item[2].x},{item[2].y}) h={round(item[0], 2)}")
        try:
            f_val, cnt, current_node = pq.pop()
        except RuntimeError as e:
            print(e); break
        current_cost = g_score.get(current_node)
        if current_node in visited:
            continue
        visited.add(current_node)
        time_history.append(round((time.perf_counter() - start_time) * 1000, 4))  # Req 5c
        print("\nEXPLORED NODES:")
        for node in visited:
            print(f"({node.x},{node.y})", end=" ")
        print("\n")
        print(f"\nSELECTED NODE : ({current_node.x},{current_node.y})")
        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        print(f"HEURISTIC VALUE : {round(current_h, 2)}")
        if current_node == end_node:
            path = []
            tmp = current_node
            while tmp:
                path.append(tmp)
                tmp = parent.get(tmp)
            path.reverse()
            cost = sum(node.score for node in path)
            show_grid(grid, path)
            weather_count = sum(1 for n in path if n.type == "W")
            nofly_count   = sum(1 for n in path if n.type == "N")
            penalty_points = weather_count * 4 + nofly_count * 8   # Req 2i
            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)
            print("PENALTY POINTS:", penalty_points)
            print("Followed Path: ")
            print(" -> ".join(f"({n.x},{n.y})" for n in path))
            runtime_ms = (time.perf_counter() - start_time) * 1000
            results[f"A*-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": len(visited) + pq.qsize(),
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": initial_h,             # initial h from start node (Req 4)
                "penalty_points": penalty_points,   # Req 4
                "traps": traps,
                "time_history": time_history
            }
            goal_found = True
            break
        added_child = False                          # Req 6: trap detection
        for node in get_neighbours(grid, current_node):
            if node in visited or node.type == "N":
                continue
            new_cost = current_cost + node.score
            if node not in g_score or new_cost < g_score.get(node):
                parent[node] = current_node
                g_score[node] = new_cost
                h_val = heuristic(grid, node, end_node, heuristic_fx)
                try:
                    pq.push((new_cost + h_val, next(counter), node))
                except RuntimeError as e:
                    print(e)
                added_child = True
        if not added_child:
            escape = None
            if not pq.empty():
                top = sorted(pq.snapshot())[0]
                escape = f"({top[2].x},{top[2].y})"
            traps.append({
                "heuristic": heuristic_fx.value,
                "value": round(current_h, 2),
                "node": (current_node.x, current_node.y),
                "escape": escape
            })
    if not goal_found:
        print("ERROR: Goal node could not be reached.")
 
def gbfs(grid, start_node: Node, end_node: Node,
         heuristic_fx: HeuristicType = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED):
    """
    Implements Greedy Best First Search.
    Evaluation Function: f(n) = h(n)
    Tracks traps (Req 6): nodes from which no new children can be added.
    Stores penalty_points and traps in results.
    """
    goal_found = False
    start_time = time.perf_counter()
    pq = BoundedPriorityQueue()
    counter = itertools.count()
    visited = set()
    parent = {}
    traps = []                        # Req 6: trap list
    time_history = []                 # Req 5c: elapsed ms at each expansion
    parent[start_node] = None
    expansion_history = []
    h_val = heuristic(grid, start_node, end_node, heuristic_fx)
    initial_h = round(h_val, 2)       # store start-node heuristic for metrics table
    try:
        pq.push((h_val, next(counter), start_node))
    except RuntimeError as e:
        print(e); return
    while not pq.empty():
        print("\nFRONTIER:")
        for item in sorted(pq.snapshot()):
            print(f"({item[2].x},{item[2].y}) h={round(item[0], 2)}")
        try:
            h_val, cnt, current_node = pq.pop()
        except RuntimeError as e:
            print(e); break
        if current_node in visited:
            continue
        visited.add(current_node)
        expansion_history.append(h_val)
        time_history.append(round((time.perf_counter() - start_time) * 1000, 4))  # Req 5c
        print("\nEXPLORED NODES:")
        for node in visited:
            print(f"({node.x},{node.y})", end=" ")
        print("\n")
        print(f"\nSELECTED NODE : ({current_node.x},{current_node.y})")
        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        print(f"HEURISTIC VALUE : {round(current_h, 2)}")
        if current_node == end_node:
            path = []
            tmp = current_node
            while tmp:
                path.append(tmp)
                tmp = parent.get(tmp)
            path.reverse()
            cost = sum(node.score for node in path)
            path_history = [heuristic(grid, n, end_node, heuristic_fx) for n in path]
            show_grid(grid, path)
            weather_count = sum(1 for n in path if n.type == "W")
            nofly_count   = sum(1 for n in path if n.type == "N")
            penalty_points = weather_count * 4 + nofly_count * 8   # Req 2i
            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)
            print("PENALTY POINTS:", penalty_points)
            print("Followed Path: ")
            print(" -> ".join(f"({n.x},{n.y})" for n in path))
            runtime_ms = (time.perf_counter() - start_time) * 1000
            results[f"GBFS-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": len(visited) + pq.qsize(),
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": initial_h,             # initial h from start node (Req 4)
                "penalty_points": penalty_points,   # Req 4
                "expansion_history": expansion_history,
                "path_history": path_history,
                "time_history": time_history,       # Req 5c: actual elapsed ms
                "traps": traps
            }
            goal_found = True
            break
        added_child = False                          # Req 6: trap detection
        for node in get_neighbours(grid, current_node):
            if node.type == "N":
                continue
            if node not in visited and node not in parent:
                parent[node] = current_node
                h_val = heuristic(grid, node, end_node, heuristic_fx)
                try:
                    pq.push((h_val, next(counter), node))
                except RuntimeError as e:
                    print(e)
                added_child = True
        if not added_child:
            escape = None
            if not pq.empty():
                top = sorted(pq.snapshot())[0]
                escape = f"({top[2].x},{top[2].y})"
            traps.append({
                "heuristic": heuristic_fx.value,
                "value": round(current_h, 2),
                "node": (current_node.x, current_node.y),
                "escape": escape
            })
    if not goal_found:
        print("ERROR: Goal node could not be reached.")
 
def print_text_chart(title: str, history: list):
    """
    Text-based bar chart of heuristic values (Req 5b, 5c).
    Shows relative scale using block characters.
    """
    print("=" * 70)
    print(f" VISUAL TREND: {title}")
    print("=" * 70)
    if not history:
        print("No data available.")
        return
    max_val = max(history) if max(history) > 0 else 1
    max_bars = 35
    print(f"{'Step / Index':<12} | {'Heuristic':<12} | Relative Visual Scale")
    print("-" * 70)
    for idx, val in enumerate(history):
        bar_length = int((val / max_val) * max_bars)
        print(f"Index {idx:<6} | {val:<12.2f} | {'█' * bar_length}")
    print("=" * 70 + "\n")
 
def save_heuristic_chart(title: str, x_vals: list, y_vals: list,
                         xlabel: str, ylabel: str, filename: str, base_dir: str):
    """
    Saves a matplotlib line chart for heuristic trend visualisation (Req 5b, 5c).
    x_vals: list of x-axis values (step index or elapsed time ms)
    y_vals: list of heuristic values
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, marker='o', color='steelblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), dpi=150)
    plt.close()
    print(f"  Chart saved: {filename}")
 
def save_complexity_charts(base_dir: str):
    """
    Req 3: Saves bar charts for nodes expanded, runtime, memory, and path cost
    across all algorithms for visual complexity comparison.
    """
    if not results:
        return
    labels = list(results.keys())
    metrics = [
        ("nodes_expanded", "Nodes Expanded by Algorithm", "Nodes Expanded", "complexity_nodes.png"),
        ("runtime",        "Runtime by Algorithm (ms)",   "Runtime (ms)",    "complexity_runtime.png"),
        ("memory",         "Memory Usage by Algorithm",   "Memory (OPEN+CLOSED)", "complexity_memory.png"),
        ("cost",           "Path Cost by Algorithm",      "Total Path Cost", "complexity_cost.png"),
    ]
    for key, title, ylabel, filename in metrics:
        values = [results[l][key] for l in labels]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, values, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        plt.title(title)
        plt.ylabel(ylabel)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{round(val, 2)}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, filename), dpi=150)
        plt.close()
        print(f"  Chart saved: {filename}")
 
def write_sequence_diagram(base_dir: str) -> str:
    """
    Req 9b: Writes execution flow as a Mermaid sequence diagram
    to sequence_diagram.mmd in the output directory.
    """
    path = os.path.join(base_dir, "sequence_diagram.mmd")
    content = """sequenceDiagram
    participant U as User
    participant M as main()
    participant G as Grid Loader
    participant F as FrontierQueue (PriorityQueue)
    participant S as Search Algorithm (GBFS / A*)
    participant O as outputPSXX.txt
 
    U->>M: Run program with inputPSXX.txt
    M->>G: load_grid_from_file(inputPSXX.txt)
    G-->>M: grid, start_node, end_node
    M->>O: Tee stdout to outputPSXX.txt
    M->>M: show_grid() — display initial environment
 
    loop For each algorithm (GBFS-h2, GBFS-h1, A*-h1, A*-h2)
        M->>S: Run algorithm with selected heuristic
        S->>F: push(heuristic_value, counter, node)
        F-->>S: ERROR if full
        loop While frontier not empty
            S->>F: pop() — select best node
            F-->>S: ERROR if empty
            S->>S: Mark node as visited (CLOSED)
            S->>S: Print FRONTIER, EXPLORED NODES, SELECTED NODE, HEURISTIC VALUE
            alt Goal reached
                S->>S: Reconstruct path via parent map
                S->>S: Compute cost, penalty, path length
                S->>M: Return metrics + trap list
            else Expand neighbours
                S->>F: push neighbours (not visited, not N)
                S->>S: Record trap if no children added
            end
        end
    end
 
    M->>M: print_peas()
    M->>M: print_complexity_analysis()
    M->>M: compare_heuristics_gbfs() / compare_heuristics_Astar()
    M->>M: compare_algorithms()
    M->>M: print_trap_analysis()
    M->>M: print_text_chart() x4 (heuristic trend visuals)
    M->>M: print_conclusion()
    M->>M: write_sequence_diagram()
    M->>O: All output saved
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nSequence diagram written to: {path}")
    return path
 
def print_peas():
    """Req 8: PEAS components of the drone agent."""
    print("\n" + "=" * 100)
    print("PEAS COMPONENTS (Req 8)")
    print("=" * 100)
    print("Performance : Reach the goal with minimum heuristic cost, fewer node expansions,")
    print("              avoid no-fly zones, minimise weather hazard penalties.")
    print("Environment : Discrete 8×8 2D grid containing passable airspace, weather hazards,")
    print("              and no-fly zones. Fully observable, deterministic, static.")
    print("Actuators   : Move North, East, South, or West (orthogonal only).")
    print("Sensors     : Current position, cell type, heuristic value to goal,")
    print("              frontier queue, explored set.")
 
def main():
    """
    Driver function.
    Execution Flow:
    1. Load grid from inputPSXX.txt (fallback to hardcoded grid).
    2. Tee stdout to outputPSXX.txt.
    3. Display initial environment.
    4. Run GBFS (h2), GBFS (h1), A* (h1), A* (h2).
    5. Print all analysis sections.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    input_path  = os.path.join(base_dir, "inputPSXX.txt")
    output_path = os.path.join(base_dir, "outputPSXX.txt")
 
    # Load grid from file; fall back to hardcoded assignment grid
    parsed = load_grid_from_file(input_path)
    if parsed:
        grid, start_node, end_node = parsed
        print(f"Grid loaded from {input_path}")
    else:
        print(f"Input file not found or invalid. Using hardcoded assignment grid.")
        grid = build_grid(GRID_SIZE, 0, 0, 6, 7)
        start_node = grid[0][0]
        end_node   = grid[6][7]
 
    original_stdout = sys.stdout
    with open(output_path, "w", encoding="utf-8") as out_file:
        sys.stdout = Tee(original_stdout, out_file)
        try:
            print("=" * 100)
            print("Initial Grid")
            print("=" * 100)
            show_grid(grid)
 
            print("=" * 100)
            print("GBFS USING H2")
            print("=" * 100)
            gbfs(grid, start_node, end_node, HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)
 
            print("=" * 100)
            print("GBFS USING H1")
            print("=" * 100)
            gbfs(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE)
 
            print("=" * 100)
            print("A* USING H1")
            print("=" * 100)
            astar(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE)
 
            print("=" * 100)
            print("A* USING H2")
            print("=" * 100)
            astar(grid, start_node, end_node, HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)
 
            print_peas()
            print_complexity_analysis()
            compare_heuristics_gbfs()
            compare_heuristics_Astar()
            compare_algorithms()
            print_trap_analysis()
 
            # Req 5b: heuristic values along the final path (text + PNG)
            for tag, hkey in [("h1", "GBFS-h1"), ("h2", "GBFS-h2")]:
                path_h = results[hkey]["path_history"]
                time_h = results[hkey]["time_history"]
                exp_h  = results[hkey]["expansion_history"]
                steps  = list(range(len(path_h)))
                exp_steps = list(range(len(exp_h)))
 
                print_text_chart(
                    f"Heuristic Values to Reach Target along Path ({tag}) [5.b]", path_h)
                save_heuristic_chart(
                    f"GBFS-{tag}: Heuristic Values along Path (5.b)",
                    steps, path_h, "Path Step", "Heuristic Value",
                    f"gbfs_{tag}_path.png", base_dir)
 
                # Req 5c: heuristic vs actual elapsed time
                print_text_chart(
                    f"Heuristic Values vs Time to Reach Target ({tag}) [5.c]", exp_h)
                save_heuristic_chart(
                    f"GBFS-{tag}: Heuristic Values vs Elapsed Time (5.c)",
                    time_h, exp_h, "Elapsed Time (ms)", "Heuristic Value",
                    f"gbfs_{tag}_vs_time.png", base_dir)
 
            print("\nCOMPLEXITY CHARTS (Req 3):")
            save_complexity_charts(base_dir)
 
            print_conclusion()
            write_sequence_diagram(base_dir)
 
            print(f"\nOutput saved to: {output_path}")
        finally:
            sys.stdout = original_stdout
 
if __name__ == "__main__":
    main()
