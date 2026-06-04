from enum import Enum
import itertools
from math import sqrt
from typing import List
import queue
import time
import os
import sys

GRID_SIZE = 8

results = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DEFAULT_INPUT_FILE = os.path.join(BASE_DIR, "inputPSXX.txt")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "outputPSXX.txt")


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
    Represents the allowed movement directions
    for the drone.

    The drone can move only in four orthogonal directions:
    North, South, East and West.

    Diagonal movement is prohibited.
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

    The class implements hashing and equality
    to allow Node objects to be stored in sets
    and dictionaries.
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
    Computes the Euclidean distance between
    the current node and the goal node.

    Formula:
    h1(n) = sqrt((xg - xn)^2 + (yg - yn)^2)

    Parameters:
    current_state : Current node
    goal_state    : Goal node

    Returns:
    Euclidean distance value.
    """
    return sqrt(((goal_state.x - current_state.x) * (goal_state.x - current_state.x) + (
            goal_state.y - current_state.y) * (goal_state.y - current_state.y)))


def manhattan_distance(current_state: Node, goal_state: Node):
    """
    Computes the Manhattan distance between
    the current node and the goal node.

    Formula:
    |xg - xn| + |yg - yn|

    Used internally by the Bounding Box
    Risk Weighted heuristic.

    Returns:
    Manhattan distance value.
    """
    return abs(goal_state.x - current_state.x) + abs(goal_state.y - current_state.y)


def bounding_box_risk_weighted(grid: List[List[Node]], current_state: Node, goal_state: Node):
    """
    Computes the Bounding Box Risk Weighted
    heuristic (h2).

    The heuristic estimates the future risk
    between the current node and goal node.

    Steps:
    1. Calculate Manhattan distance.
    2. Construct the bounding box between
       current node and goal node.
    3. Sum the traversal costs of all cells
       inside the bounding box.
    4. Compute average risk.
    5. Multiply Manhattan distance by
       average risk.

    Formula:
    h2 = ManhattanDistance × AverageRisk

    Returns:
    Risk weighted heuristic value.
    """
    manhattan = manhattan_distance(current_state, goal_state)
    k = (abs(goal_state.x - current_state.x) + 1) * (abs(goal_state.y - current_state.y) + 1)
    x_min = min(current_state.x, goal_state.x)
    x_max = max(current_state.x, goal_state.x)
    y_min = min(current_state.y, goal_state.y)
    y_max = max(current_state.y, goal_state.y)
    partial_score = 0
    for row in range(x_min, x_max + 1):
        for col in range(y_min, y_max + 1):
            partial_score += grid[row][col].score
    return manhattan * (partial_score / k)


def heuristic(grid, current_state: Node, goal_state: Node, type: HeuristicType):
    """
    Dispatcher function used to select
    the heuristic requested by the user.

    Supported heuristics:
    - Euclidean Distance (h1)
    - Bounding Box Risk Weighted (h2)

    Returns:
    Heuristic value.
    """
    if type == HeuristicType.EUCLIDEAN_DISTANCE:
        return euclidean_distance(current_state, goal_state)
    elif type == HeuristicType.BOUNDING_BOX_RISK_WEIGHTED:
        return bounding_box_risk_weighted(grid, current_state, goal_state)
    else:
        raise ValueError(f"Unsupported heuristic: {type}")


def build_grid(size: int, start_x, start_y, end_x, end_y):
    """
    Creates and initializes the 8×8 environment.

    The method:
    1. Creates passable airspace cells.
    2. Places the start node.
    3. Places the goal node.
    4. Places weather hazard zones.
    5. Places no-fly zones.

    Returns:
    Fully initialized grid.
    """
    grid = [[Node(x, y, NodeType.PASSABLE_AIRSPACE.value) for y in range(size)] for x in range(size)]

    grid[start_x][start_y] = Node(start_x, start_y, NodeType.START.value)
    grid[end_x][end_y] = Node(end_x, end_y, NodeType.END.value)

    weather_hazards = [
        (1, 1),
        (2, 3), (2, 6),
        (4, 3),
        (6, 2),
        (7, 3), (7, 5)
    ]
    for x, y in weather_hazards:
        grid[x][y] = Node(x, y, NodeType.WEATHER_HAZARD.value)

    no_fly_zones = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 0), (3, 1), (3, 6),
        (5, 5), (5, 6)
    ]
    for x, y in no_fly_zones:
        grid[x][y] = Node(x, y, NodeType.NO_FLY_ZONE.value)

    return grid


class Tee:
    """Writes output to both the console and a file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_grid_from_file(file_path: str):
    """
    Reads an 8x8 grid from a text file using symbols S, E, ., W, and N.

    The parser accepts either compact rows or spaced rows, so the evaluator
    can provide the grid in a slightly different formatting style without
    breaking the program.

    Returns:
        (grid, start_node, end_node) if the file is valid, otherwise None.
    """
    if not os.path.exists(file_path):
        return None

    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            tokens = [ch for ch in line if ch in 'SE.WN']
            if len(tokens) == GRID_SIZE:
                rows.append(tokens)

    if len(rows) != GRID_SIZE:
        return None

    grid = []
    start_node = None
    end_node = None

    for x, row in enumerate(rows):
        grid_row = []
        for y, symbol in enumerate(row):
            if symbol == 'S':
                node = Node(x, y, NodeType.START.value)
                start_node = node
            elif symbol == 'E':
                node = Node(x, y, NodeType.END.value)
                end_node = node
            elif symbol == 'W':
                node = Node(x, y, NodeType.WEATHER_HAZARD.value)
            elif symbol == 'N':
                node = Node(x, y, NodeType.NO_FLY_ZONE.value)
            else:
                node = Node(x, y, NodeType.PASSABLE_AIRSPACE.value)
            grid_row.append(node)
        grid.append(grid_row)

    if start_node is None or end_node is None:
        return None

    return grid, start_node, end_node


def print_peas_components():
    """Prints the PEAS description required by the assignment."""
    print("\n" + "=" * 100)
    print("PEAS COMPONENTS")
    print("=" * 100)
    print("Performance: Reach the goal with low heuristic cost, fewer expansions, and avoid no-fly zones.")
    print("Environment: 8x8 grid with passable airspace, weather hazards, and no-fly zones.")
    print("Actuators: Move North, East, South, or West.")
    print("Sensors: Current node, frontier, explored set, heuristic values, and grid cell types.")


def print_trap_analysis():
    """Prints a compact trap/inefficiency summary collected during search."""
    print("\n" + "=" * 100)
    print("TRAP / INEFFICIENT REGION ANALYSIS")
    print("=" * 100)
    print(f"{'Algorithm':<14}{'Heuristic':<10}{'Heuristic Value':<18}{'Trapped At':<16}{'Next Iteration':<16}")
    any_trap = False
    for algo, data in results.items():
        for trap in data.get('traps', []):
            any_trap = True
            trapped_at = f"({trap['node'][0]},{trap['node'][1]})"
            next_iter = trap['escape'] if trap['escape'] is not None else 'N/A'
            print(f"{algo:<14}{trap['heuristic']:<10}{trap['value']:<18}{trapped_at:<16}{next_iter:<16}")
    if not any_trap:
        print("No trap-like dead-end situations were encountered in this run.")


def show_grid(grid, path=[]):
    """
    Displays the environment in a visual grid format.

    If a path is provided, the cells belonging
    to the final solution path are highlighted
    using '*'.

    Parameters:
    grid : Environment grid
    path : Final solution path
    """
    path_set = set(path)

    size = len(grid)

    boundary = "+---" * size + "+"

    print(boundary)

    for row in grid:
        print("|", end="")

        for node in row:
            if node in path_set:
                print(" * |", end="")
            else:
                print(f" {node.type} |", end="")

        print()
        print(boundary)
    print("\n")


def print_complexity_analysis():
    """
    Displays the complexity and performance metrics
    collected from all search algorithms.

    The method summarizes:

    1. Nodes Expanded
       Total number of explored nodes.

    2. Runtime
       Execution time in milliseconds.

    3. Memory Usage
       Combined size of the OPEN and CLOSED lists.

    4. Path Cost
       Total transition cost of the final path.

    5. Path Length
       Number of moves required to reach the goal.

    In addition, the method displays the
    theoretical time and space complexity
    of GBFS and A*.

    Complexity:
        GBFS
            Time  : O(V log V)
            Space : O(V)

        A*
            Time  : O(V log V)
            Space : O(V)

    Returns:
        None
    """
    print("\n" + "=" * 100)
    print("COMPLEXITY ANALYSIS")
    print("=" * 100)

    print(
        f"{'Algorithm':<15}"
        f"{'Nodes':<10}"
        f"{'Runtime(ms)':<15}"
        f"{'Memory':<10}"
        f"{'Cost':<10}"
        f"{'Length':<10}"
    )

    for algo, data in results.items():
        print(
            f"{algo:<15}"
            f"{data['nodes_expanded']:<10}"
            f"{data['runtime']:<15.3f}"
            f"{data['memory']:<10}"
            f"{data['cost']:<10}"
            f"{data['path_length']:<10}"
        )

    print("\nTHEORETICAL COMPLEXITY")
    print("GBFS Time Complexity : O(V log V)")
    print("GBFS Space Complexity: O(V)")
    print("A* Time Complexity   : O(V log V)")
    print("A* Space Complexity  : O(V)")


def compare_heuristics_gbfs():
    """
    Compares the performance of the two
    heuristic functions used by GBFS.

    Compared Heuristics:
        h1 - Euclidean Distance
        h2 - Bounding Box Risk Weighted

    The comparison includes:

    1. Nodes Expanded
       Measures search efficiency.

    2. Runtime
       Measures execution speed.

    3. Path Cost
       Measures the quality of the path found.

    The method also identifies which
    heuristic produced the better path
    based on the final path cost.

    This analysis helps evaluate whether
    the future-aware heuristic (h2)
    provides an advantage over the
    traditional Euclidean heuristic (h1).

    Returns:
        None
    """
    print("\n" + "=" * 100)
    print("GBFS HEURISTIC COMPARISON")
    print("=" * 100)

    h1 = results.get("GBFS-h1")
    h2 = results.get("GBFS-h2")

    if not h1 or not h2:
        return

    print(
        f"{'Metric':<20}"
        f"{'h1':<15}"
        f"{'h2':<15}"
    )

    print(
        f"{'Nodes Expanded':<20}"
        f"{h1['nodes_expanded']:<15}"
        f"{h2['nodes_expanded']:<15}"
    )

    print(
        f"{'Runtime(ms)':<20}"
        f"{round(h1['runtime'], 3):<15}"
        f"{round(h2['runtime'], 3):<15}"
    )

    print(
        f"{'Path Cost':<20}"
        f"{h1['cost']:<15}"
        f"{h2['cost']:<15}"
    )

    if h1["cost"] < h2["cost"]:
        print("\nh1 produced a lower cost path.")

    elif h2["cost"] < h1["cost"]:
        print("\nh2 produced a lower cost path.")

    else:
        print("\nBoth heuristics produced the same cost.")

def compare_heuristics_Astar():
    """
    Compares the performance of the two
    heuristic functions used by GBFS.

    Compared Heuristics:
        h1 - Euclidean Distance
        h2 - Bounding Box Risk Weighted

    The comparison includes:

    1. Nodes Expanded
       Measures search efficiency.

    2. Runtime
       Measures execution speed.

    3. Path Cost
       Measures the quality of the path found.

    The method also identifies which
    heuristic produced the better path
    based on the final path cost.

    This analysis helps evaluate whether
    the future-aware heuristic (h2)
    provides an advantage over the
    traditional Euclidean heuristic (h1).

    Returns:
        None
    """
    print("\n" + "=" * 100)
    print("GBFS HEURISTIC COMPARISON")
    print("=" * 100)

    h1 = results.get(f"A*-{HeuristicType.EUCLIDEAN_DISTANCE.value}")
    h2 = results.get(f"A*-{HeuristicType.BOUNDING_BOX_RISK_WEIGHTED.value}")

    if not h1 or not h2:
        print("check why heuristics are not found")
        return

    print(
        f"{'Metric':<20}"
        f"{'h1':<15}"
        f"{'h2':<15}"
    )

    print(
        f"{'Nodes Expanded':<20}"
        f"{h1['nodes_expanded']:<15}"
        f"{h2['nodes_expanded']:<15}"
    )

    print(
        f"{'Runtime(ms)':<20}"
        f"{round(h1['runtime'], 3):<15}"
        f"{round(h2['runtime'], 3):<15}"
    )

    print(
        f"{'Path Cost':<20}"
        f"{h1['cost']:<15}"
        f"{h2['cost']:<15}"
    )

    if h1["cost"] < h2["cost"]:
        print("\nh1 produced a lower cost path.")

    elif h2["cost"] < h1["cost"]:
        print("\nh2 produced a lower cost path.")

    else:
        print("\nBoth heuristics produced the same cost.")


def compare_algorithms():
    """
    Compares GBFS and A* Search algorithms.

    The comparison is performed using:

    1. Nodes Expanded
    2. Runtime
    3. Path Cost
    4. Completeness

    Returns:
        None
    """

    print("\n" + "=" * 100)
    print("GBFS vs A* COMPARISON")
    print("=" * 100)

    print(
        f"{'Algorithm':<15}"
        f"{'Nodes':<15}"
        f"{'Runtime(ms)':<15}"
        f"{'Cost':<15}"
    )

    for algo, data in results.items():
        print(
            f"{algo:<15}"
            f"{data['nodes_expanded']:<15}"
            f"{round(data['runtime'], 3):<15}"
            f"{data['cost']:<15}"
        )

    print("\nCOMPLETENESS ANALYSIS")
    print("---------------------")
    print("A*   : Complete")
    print("GBFS : Not guaranteed complete in general")

    print("\nFASTEST ALGORITHM")
    print("-----------------")

    fastest = min(
        results.items(),
        key=lambda x: x[1]["runtime"]
    )

    print(
        f"{fastest[0]} "
        f"was fastest with "
        f"{fastest[1]['runtime']:.3f} ms"
    )

    print("\nLOWEST COST PATH")
    print("----------------")

    cheapest = min(
        results.items(),
        key=lambda x: x[1]["cost"]
    )

    print(
        f"{cheapest[0]} "
        f"produced the lowest path cost "
        f"of {cheapest[1]['cost']}"
    )


def get_neighbours(grid, node: Node):
    """
    Generates all valid neighboring nodes.

    Neighbor expansion follows the mandatory
    tie-breaking order specified in the assignment:

    1. North
    2. East
    3. South
    4. West

    This ordering ensures consistent node
    selection when heuristic values are equal.

    Parameters:
    grid : Environment grid
    node : Current node

    Returns:
    List of neighboring nodes.
    """
    neighbours = []

    # NORTH
    if node.x >= 1:
        neighbours.append(grid[node.x - 1][node.y])

    # EAST
    if node.y < GRID_SIZE - 1:
        neighbours.append(grid[node.x][node.y + 1])

    # SOUTH
    if node.x < GRID_SIZE - 1:
        neighbours.append(grid[node.x + 1][node.y])

    # WEST
    if node.y >= 1:
        neighbours.append(grid[node.x][node.y - 1])

    return neighbours


def astar(grid, start_node: Node, end_node: Node, heuristic_fx: HeuristicType = HeuristicType.EUCLIDEAN_DISTANCE):
    """
    Implements the A* Search algorithm.

    Evaluation Function:
    f(n) = g(n) + h(n)

    where:
    g(n) = Path cost from start node
    h(n) = Euclidean distance heuristic

    Features:
    - Frontier management using Priority Queue
    - Explored set tracking
    - Path reconstruction
    - Runtime measurement
    - Memory usage measurement
    - Path statistics generation

    Parameters:
    grid         : Environment grid
    start_node   : Source node
    end_node     : Goal node
    heuristic_fx : Heuristic function

    Returns:
    Optimal path to goal node.
    """
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    parent = {}
    traps = []
    g_score = {}
    g_score[start_node] = 0
    parent[start_node] = None
    h_val = heuristic(grid, grid[start_node.x][start_node.y], grid[end_node.x][end_node.y], heuristic_fx)
    f_val = g_score.get(start_node) + h_val
    pq.put((f_val, next(counter), start_node))
    if pq.empty():
        print("ERROR: Frontier is empty.")
        return
    while not pq.empty():

        print("\nFRONTIER:")

        for item in list(pq.queue):
            print(f"({item[2].x},{item[2].y}) "f"h={round(item[0], 2)}")

        f_val, cnt, current_node = pq.get()

        current_cost = g_score.get(current_node)
        if current_node in visited:
            continue
        visited.add(current_node)

        print("\nEXPLORED NODES:")

        for node in visited:
            print(f"({node.x},{node.y})", end=" ")

        print("\n")

        print(f"\nSELECTED NODE : "f"({current_node.x},{current_node.y})")

        current_h = heuristic(grid, current_node, end_node, heuristic_fx)

        print(f"HEURISTIC VALUE : {round(current_h, 2)}")

        if current_node == end_node:
            path = []
            while (current_node):
                path.append(current_node)
                current_node = parent.get(current_node)

            path.reverse()

            cost = sum(node.score for node in path)

            show_grid(grid, path)

            weather_count = 0
            nofly_count = 0

            for node in path:
                if node.type == "W":
                    weather_count += 1
                elif node.type == "N":
                    nofly_count += 1

            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)

            print("Followed Path: ")
            print(" -> ".join(f"({node.x},{node.y})" for node in path))

            end_time = time.perf_counter()

            runtime_ms = (end_time - start_time) * 1000

            memory_usage = len(visited) + pq.qsize()

            results[f"A*-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": memory_usage,
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": round(current_h, 2),
                "traps": traps
            }

            goal_found = True
            break

        added_child = False
        for node in get_neighbours(grid, current_node):
            if node in visited or node.type == "N":
                continue
            new_cost = current_cost + node.score
            if node not in g_score or new_cost < g_score.get(node):
                parent[node] = current_node
                g_score[node] = new_cost
                h_val = heuristic(grid, grid[node.x][node.y], grid[end_node.x][end_node.y], heuristic_fx)
                f_val = new_cost + h_val
                pq.put((f_val, next(counter), node))
                added_child = True

        if not added_child:
            escape = None
            if not pq.empty():
                escape = f"({pq.queue[0][2].x},{pq.queue[0][2].y})"
            traps.append({
                "heuristic": heuristic_fx.value,
                "value": round(current_h, 2),
                "node": (current_node.x, current_node.y),
                "escape": escape,
            })

    if not goal_found:
        print("ERROR: Goal node could not be reached.")


def gbfs(grid, start_node: Node, end_node: Node,
         heuristic_fx: HeuristicType = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED):
    """
    Implements Greedy Best First Search.

    Node selection is based only on:
    f(n) = h(n)

    Supported heuristics:
    - Euclidean Distance (h1)
    - Bounding Box Risk Weighted (h2)

    Features:
    - Frontier management
    - Explored node tracking
    - Path reconstruction
    - Runtime measurement
    - Memory usage measurement
    - Path statistics generation

    Parameters:
    grid         : Environment grid
    start_node   : Source node
    end_node     : Goal node
    heuristic_fx : Selected heuristic

    Returns:
    Path from start node to goal node.
    """
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    explored = set()
    traps = []
    parent = {}
    parent[start_node] = None
    expansion_history = []
    explored.add(start_node)
    h_val = heuristic(grid, grid[start_node.x][start_node.y], grid[end_node.x][end_node.y], heuristic_fx)
    pq.put((h_val, next(counter), start_node))
    if pq.empty():
        print("ERROR: Frontier is empty.")
        return
    while not pq.empty():
        print("\nFRONTIER:")

        for item in list(pq.queue):
            print(f"({item[2].x},{item[2].y}) "f"h={round(item[0], 2)}")

        h_val, cnt, current_node = pq.get()

        if current_node in visited:
            continue
        visited.add(current_node)
        expansion_history.append(h_val)

        print("\nEXPLORED NODES:")

        for node in visited:
            print(f"({node.x},{node.y})", end=" ")

        print("\n")

        print(f"\nSELECTED NODE : "f"({current_node.x},{current_node.y})")

        current_h = heuristic(grid, current_node, end_node, heuristic_fx)

        print(f"HEURISTIC VALUE : "f"{round(current_h, 2)}")

        if current_node == end_node:
            path = []
            while (current_node):
                path.append(current_node)
                current_node = parent.get(current_node)

            path.reverse()

            cost = sum(node.score for node in path)
            path_history = [heuristic(grid, n, end_node, heuristic_fx) for n in path]

            show_grid(grid, path)

            weather_count = 0
            nofly_count = 0

            for node in path:
                if node.type == "W":
                    weather_count += 1
                elif node.type == "N":
                    nofly_count += 1

            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)

            print("Followed Path: ")
            print(" -> ".join(f"({node.x},{node.y})" for node in path))

            end_time = time.perf_counter()

            runtime_ms = (end_time - start_time) * 1000

            memory_usage = len(visited) + pq.qsize()

            results[f"GBFS-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": memory_usage,
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": round(current_h, 2),
                "expansion_history": expansion_history,
                "path_history": path_history,
                "traps": traps
            }

            goal_found = True
            break

        added_child = False
        for node in get_neighbours(grid, current_node):
            if node.type == "N":
                continue
            if node not in visited:
                #explored.add(node)
                if node not in parent:
                    parent[node] = current_node
                    h_val = heuristic(grid, grid[node.x][node.y], grid[end_node.x][end_node.y], heuristic_fx)
                    pq.put((h_val, next(counter), node))
                    added_child = True

        if not added_child:
            escape = None
            if not pq.empty():
                escape = f"({pq.queue[0][2].x},{pq.queue[0][2].y})"
            traps.append({
                "heuristic": heuristic_fx.value,
                "value": round(current_h, 2),
                "node": (current_node.x, current_node.y),
                "escape": escape,
            })

    if not goal_found:
        print("ERROR: Goal node could not be reached.")

def print_text_chart(title: str, history: list):
    print("=" * 70)
    print(f" VISUAL TREND RESTRUCTURING: {title}")
    print("=" * 70)
    if not history:
        return
    max_val = max(history) if max(history) > 0 else 1
    max_bars = 35
    print(f"{'Step / Index':<12} | {'Heuristic':<12} | Relative Visual Scale")
    print("-" * 70)
    for idx, val in enumerate(history):
        bar_length = int((val / max_val) * max_bars)
        print(f"Index {idx:<6} | {val:<12.2f} | {'█' * bar_length}")
    print("=" * 70 + "\n")

def main():
    """
    Driver function of the program.

    Execution Flow:
    1. Build environment grid.
    2. Display initial environment.
    3. Execute GBFS using h2.
    4. Execute GBFS using h1.
    5. Execute A* using h1.
    6. Display search statistics and results.

    This function serves as the entry point
    of the application.
    """
    output_path = DEFAULT_OUTPUT_FILE
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE

    parsed = load_grid_from_file(input_path)
    if parsed:
        grid, start_node, end_node = parsed
    else:
        grid = build_grid(GRID_SIZE, 0, 0, 6, 7)
        start_node = grid[0][0]
        end_node = grid[6][7]

    original_stdout = sys.stdout
    with open(output_path, 'w', encoding='utf-8') as output_file:
        sys.stdout = Tee(original_stdout, output_file)
        try:
            print("=" * 100)
            print("=" * 100)

            print("\nInitial Grid\n")

            show_grid(grid)

            print("=" * 100)
            print("=" * 100)

            print("\nGBFS USING H2\n")
            gbfs(grid, start_node, end_node, HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)

            print("=" * 100)
            print("=" * 100)

            print("\nGBFS USING H1\n")
            gbfs(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE)

            print("=" * 100)
            print("=" * 100)

            print("\nA* USING H1\n")
            astar(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE)

            print("\nA* using H2")
            astar(grid=grid, start_node=start_node, end_node=end_node, heuristic_fx=HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)

            print("=" * 100)

            print_peas_components()
            print_complexity_analysis()

            print("=" * 100)

            compare_heuristics_gbfs()
            compare_heuristics_Astar()

            print("=" * 100)

            compare_algorithms()
            print_trap_analysis()

            print_text_chart("Heuristic Values to Reach Target along Path (h1) [5.b]", results["GBFS-h1"]["path_history"])
            print_text_chart("Heuristic Values vs Chronological Expansions (h1) [5.c]", results["GBFS-h1"]["expansion_history"])
            print_text_chart("Heuristic Values to Reach Target along Path (h2) [5.b]", results["GBFS-h2"]["path_history"])
            print_text_chart("Heuristic Values vs Chronological Expansions (h2) [5.c]", results["GBFS-h2"]["expansion_history"])
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
