from enum import Enum
import argparse
import itertools
from math import sqrt
from pathlib import Path
from typing import List, Optional
import queue
import time
import contextlib
import sys
import re

GRID_SIZE = 8
results = {}
trap_records = []


class NodeType(Enum):
    """Cell types and traversal costs."""
    PASSABLE_AIRSPACE = (".", 1)
    WEATHER_HAZARD = ("W", 4)
    NO_FLY_ZONE = ("N", 8)
    START = ("S", 2)
    END = ("E", 2)


class MovementModel(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class Node:
    def __init__(self, x: int, y: int, type: NodeType) -> None:
        self.x = x
        self.y = y
        self.type = type[0]
        self.score = type[1]

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Node({self.x},{self.y},{self.type})"


class HeuristicType(Enum):
    EUCLIDEAN_DISTANCE = "h1"
    BOUNDING_BOX_RISK_WEIGHTED = "h2"


def euclidean_distance(current_state: Node, goal_state: Node):
    return sqrt(((goal_state.x - current_state.x) * (goal_state.x - current_state.x) +
                 (goal_state.y - current_state.y) * (goal_state.y - current_state.y)))


def manhatten_distance(current_state: Node, goal_state: Node):
    return abs(goal_state.x - current_state.x) + abs(goal_state.y - current_state.y)


# keep the original function name for minimal change

def bounding_box_risk_weighted(grid: List[List[Node]], current_state: Node, goal_state: Node):
    manhatten = manhatten_distance(current_state, goal_state)
    k = (abs(goal_state.x - current_state.x) + 1) * (abs(goal_state.y - current_state.y) + 1)
    x_min = min(current_state.x, goal_state.x)
    x_max = max(current_state.x, goal_state.x)
    y_min = min(current_state.y, goal_state.y)
    y_max = max(current_state.y, goal_state.y)
    partial_score = 0
    for row in range(x_min, x_max + 1):
        for col in range(y_min, y_max + 1):
            partial_score += grid[row][col].score
    return manhatten * (partial_score / k)


def heuristic(grid, current_state: Node, goal_state: Node, type: HeuristicType):
    if type == HeuristicType.EUCLIDEAN_DISTANCE:
        return euclidean_distance(current_state, goal_state)
    elif type == HeuristicType.BOUNDING_BOX_RISK_WEIGHTED:
        return bounding_box_risk_weighted(grid, current_state, goal_state)
    raise ValueError(f"Unsupported heuristic: {type}")


def parse_layout_file(path: Path) -> List[List[str]]:
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        symbols = re.findall(r"[SENW\.]", line)
        if symbols:
            rows.append(symbols)
    if not rows:
        raise ValueError("No grid symbols found in input file.")
    target_len = max(set(len(r) for r in rows), key=lambda n: sum(1 for r in rows if len(r) == n))
    rows = [r for r in rows if len(r) == target_len]
    if any(len(r) != target_len for r in rows):
        raise ValueError("Input grid rows are not consistent.")
    return rows


def build_grid(size: int, start_x, start_y, end_x, end_y, layout: Optional[List[List[str]]] = None):
    global GRID_SIZE
    if layout is not None:
        size = len(layout)
    GRID_SIZE = size
    grid = [[Node(x, y, NodeType.PASSABLE_AIRSPACE.value) for y in range(size)] for x in range(size)]

    if layout is not None:
        for x in range(size):
            for y in range(size):
                symbol = layout[x][y]
                if symbol == "S":
                    grid[x][y] = Node(x, y, NodeType.START.value)
                elif symbol == "E":
                    grid[x][y] = Node(x, y, NodeType.END.value)
                elif symbol == "W":
                    grid[x][y] = Node(x, y, NodeType.WEATHER_HAZARD.value)
                elif symbol == "N":
                    grid[x][y] = Node(x, y, NodeType.NO_FLY_ZONE.value)
                else:
                    grid[x][y] = Node(x, y, NodeType.PASSABLE_AIRSPACE.value)
        return grid

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


def show_grid(grid, path=[]):
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


def get_neighbours(grid, node: Node):
    neighbours = []
    # North, East, South, West (assignment order)
    if node.x >= 1:
        neighbours.append(grid[node.x - 1][node.y])
    if node.y < GRID_SIZE - 1:
        neighbours.append(grid[node.x][node.y + 1])
    if node.x < GRID_SIZE - 1:
        neighbours.append(grid[node.x + 1][node.y])
    if node.y >= 1:
        neighbours.append(grid[node.x][node.y - 1])
    return neighbours


def _pq_snapshot(pq):
    return [(item[2], item[0]) for item in sorted(list(pq.queue), key=lambda x: (x[0], x[1]))]


def _path_cost(path):
    return sum(node.score for node in path)


def _path_hazard_counts(path):
    weather_count = 0
    nofly_count = 0
    for node in path:
        if node.type == "W":
            weather_count += 1
        elif node.type == "N":
            nofly_count += 1
    return weather_count, nofly_count


def _store_result(key, nodes_expanded, runtime_ms, memory_usage, cost, path_length, heuristic_value, extra=None):
    results[key] = {
        "nodes_expanded": nodes_expanded,
        "runtime": runtime_ms,
        "memory": memory_usage,
        "cost": cost,
        "path_length": path_length,
        "heuristic": heuristic_value,
    }
    if extra:
        results[key].update(extra)


def _record_trap(algo, heuristic_fx, node, hval):
    trap_records.append({
        "algorithm": algo,
        "heuristic": heuristic_fx.value,
        "trapped_at": f"({node.x},{node.y})",
        "heuristic_value": round(hval, 2),
        "next_iteration": "Frontier will continue with the next best node",
    })


def astar(grid, start_node: Node, end_node: Node, heuristic_fx: HeuristicType = HeuristicType.EUCLIDEAN_DISTANCE):
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    parent = {}
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
        for item in _pq_snapshot(pq):
            print(f"({item[0].x},{item[0].y}) h={round(item[1], 2)}")
        f_val, cnt, current_node = pq.get()
        current_cost = g_score.get(current_node)
        if current_node in visited:
            continue
        visited.add(current_node)
        print("\nEXPLORED NODES:")
        for node in visited:
            print(f"({node.x},{node.y})", end=" ")
        print("\n")
        print(f"\nSELECTED NODE : ({current_node.x},{current_node.y})")
        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        print(f"HEURISTIC VALUE : {round(current_h, 2)}")
        next_moves = [n for n in get_neighbours(grid, current_node) if n not in visited and n.type != "N"]
        if not next_moves and current_node != end_node:
            _record_trap("A*", heuristic_fx, current_node, current_h)
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent.get(current_node)
            path.reverse()
            cost = _path_cost(path)
            show_grid(grid, path)
            weather_count, nofly_count = _path_hazard_counts(path)
            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)
            print("Followed Path: ")
            print(" -> ".join(f"({node.x},{node.y})" for node in path))
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            memory_usage = len(visited) + pq.qsize()
            _store_result(f"A*-{heuristic_fx.value}", len(visited), runtime_ms, memory_usage,
                          cost, len(path) - 1, round(current_h, 2))
            goal_found = True
            break
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
    if not goal_found:
        print("ERROR: Goal node could not be reached.")


def gbfs(grid, start_node: Node, end_node: Node, heuristic_fx: HeuristicType = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED):
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    explored = set()
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
        for item in _pq_snapshot(pq):
            print(f"({item[0].x},{item[0].y}) h={round(item[1], 2)}")
        h_val, cnt, current_node = pq.get()
        if current_node in visited:
            continue
        visited.add(current_node)
        expansion_history.append(h_val)
        print("\nEXPLORED NODES:")
        for node in visited:
            print(f"({node.x},{node.y})", end=" ")
        print("\n")
        print(f"\nSELECTED NODE : ({current_node.x},{current_node.y})")
        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        print(f"HEURISTIC VALUE : {round(current_h, 2)}")
        next_moves = [n for n in get_neighbours(grid, current_node) if n not in visited and n.type != "N"]
        if not next_moves and current_node != end_node:
            _record_trap("GBFS", heuristic_fx, current_node, current_h)
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent.get(current_node)
            path.reverse()
            cost = _path_cost(path)
            path_history = [heuristic(grid, n, end_node, heuristic_fx) for n in path]
            show_grid(grid, path)
            weather_count, nofly_count = _path_hazard_counts(path)
            print("WEATHER CELLS:", weather_count)
            print("NO FLY CELLS:", nofly_count)
            print("Followed Path: ")
            print(" -> ".join(f"({node.x},{node.y})" for node in path))
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            memory_usage = len(visited) + pq.qsize()
            _store_result(f"GBFS-{heuristic_fx.value}", len(visited), runtime_ms, memory_usage,
                          cost, len(path) - 1, round(current_h, 2),
                          extra={"expansion_history": expansion_history, "path_history": path_history})
            goal_found = True
            break
        for node in get_neighbours(grid, current_node):
            if node.type == "N":
                continue
            if node not in visited:
                if node not in parent:
                    parent[node] = current_node
                    h_val = heuristic(grid, grid[node.x][node.y], grid[end_node.x][end_node.y], heuristic_fx)
                    pq.put((h_val, next(counter), node))
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


def print_complexity_analysis():
    print("\n" + "=" * 100)
    print("COMPLEXITY ANALYSIS")
    print("=" * 100)
    print(f"{'Algorithm':<15}{'Nodes':<10}{'Runtime(ms)':<15}{'Memory':<10}{'Cost':<10}{'Length':<10}")
    for algo, data in results.items():
        print(f"{algo:<15}{data['nodes_expanded']:<10}{data['runtime']:<15.3f}{data['memory']:<10}{data['cost']:<10}{data['path_length']:<10}")
    print("\nTHEORETICAL COMPLEXITY")
    print("GBFS Time Complexity : O(V log V)")
    print("GBFS Space Complexity: O(V)")
    print("A* Time Complexity   : O(V log V)")
    print("A* Space Complexity  : O(V)")


def compare_heuristics_gbfs():
    print("\n" + "=" * 100)
    print("GBFS HEURISTIC COMPARISON")
    print("=" * 100)
    h1 = results.get("GBFS-h1")
    h2 = results.get("GBFS-h2")
    if not h1 or not h2:
        return
    print(f"{'Metric':<20}{'h1':<15}{'h2':<15}")
    print(f"{'Nodes Expanded':<20}{h1['nodes_expanded']:<15}{h2['nodes_expanded']:<15}")
    print(f"{'Runtime(ms)':<20}{round(h1['runtime'], 3):<15}{round(h2['runtime'], 3):<15}")
    print(f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    if h1['cost'] < h2['cost']:
        print("\nh1 produced a lower cost path.")
    elif h2['cost'] < h1['cost']:
        print("\nh2 produced a lower cost path.")
    else:
        print("\nBoth heuristics produced the same cost.")


def compare_heuristics_Astar():
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
    print(f"{'Runtime(ms)':<20}{round(h1['runtime'], 3):<15}{round(h2['runtime'], 3):<15}")
    print(f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    if h1['cost'] < h2['cost']:
        print("\nh1 produced a lower cost path.")
    elif h2['cost'] < h1['cost']:
        print("\nh2 produced a lower cost path.")
    else:
        print("\nBoth heuristics produced the same cost.")


def compare_algorithms():
    print("\n" + "=" * 100)
    print("GBFS vs A* COMPARISON")
    print("=" * 100)
    print(f"{'Algorithm':<15}{'Nodes':<15}{'Runtime(ms)':<15}{'Cost':<15}")
    for algo, data in results.items():
        print(f"{algo:<15}{data['nodes_expanded']:<15}{round(data['runtime'], 3):<15}{data['cost']:<15}")
    print("\nCOMPLETENESS ANALYSIS")
    print("---------------------")
    print("A*   : Complete")
    print("GBFS : Not guaranteed complete in general")
    print("\nFASTEST ALGORITHM")
    print("-----------------")
    fastest = min(results.items(), key=lambda x: x[1]['runtime'])
    print(f"{fastest[0]} was fastest with {fastest[1]['runtime']:.3f} ms")
    print("\nLOWEST COST PATH")
    print("----------------")
    cheapest = min(results.items(), key=lambda x: x[1]['cost'])
    print(f"{cheapest[0]} produced the lowest path cost of {cheapest[1]['cost']}")


def print_trap_table():
    print("\n" + "=" * 100)
    print("TRAP / INEFFICIENT REGION ANALYSIS")
    print("=" * 100)
    if not trap_records:
        print("No dead-end or trap states were encountered on this grid.")
        return
    print(f"{'Algo':<10}{'Heuristic':<12}{'Trapped At':<15}{'Heuristic Value':<18}{'Next Iteration':<35}")
    for rec in trap_records:
        print(f"{rec['algorithm']:<10}{rec['heuristic']:<12}{rec['trapped_at']:<15}{rec['heuristic_value']:<18}{rec['next_iteration']:<35}")


def print_peas():
    print("\n" + "=" * 100)
    print("PEAS DESCRIPTION")
    print("=" * 100)
    print("Performance: Reach the goal with minimum heuristic-guided path cost while avoiding no-fly zones.")
    print("Environment: 8x8 grid with passable cells, weather hazards, and no-fly zones.")
    print("Actuators: Move North, East, South, West.")
    print("Sensors: Current cell type, neighboring cell types, heuristic values, frontier status.")


def print_execution_flow():
    print("\n" + "=" * 100)
    print("EXECUTION FLOW")
    print("=" * 100)
    print("1. Read or build the grid.")
    print("2. Display the initial environment.")
    print("3. Run GBFS with h2.")
    print("4. Run GBFS with h1.")
    print("5. Run A* with h1.")
    print("6. Run A* with h2.")
    print("7. Print metric tables, trap analysis, and comparisons.")


def run(grid):
    print("=" * 100)
    print("Initial Grid\n")
    show_grid(grid)
    print("=" * 100)
    print("\nGBFS USING H2\n")
    gbfs(grid, grid[0][0], grid[6][7], HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)
    print("=" * 100)
    print("\nGBFS USING H1\n")
    gbfs(grid, grid[0][0], grid[6][7], HeuristicType.EUCLIDEAN_DISTANCE)
    print("=" * 100)
    print("\nA* USING H1\n")
    astar(grid, grid[0][0], grid[6][7], HeuristicType.EUCLIDEAN_DISTANCE)
    print("\nA* USING H2")
    astar(grid=grid, start_node=grid[0][0], end_node=grid[6][7], heuristic_fx=HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)
    print("=" * 100)
    print_execution_flow()
    print_peas()
    print_complexity_analysis()
    print("=" * 100)
    compare_heuristics_gbfs()
    compare_heuristics_Astar()
    print("=" * 100)
    compare_algorithms()
    print_trap_table()
    if "GBFS-h1" in results:
        print_text_chart("Heuristic Values to Reach Target along Path (h1) [5.b]", results["GBFS-h1"].get("path_history", []))
        print_text_chart("Heuristic Values vs Chronological Expansions (h1) [5.c]", results["GBFS-h1"].get("expansion_history", []))
    if "GBFS-h2" in results:
        print_text_chart("Heuristic Values to Reach Target along Path (h2) [5.b]", results["GBFS-h2"].get("path_history", []))
        print_text_chart("Heuristic Values vs Chronological Expansions (h2) [5.c]", results["GBFS-h2"].get("expansion_history", []))


def main():
    parser = argparse.ArgumentParser(description="Defense Drone GBFS/A* solution")
    parser.add_argument("--input", type=str, default="", help="Input grid text file")
    parser.add_argument("--output", type=str, default="outputPSXX.txt", help="Output report file")
    args = parser.parse_args()

    if args.input:
        layout = parse_layout_file(Path(args.input))
        grid = build_grid(len(layout), 0, 0, 0, 0, layout=layout)
    else:
        grid = build_grid(GRID_SIZE, 0, 0, 6, 7)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(output_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(Tee(sys.stdout, f)):
            run(grid)


if __name__ == "__main__":
    main()
