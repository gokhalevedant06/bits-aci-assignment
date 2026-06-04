from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import argparse
import itertools
import queue
import sys
import time
from math import sqrt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any

GRID_SIZE = 8
DEFAULT_START = (0, 0)
DEFAULT_GOAL = (6, 7)
DEFAULT_INPUT_CANDIDATES = ["inputPS4.txt", "inputPSXX.txt", "input.txt"]

results: Dict[str, Dict[str, Any]] = {}
trap_results: Dict[str, List[Dict[str, Any]]] = {}


class NodeType(Enum):
    """Cell types in the environment and their traversal costs."""
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


class HeuristicType(Enum):
    EUCLIDEAN_DISTANCE = "h1"
    BOUNDING_BOX_RISK_WEIGHTED = "h2"


@dataclass(frozen=True)
class Node:
    x: int
    y: int
    type: str
    score: int

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class SearchStep:
    iteration: int
    selected_node: Tuple[int, int]
    heuristic_value: float
    frontier_snapshot: List[Tuple[int, int, float]]
    explored_snapshot: List[Tuple[int, int]]
    trap_detected: bool = False
    trap_reason: str = ""


def node_from_type(x: int, y: int, node_type: NodeType) -> Node:
    return Node(x, y, node_type.value[0], node_type.value[1])


def build_default_grid(size: int, start_x: int, start_y: int, end_x: int, end_y: int) -> List[List[Node]]:
    grid = [[node_from_type(x, y, NodeType.PASSABLE_AIRSPACE) for y in range(size)] for x in range(size)]

    grid[start_x][start_y] = node_from_type(start_x, start_y, NodeType.START)
    grid[end_x][end_y] = node_from_type(end_x, end_y, NodeType.END)

    weather_hazards = [
        (1, 1),
        (2, 3), (2, 6),
        (4, 3),
        (6, 2),
        (7, 3), (7, 5)
    ]
    for x, y in weather_hazards:
        grid[x][y] = node_from_type(x, y, NodeType.WEATHER_HAZARD)

    no_fly_zones = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 0), (3, 1), (3, 6),
        (5, 5), (5, 6)
    ]
    for x, y in no_fly_zones:
        grid[x][y] = node_from_type(x, y, NodeType.NO_FLY_ZONE)

    return grid


def parse_coordinate_line(text: str) -> Optional[Tuple[int, int]]:
    tokens = text.replace(",", " ").replace("(", " ").replace(")", " ").split()
    if len(tokens) < 2:
        return None
    try:
        return int(tokens[-2]), int(tokens[-1])
    except ValueError:
        return None


def parse_input_file(path: Path) -> Tuple[List[List[Node]], Tuple[int, int], Tuple[int, int]]:
    """Parse a flexible input format.

    Supported forms:
    1) Plain 8 lines of 8 symbols each (S, E, ., W, N)
    2) Optional header lines with SIZE / START / GOAL
    3) Grid lines with spaces or compact rows

    If parsing fails, the built-in default sample grid is returned.
    """
    if not path.exists():
        return build_default_grid(GRID_SIZE, *DEFAULT_START, *DEFAULT_GOAL), DEFAULT_START, DEFAULT_GOAL

    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        return build_default_grid(GRID_SIZE, *DEFAULT_START, *DEFAULT_GOAL), DEFAULT_START, DEFAULT_GOAL

    size = GRID_SIZE
    start = DEFAULT_START
    goal = DEFAULT_GOAL
    grid_tokens: List[List[str]] = []

    for line in raw_lines:
        upper = line.upper()
        if upper.startswith("SIZE"):
            try:
                size = int(line.replace("=", " ").split()[-1])
            except Exception:
                pass
            continue
        if upper.startswith("START"):
            coord = parse_coordinate_line(line)
            if coord:
                start = coord
            continue
        if upper.startswith("GOAL") or upper.startswith("END"):
            coord = parse_coordinate_line(line)
            if coord:
                goal = coord
            continue

        tokens = line.replace("|", " ").replace(",", " ").split()
        row_symbols = [tok for tok in tokens if tok in {"S", "E", ".", "W", "N"}]
        if len(row_symbols) == size:
            grid_tokens.append(row_symbols)
            continue

        compact = [ch for ch in line if ch in {"S", "E", ".", "W", "N"}]
        if len(compact) == size:
            grid_tokens.append(compact)

    if len(grid_tokens) != size:
        return build_default_grid(GRID_SIZE, *DEFAULT_START, *DEFAULT_GOAL), DEFAULT_START, DEFAULT_GOAL

    mapping = {
        ".": NodeType.PASSABLE_AIRSPACE,
        "W": NodeType.WEATHER_HAZARD,
        "N": NodeType.NO_FLY_ZONE,
        "S": NodeType.START,
        "E": NodeType.END,
    }

    grid = [[None for _ in range(size)] for _ in range(size)]
    found_start = None
    found_goal = None

    for i in range(size):
        for j in range(size):
            symbol = grid_tokens[i][j]
            ntype = mapping[symbol]
            node = node_from_type(i, j, ntype)
            grid[i][j] = node
            if symbol == "S":
                found_start = (i, j)
            elif symbol == "E":
                found_goal = (i, j)

    if found_start is None:
        found_start = start
        grid[start[0]][start[1]] = node_from_type(start[0], start[1], NodeType.START)
    if found_goal is None:
        found_goal = goal
        grid[goal[0]][goal[1]] = node_from_type(goal[0], goal[1], NodeType.END)

    # Preserve explicit start/goal if present in headers.
    return grid, found_start, found_goal


def show_grid(grid: List[List[Node]], path: Sequence[Node] = ()):  # noqa: B006
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
    print()


def format_path(path: Sequence[Node]) -> str:
    return " -> ".join(f"({n.x},{n.y})" for n in path)


def node_key(node: Node) -> Tuple[int, int]:
    return node.x, node.y


def euclidean_distance(current_state: Node, goal_state: Node) -> float:
    return sqrt((goal_state.x - current_state.x) ** 2 + (goal_state.y - current_state.y) ** 2)


def manhattan_distance(current_state: Node, goal_state: Node) -> int:
    return abs(goal_state.x - current_state.x) + abs(goal_state.y - current_state.y)


def bounding_box_risk_weighted(grid: List[List[Node]], current_state: Node, goal_state: Node) -> float:
    manhattan = manhattan_distance(current_state, goal_state)
    k = (abs(goal_state.x - current_state.x) + 1) * (abs(goal_state.y - current_state.y) + 1)
    x_min, x_max = min(current_state.x, goal_state.x), max(current_state.x, goal_state.x)
    y_min, y_max = min(current_state.y, goal_state.y), max(current_state.y, goal_state.y)
    partial_score = 0
    for row in range(x_min, x_max + 1):
        for col in range(y_min, y_max + 1):
            partial_score += grid[row][col].score
    return manhattan * (partial_score / k)


def heuristic(grid: List[List[Node]], current_state: Node, goal_state: Node, htype: HeuristicType) -> float:
    if htype == HeuristicType.EUCLIDEAN_DISTANCE:
        return euclidean_distance(current_state, goal_state)
    if htype == HeuristicType.BOUNDING_BOX_RISK_WEIGHTED:
        return bounding_box_risk_weighted(grid, current_state, goal_state)
    raise ValueError(f"Unsupported heuristic: {htype}")


def get_neighbours(grid: List[List[Node]], node: Node) -> List[Node]:
    """Return neighbours in mandatory order: North, East, South, West."""
    neighbours = []
    if node.x >= 1:
        neighbours.append(grid[node.x - 1][node.y])
    if node.y < GRID_SIZE - 1:
        neighbours.append(grid[node.x][node.y + 1])
    if node.x < GRID_SIZE - 1:
        neighbours.append(grid[node.x + 1][node.y])
    if node.y >= 1:
        neighbours.append(grid[node.x][node.y - 1])
    return neighbours


def path_cost(path: Sequence[Node]) -> int:
    # Transition-cost style metric: exclude the start cell.
    return sum(node.score for node in path[1:])


def trap_summary(traps: List[Dict[str, Any]]) -> str:
    if not traps:
        return "No dead-end / inefficient trap encountered in the chosen run."
    lines = ["heuristic | heuristic_value | trapped_at_node | next_iteration_to_come_out | exit_node"]
    for row in traps:
        lines.append(
            f"{row['heuristic']} | {row['heuristic_value']:.2f} | {row['trapped_at_node']} | {row['next_iteration_to_come_out']} | {row['exit_node']}"
        )
    return "\n".join(lines)


def record_step(history: List[SearchStep], iteration: int, selected: Node, h_value: float,
                frontier: List[Tuple[float, int, Node]], visited: set, trap_detected: bool = False,
                trap_reason: str = ""):
    history.append(
        SearchStep(
            iteration=iteration,
            selected_node=node_key(selected),
            heuristic_value=float(h_value),
            frontier_snapshot=[(n.x, n.y, float(v)) for (v, _, n) in frontier],
            explored_snapshot=[node_key(n) for n in visited],
            trap_detected=trap_detected,
            trap_reason=trap_reason,
        )
    )


def write_result_line(outfile, text: str = ""):
    print(text)
    outfile.write(text + "\n")


def run_gbfs(grid: List[List[Node]], start_node: Node, end_node: Node, heuristic_fx: HeuristicType,
             outfile, log_prefix: str = "GBFS"):
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    explored = set()
    parent: Dict[Node, Optional[Node]] = {start_node: None}
    expansion_history: List[float] = []
    step_history: List[SearchStep] = []
    traps: List[Dict[str, Any]] = []
    pending_trap: Optional[Dict[str, Any]] = None

    explored.add(start_node)
    h_val = heuristic(grid, start_node, end_node, heuristic_fx)
    pq.put((h_val, next(counter), start_node))

    if pq.empty():
        raise RuntimeError("Frontier is empty at start.")

    iteration = 0
    while not pq.empty():
        iteration += 1
        frontier_snapshot = list(pq.queue)
        h_val, _, current_node = pq.get()
        if current_node in visited:
            continue
        visited.add(current_node)
        expansion_history.append(float(h_val))

        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        if pending_trap is not None:
            pending_trap["next_iteration_to_come_out"] = iteration
            pending_trap["exit_node"] = node_key(current_node)
            traps.append(pending_trap)
            pending_trap = None

        # Log frontier and explored sets for each iteration.
        write_result_line(outfile, f"FRONTIER (iteration {iteration}):")
        for item in frontier_snapshot:
            write_result_line(outfile, f"  ({item[2].x},{item[2].y}) h={item[0]:.2f}")
        write_result_line(outfile, "EXPLORED NODES:")
        write_result_line(outfile, "  " + " ".join(f"({n.x},{n.y})" for n in visited))
        write_result_line(outfile, f"SELECTED NODE : ({current_node.x},{current_node.y})")
        write_result_line(outfile, f"HEURISTIC VALUE : {current_h:.2f}")

        record_step(step_history, iteration, current_node, current_h, frontier_snapshot, visited)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent.get(current_node)
            path.reverse()

            show_grid(grid, path)
            write_result_line(outfile, "Updated grid with final path shown above.")
            write_result_line(outfile, "Followed Path:")
            write_result_line(outfile, format_path(path))

            weather_count = sum(1 for node in path if node.type == "W")
            nofly_count = sum(1 for node in path if node.type == "N")
            write_result_line(outfile, f"WEATHER CELLS: {weather_count}")
            write_result_line(outfile, f"NO FLY CELLS: {nofly_count}")

            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            memory_usage = len(visited) + pq.qsize()
            cost = path_cost(path)

            results[f"{log_prefix}-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": memory_usage,
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": round(current_h, 2),
                "expansion_history": expansion_history,
                "path_history": [heuristic(grid, n, end_node, heuristic_fx) for n in path],
                "steps": step_history,
            }
            trap_results[f"{log_prefix}-{heuristic_fx.value}"] = traps
            goal_found = True
            break

        valid_children = 0
        for node in get_neighbours(grid, current_node):
            if node.type == "N" or node in visited:
                continue
            valid_children += 1
            if node not in explored:
                explored.add(node)
                parent[node] = current_node
                h_child = heuristic(grid, node, end_node, heuristic_fx)
                pq.put((h_child, next(counter), node))

        if valid_children == 0 and current_node != end_node:
            pending_trap = {
                "heuristic": heuristic_fx.value,
                "heuristic_value": current_h,
                "trapped_at_node": node_key(current_node),
                "next_iteration_to_come_out": "PENDING",
                "exit_node": "PENDING",
            }

    if not goal_found:
        write_result_line(outfile, "ERROR: Goal node could not be reached.")


def run_astar(grid: List[List[Node]], start_node: Node, end_node: Node, heuristic_fx: HeuristicType, outfile):
    goal_found = False
    start_time = time.perf_counter()
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    parent: Dict[Node, Optional[Node]] = {}
    g_score: Dict[Node, float] = {}
    g_score[start_node] = 0
    parent[start_node] = None
    h_val = heuristic(grid, start_node, end_node, heuristic_fx)
    f_val = g_score[start_node] + h_val
    pq.put((f_val, next(counter), start_node))

    if pq.empty():
        raise RuntimeError("Frontier is empty at start.")

    iteration = 0
    while not pq.empty():
        iteration += 1
        frontier_snapshot = list(pq.queue)
        f_val, _, current_node = pq.get()
        current_cost = g_score.get(current_node, float("inf"))
        if current_node in visited:
            continue
        visited.add(current_node)

        current_h = heuristic(grid, current_node, end_node, heuristic_fx)
        write_result_line(outfile, f"FRONTIER (iteration {iteration}):")
        for item in frontier_snapshot:
            write_result_line(outfile, f"  ({item[2].x},{item[2].y}) f={item[0]:.2f}")
        write_result_line(outfile, "EXPLORED NODES:")
        write_result_line(outfile, "  " + " ".join(f"({n.x},{n.y})" for n in visited))
        write_result_line(outfile, f"SELECTED NODE : ({current_node.x},{current_node.y})")
        write_result_line(outfile, f"HEURISTIC VALUE : {current_h:.2f}")

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent.get(current_node)
            path.reverse()

            show_grid(grid, path)
            write_result_line(outfile, "Updated grid with final path shown above.")
            write_result_line(outfile, "Followed Path:")
            write_result_line(outfile, format_path(path))

            weather_count = sum(1 for node in path if node.type == "W")
            nofly_count = sum(1 for node in path if node.type == "N")
            write_result_line(outfile, f"WEATHER CELLS: {weather_count}")
            write_result_line(outfile, f"NO FLY CELLS: {nofly_count}")

            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            memory_usage = len(visited) + pq.qsize()
            cost = path_cost(path)

            results[f"A*-{heuristic_fx.value}"] = {
                "nodes_expanded": len(visited),
                "runtime": runtime_ms,
                "memory": memory_usage,
                "cost": cost,
                "path_length": len(path) - 1,
                "heuristic": round(current_h, 2),
            }
            goal_found = True
            break

        for node in get_neighbours(grid, current_node):
            if node in visited or node.type == "N":
                continue
            new_cost = current_cost + node.score
            if node not in g_score or new_cost < g_score[node]:
                parent[node] = current_node
                g_score[node] = new_cost
                h_child = heuristic(grid, node, end_node, heuristic_fx)
                f_child = new_cost + h_child
                pq.put((f_child, next(counter), node))

    if not goal_found:
        write_result_line(outfile, "ERROR: Goal node could not be reached.")


def print_complexity_analysis(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "COMPLEXITY ANALYSIS")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, f"{'Algorithm':<15}{'Nodes':<10}{'Runtime(ms)':<15}{'Memory':<10}{'Cost':<10}{'Length':<10}")
    for algo, data in results.items():
        write_result_line(
            outfile,
            f"{algo:<15}{data['nodes_expanded']:<10}{data['runtime']:<15.3f}{data['memory']:<10}{data['cost']:<10}{data['path_length']:<10}",
        )
    write_result_line(outfile, "")
    write_result_line(outfile, "THEORETICAL COMPLEXITY")
    write_result_line(outfile, "GBFS Time Complexity : O(V log V)")
    write_result_line(outfile, "GBFS Space Complexity: O(V)")
    write_result_line(outfile, "A* Time Complexity   : O(V log V)")
    write_result_line(outfile, "A* Space Complexity  : O(V)")


def compare_heuristics_gbfs(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "GBFS HEURISTIC COMPARISON")
    write_result_line(outfile, "=" * 100)
    h1 = results.get("GBFS-h1")
    h2 = results.get("GBFS-h2")
    if not h1 or not h2:
        write_result_line(outfile, "Missing GBFS heuristic results.")
        return
    write_result_line(outfile, f"{'Metric':<20}{'h1':<15}{'h2':<15}")
    write_result_line(outfile, f"{'Nodes Expanded':<20}{h1['nodes_expanded']:<15}{h2['nodes_expanded']:<15}")
    write_result_line(outfile, f"{'Runtime(ms)':<20}{round(h1['runtime'], 3):<15}{round(h2['runtime'], 3):<15}")
    write_result_line(outfile, f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    if h1["cost"] < h2["cost"]:
        write_result_line(outfile, "h1 produced a lower cost path.")
    elif h2["cost"] < h1["cost"]:
        write_result_line(outfile, "h2 produced a lower cost path.")
    else:
        write_result_line(outfile, "Both heuristics produced the same cost.")


def compare_heuristics_astar(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "A* HEURISTIC COMPARISON")
    write_result_line(outfile, "=" * 100)
    h1 = results.get("A*-h1")
    h2 = results.get("A*-h2")
    if not h1 or not h2:
        write_result_line(outfile, "Missing A* heuristic results.")
        return
    write_result_line(outfile, f"{'Metric':<20}{'h1':<15}{'h2':<15}")
    write_result_line(outfile, f"{'Nodes Expanded':<20}{h1['nodes_expanded']:<15}{h2['nodes_expanded']:<15}")
    write_result_line(outfile, f"{'Runtime(ms)':<20}{round(h1['runtime'], 3):<15}{round(h2['runtime'], 3):<15}")
    write_result_line(outfile, f"{'Path Cost':<20}{h1['cost']:<15}{h2['cost']:<15}")
    if h1["cost"] < h2["cost"]:
        write_result_line(outfile, "h1 produced a lower cost path.")
    elif h2["cost"] < h1["cost"]:
        write_result_line(outfile, "h2 produced a lower cost path.")
    else:
        write_result_line(outfile, "Both heuristics produced the same cost.")


def compare_algorithms(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "GBFS vs A* COMPARISON")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, f"{'Algorithm':<15}{'Nodes':<15}{'Runtime(ms)':<15}{'Cost':<15}")
    for algo, data in results.items():
        write_result_line(outfile, f"{algo:<15}{data['nodes_expanded']:<15}{round(data['runtime'], 3):<15}{data['cost']:<15}")

    write_result_line(outfile, "")
    write_result_line(outfile, "COMPLETENESS ANALYSIS")
    write_result_line(outfile, "A*   : Complete on finite graph with admissible/consistent heuristic assumptions")
    write_result_line(outfile, "GBFS : Not guaranteed complete in general")

    fastest = min(results.items(), key=lambda x: x[1]["runtime"])
    cheapest = min(results.items(), key=lambda x: x[1]["cost"])
    write_result_line(outfile, f"FASTEST ALGORITHM: {fastest[0]} at {fastest[1]['runtime']:.3f} ms")
    write_result_line(outfile, f"LOWEST COST PATH : {cheapest[0]} with cost {cheapest[1]['cost']}")


def print_text_chart(outfile, title: str, history: List[float]):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 70)
    write_result_line(outfile, f" VISUAL TREND RESTRUCTURING: {title}")
    write_result_line(outfile, "=" * 70)
    if not history:
        write_result_line(outfile, "No data.")
        return
    max_val = max(history) if max(history) > 0 else 1
    max_bars = 35
    write_result_line(outfile, f"{'Step / Index':<12} | {'Heuristic':<12} | Relative Visual Scale")
    write_result_line(outfile, "-" * 70)
    for idx, val in enumerate(history):
        bar_length = int((val / max_val) * max_bars)
        write_result_line(outfile, f"Index {idx:<6} | {val:<12.2f} | {'█' * bar_length}")
    write_result_line(outfile, "=" * 70)


def explain_peas(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "PEAS ANALYSIS")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "Performance: Reach the goal quickly while avoiding no-fly zones and minimizing hazard exposure.")
    write_result_line(outfile, "Environment: 8x8 grid with passable cells, weather hazards, and no-fly zones.")
    write_result_line(outfile, "Actuators: Move North, East, South, West.")
    write_result_line(outfile, "Sensors: Current node, neighboring cell types, frontier, explored set, heuristic values.")


def describe_traps(outfile):
    write_result_line(outfile, "")
    write_result_line(outfile, "=" * 100)
    write_result_line(outfile, "TRAP / INEFFICIENT REGION ANALYSIS")
    write_result_line(outfile, "=" * 100)
    for key, traps in trap_results.items():
        write_result_line(outfile, f"{key}")
        write_result_line(outfile, trap_summary(traps))
        write_result_line(outfile, "")


def write_output_file_header(outfile, input_path: Path, start: Tuple[int, int], goal: Tuple[int, int]):
    write_result_line(outfile, "DEFENCE DRONE SEARCH REPORT")
    write_result_line(outfile, f"Input file : {input_path}")
    write_result_line(outfile, f"Start      : {start}")
    write_result_line(outfile, f"Goal       : {goal}")


def main():
    parser = argparse.ArgumentParser(description="Defence Drone GBFS/A* solution")
    parser.add_argument("input_file", nargs="?", help="Input file path")
    parser.add_argument("output_file", nargs="?", help="Output file path")
    args = parser.parse_args()

    input_path = Path(args.input_file) if args.input_file else None
    if input_path is None:
        for candidate in DEFAULT_INPUT_CANDIDATES:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                input_path = candidate_path
                break
    if input_path is None:
        input_path = Path(DEFAULT_INPUT_CANDIDATES[0])

    output_path = Path(args.output_file) if args.output_file else Path("outputPSXX.txt")

    grid, start, goal = parse_input_file(input_path)
    start_node = grid[start[0]][start[1]]
    end_node = grid[goal[0]][goal[1]]

    with output_path.open("w", encoding="utf-8") as outfile:
        write_output_file_header(outfile, input_path, start, goal)
        write_result_line(outfile, "")
        write_result_line(outfile, "Initial Grid")
        show_grid(grid)

        write_result_line(outfile, "GBFS USING H2")
        run_gbfs(grid, start_node, end_node, HeuristicType.BOUNDING_BOX_RISK_WEIGHTED, outfile, "GBFS")

        write_result_line(outfile, "GBFS USING H1")
        run_gbfs(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE, outfile, "GBFS")

        write_result_line(outfile, "A* USING H1")
        run_astar(grid, start_node, end_node, HeuristicType.EUCLIDEAN_DISTANCE, outfile)

        write_result_line(outfile, "A* USING H2")
        run_astar(grid, start_node, end_node, HeuristicType.BOUNDING_BOX_RISK_WEIGHTED, outfile)

        print_complexity_analysis(outfile)
        explain_peas(outfile)
        describe_traps(outfile)
        compare_heuristics_gbfs(outfile)
        compare_heuristics_astar(outfile)
        compare_algorithms(outfile)

        if "GBFS-h1" in results:
            print_text_chart(outfile, "Heuristic Values to Reach Target along Path (h1)", results["GBFS-h1"].get("path_history", []))
            print_text_chart(outfile, "Heuristic Values vs Chronological Expansions (h1)", results["GBFS-h1"].get("expansion_history", []))
        if "GBFS-h2" in results:
            print_text_chart(outfile, "Heuristic Values to Reach Target along Path (h2)", results["GBFS-h2"].get("path_history", []))
            print_text_chart(outfile, "Heuristic Values vs Chronological Expansions (h2)", results["GBFS-h2"].get("expansion_history", []))

        write_result_line(outfile, "")
        write_result_line(outfile, "Conclusion:")
        write_result_line(outfile, "GBFS is fast but not optimal in general; A* is the safer choice for guaranteed shortest-path behavior under the chosen heuristic assumptions.")

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
