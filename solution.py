from enum import Enum
from math import sqrt
from typing import List

class NodeType(Enum):
    PASSABLE_AIRSPACE = (".",1)
    WEATHER_HAZARD = ("W",4)
    NO_FLY_ZONE = ("N",8)
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


class HeuristicType(Enum):
    EUCLIDEAN_DISTANCE = "h1"
    BOUNDING_BOX_RISK_WEIGHTED = "h2"

def euclidean_distance(current_state: Node, goal_state: Node):
    return sqrt(((goal_state.x - current_state.x)*(goal_state.x - current_state.x) + (goal_state.y - current_state.y)*(goal_state.y - current_state.y)))

def manhatten_distance(current_state: Node, goal_state: Node):
    return abs(goal_state.x - current_state.x) + abs(goal_state.y - current_state.y)


def bounding_box_risk_weighted(grid: List[List[Node]], current_state: Node, goal_state: Node):
    manhatten = manhatten_distance(current_state, goal_state)
    k = abs(goal_state.x - current_state.x+1) * abs(goal_state.y - current_state.y+1)
    x_min = min(current_state.x, goal_state.x)
    x_max = max(current_state.x, goal_state.x)
    y_min = min(current_state.y, goal_state.y)
    y_max = max(current_state.y, goal_state.y)
    partial_score = 0
    for row in range(x_min, x_max+1):
        for col in range(y_min, y_max+1):
            partial_score+=grid[row][col].score
    return manhatten * (partial_score/k)


def heuristic(grid, current_state: Node, goal_state: Node, type: HeuristicType):
    if type == HeuristicType.EUCLIDEAN_DISTANCE:
        return euclidean_distance(current_state, goal_state)
    elif type == HeuristicType.BOUNDING_BOX_RISK_WEIGHTED:
        return bounding_box_risk_weighted(grid, current_state, goal_state)
    else:
        pass

def build_grid(size: int, start_x, start_y, end_x, end_y):
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


def show_grid(grid):
    for row in grid:
        for col in row:
            print(col.x, col.y, col.type, col.score)

def main():
    GRID_SIZE = 8
    grid = build_grid(GRID_SIZE, 0,0,6,7)
    show_grid(grid)
    print(heuristic(grid, grid[3][3], grid[6][7], HeuristicType.BOUNDING_BOX_RISK_WEIGHTED))

if __name__ == "__main__":
    main()