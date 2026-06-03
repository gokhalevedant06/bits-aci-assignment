from enum import Enum
import itertools
from math import sqrt
from typing import List
import queue

GRID_SIZE = 8

class NodeType(Enum):
    PASSABLE_AIRSPACE = (".",1)
    WEATHER_HAZARD = ("W",4)
    NO_FLY_ZONE = ("N",8)
    START = ("S", 2)
    END = ("E", 2)

class MovementModel(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
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
    k = (abs(goal_state.x - current_state.x)+1) * (abs(goal_state.y - current_state.y)+1)
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

# def show_grid(grid, path= []):
#     path_set = set(path)
#     for row in grid:
#         for node in row:
#             if node in path_set:
#                 print("*", end="   ")
#             else:
#                 print(node.type, end="   ")
#         print()

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

# get valid neighbours of a node in the grid
# we can move as per MovementModel in 4 directions, so we will have at max 4 neighbours for a node
# we need to get the correct neighbours as per the priority order of movement model, and also check for the boundaries of the grid
def get_neighbours(grid, node: Node):
    """
    Return orthogonal neighbours in priority order: NORTH, EAST, SOUTH, WEST.

    Note: grid is indexed as grid[x][y] where x is row (vertical axis) and
    y is column (horizontal axis). NORTH decreases x, SOUTH increases x,
    EAST increases y, WEST decreases y.
    """
    neighbours = []
    size = len(grid)

    #NORTH
    if node.x - 1 >= 0:
        neighbours.append(grid[node.x - 1][node.y])

    #EAST
    if node.y + 1 < size:
        neighbours.append(grid[node.x][node.y + 1])

    #SOUTH
    if node.x + 1 < size:
        neighbours.append(grid[node.x + 1][node.y])

    #WEST
    if node.y - 1 >= 0:
        neighbours.append(grid[node.x][node.y - 1])

    return neighbours

def astar(grid, start_node: Node, end_node: Node, heuristic_fx: HeuristicType = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED):
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
    while not pq.empty():
        f_val, cnt, current_node = pq.get()
        current_cost = g_score.get(current_node)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end_node:
            show_grid(grid, visited)
            path = []
            while(current_node):
                path.append(current_node)
                current_node = parent.get(current_node)
            
            path.reverse()

            print(" -> ".join(f"({node.x},{node.y})" for node in path))

            print("DESTINATION REACHED WITH COST", g_score.get(end_node))
            break
    
        for node in get_neighbours(grid, current_node):
            if node in visited:
                continue
            new_cost = current_cost + node.score
            if node not in g_score or new_cost < g_score.get(node):   
                parent[node] = current_node
                g_score[node] = new_cost
                h_val = heuristic(grid, grid[node.x][node.y], grid[end_node.x][end_node.y], heuristic_fx)
                f_val = new_cost + h_val
                pq.put((f_val, next(counter), node))


def gbfs(grid, start_node: Node, end_node: Node, heuristic_fx: HeuristicType = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED):
    pq = queue.PriorityQueue()
    counter = itertools.count()
    visited = set()
    #explored = set()
    parent = {}
    parent[start_node] = None
    #explored.add(start_node)
    h_val = heuristic(grid, grid[start_node.x][start_node.y], grid[end_node.x][end_node.y], heuristic_fx)
    pq.put((h_val, next(counter), start_node))
    while not pq.empty():
        h_val, cnt, current_node = pq.get()
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end_node:
            show_grid(grid, visited)
            path = []
            while(current_node):
                path.append(current_node)
                current_node = parent.get(current_node)
            
            path.reverse()

            print(" -> ".join(f"({node.x},{node.y})" for node in path))

            print("DESTINATION REACHED WITH COST")
            break
        neighbours = get_neighbours(grid, current_node)
        for node in neighbours:
            # commented below as we want to add all the neighbours node, and not fix the path for gbfs.
            # as it is a greedy algorithm and we want to explore all the nodes in the priority order of heuristic value, and not get stuck in a local minima
            # if node not in explored:
            #     explored.add(node)
            if node not in visited:
                # do not override if already there in the parent
                # don't know what would be the situation but I think it is correct
                if node not in parent:
                    parent[node] = current_node
                h_val = heuristic(grid, grid[node.x][node.y], grid[end_node.x][end_node.y], heuristic_fx)
                pq.put((h_val, next(counter), node))
        
    

def main():
    grid = build_grid(GRID_SIZE, 0,0,6,7)
    show_grid(grid)
    # astar(grid, grid[0][0], grid[6][7])
    gbfs(grid, grid[0][0], grid[6][7], heuristic_fx = HeuristicType.BOUNDING_BOX_RISK_WEIGHTED)

if __name__ == "__main__":
    main()