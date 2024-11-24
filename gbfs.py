class Node:
    def __init__(self, state, parent, move, h_cost):
        self.state = state
        self.parent = parent
        self.move = move
        self.h_cost = h_cost

    def generate_children(self, goal_state):
        children = []
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        empty_tile = [(i, row.index(0)) for i, row in enumerate(self.state) if 0 in row][0]

        for move, (dx, dy) in moves.items():
            new_x, new_y = empty_tile[0] + dx, empty_tile[1] + dy
            if 0 <= new_x < len(self.state) and 0 <= new_y < len(self.state[0]):
                new_state = [row[:] for row in self.state]
                new_state[empty_tile[0]][empty_tile[1]], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[empty_tile[0]][empty_tile[1]]
                h_cost = self.calculate_heuristic(new_state, goal_state)
                children.append(Node(new_state, self, move, h_cost))

        return children

    def calculate_heuristic(self, current_state, goal_state):
        manhattan_distance = 0
        for i in range(len(current_state)):
            for j in range(len(current_state[i])):
                if current_state[i][j] != 0:
                    x_goal, y_goal = [(x, row.index(current_state[i][j])) for x, row in enumerate(goal_state) if current_state[i][j] in row][0]
                    manhattan_distance += abs(i - x_goal) + abs(j - y_goal)
        return manhattan_distance

class GreedyBestFirstSearch:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.open_list = []
        self.closed_list = []

    def solve(self):
        start_node = Node(self.start_state, None, None, 0)
        start_node.h_cost = start_node.calculate_heuristic(self.start_state, self.goal_state)
        self.open_list.append(start_node)

        while self.open_list:
            self.open_list.sort(key=lambda node: node.h_cost)
            current_node = self.open_list.pop(0)

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            self.closed_list.append(current_node)

            for child in current_node.generate_children(self.goal_state):
                if child not in self.closed_list and child not in self.open_list:
                    self.open_list.append(child)

        return None

    def trace_solution(self, node):
        path = []
        states = []
        while node:
            if node.move:
                path.append(node.move)
            states.append(node.state)
            node = node.parent
        return path[::-1], states[::-1]

    def print_solution_states(self, states):
        print("Solution States:")
        for i, state in enumerate(states):
            print(f"State {i}:")
            for row in state:
                print(row)
            print()

start_state = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

gbfs_solver = GreedyBestFirstSearch(start_state, goal_state)
solution_path, solution_states = gbfs_solver.solve()

print("Solution Path:", solution_path)
gbfs_solver.print_solution_states(solution_states)
