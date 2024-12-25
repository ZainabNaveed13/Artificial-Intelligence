import heapq

class PuzzleNode:
    def __init__(self, state, parent, g_cost, h_cost):
        self.state = state
        self.parent = parent
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def generate_children(self):
        children = []
        blank_pos = self.state.index(0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        x, y = divmod(blank_pos, 3)

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_blank_pos = new_x * 3 + new_y
                new_state = self.state[:]
                new_state[blank_pos], new_state[new_blank_pos] = new_state[new_blank_pos], new_state[blank_pos]
                child_node = PuzzleNode(new_state, self, self.g_cost + 1, 0)
                children.append(child_node)
        return children

    def calculate_heuristic(self, goal_state):
        return sum([1 if self.state[i] != goal_state[i] and self.state[i] != 0 else 0 for i in range(9)])

class AStarSolver:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

    def solve(self):
        open_list = []
        closed_list = set()
        start_node = PuzzleNode(self.start_state, None, 0, 0)
        start_node.h_cost = start_node.calculate_heuristic(self.goal_state)
        heapq.heappush(open_list, (start_node.f_cost, start_node))

        print("Initial State:")
        self.print_state(self.start_state)
        print("Goal State:")
        self.print_state(self.goal_state)

        while open_list:
            _, current_node = heapq.heappop(open_list)
            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            closed_list.add(tuple(current_node.state))
            for child in current_node.generate_children():
                if tuple(child.state) in closed_list:
                    continue
                child.h_cost = child.calculate_heuristic(self.goal_state)
                heapq.heappush(open_list, (child.f_cost, child))
        return None

    def trace_solution(self, node):
        solution = []
        while node:
            solution.append(node.state)
            node = node.parent
        solution = solution[::-1]
        print("Path to goal state:")
        for step in solution:
            self.print_state(step)
        print("Found path")
        return solution

    def is_solvable(self, state):
        inv_count = 0
        for i in range(9):
            for j in range(i + 1, 9):
                if state[i] and state[j] and state[i] > state[j]:
                    inv_count += 1
        return inv_count % 2 == 0

    def print_state(self, state):
        for i in range(0, 9, 3):
            print(state[i:i+3])
        print()

#test case 1
start_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
solver = AStarSolver(start_state, goal_state)

if solver.is_solvable(start_state):
    solution = solver.solve()
    if not solution:
        print("No path found")
else:
    print("The puzzle is not solvable")

print("-------------------")

#test case 2
start_state = [1, 2, 3, 4, 5, 6, 8, 7, 0]
solver = AStarSolver(start_state, goal_state)

print("Initial State:")
solver.print_state(start_state)
print("Goal State:")
solver.print_state(goal_state)

if solver.is_solvable(start_state):
    solution = solver.solve()
    if not solution:
        print("No path found")
else:
    print("The puzzle is not solvable")
