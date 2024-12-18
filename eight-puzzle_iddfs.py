def is_goal(puzzle):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    return puzzle == goal

def move(puzzle, direction):
    new_puzzle = [row[:] for row in puzzle]

    row, col = [(i, row.index(0)) for i, row in enumerate(new_puzzle) if 0 in row][0]

    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    new_row, new_col = row + moves[direction][0], col + moves[direction][1]

    if 0 <= new_row < 3 and 0 <= new_col < 3:
        new_puzzle[row][col], new_puzzle[new_row][new_col] = new_puzzle[new_row][new_col], new_puzzle[row][col]
        return new_puzzle
    return None

def dfs(puzzle, depth, path):
    if is_goal(puzzle):
        return True, path + [puzzle]
    if depth == 0:
        return False, []

    for direction in ['up', 'down', 'left', 'right']:
        new_puzzle = move(puzzle, direction)
        if new_puzzle:
            found, new_path = dfs(new_puzzle, depth - 1, path + [puzzle])
            if found:
                return True, new_path
    return False, []

def iddfs(puzzle, max_depth):
    for depth in range(max_depth + 1):
        print(f"Searching at Depth: {depth}")
        found, path = dfs(puzzle, depth, [])
        if found:
            print("Goal state reached!")
            for state in path[::-1]:
                for row in state:
                    print(row)
                print()
            return True
    return False

def generate_puzzle():
    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    return initial_state, goal_state

if __name__ == '__main__':
    initial_state, goal_state = generate_puzzle()

    print("Initial Puzzle:")
    for row in initial_state:
        print(row)
    print()

    max_depth = 10

    if not iddfs(initial_state, max_depth):
        print("Solution not found within the depth limit.")
