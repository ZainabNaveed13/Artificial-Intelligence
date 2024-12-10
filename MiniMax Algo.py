class Minimax:
    def __init__(self, game_state):
        self.game_state = game_state

    def is_terminal(self, state):
        return self.check_win(state, 'X') or self.check_win(state, 'O') or self.check_draw(state)

    def utility(self, state):
        if self.check_win(state, 'X'):
            return 1
        elif self.check_win(state, 'O'):
            return -1
        else:
            return 0

    def minimax(self, state, depth, maximizing_player):
        if self.is_terminal(state) or depth == 0:
            return self.utility(state)

        if maximizing_player:
            max_eval = float('-inf')
            for child in self.get_available_moves(state):
                eval = self.minimax(child, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for child in self.get_available_moves(state):
                eval = self.minimax(child, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def best_move(self, state):
        best_score = float('-inf')
        best_move = None
        for move in self.get_available_moves(state):
            move_score = self.minimax(move, depth=9, maximizing_player=True)
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_move

    def get_available_moves(self, state):
        return [next_state for next_state in self.generate_possible_states(state)]

    def check_win(self, state, player):
        win_conditions = [
            [state[0], state[1], state[2]],
            [state[3], state[4], state[5]],
            [state[6], state[7], state[8]],
            [state[0], state[3], state[6]],
            [state[1], state[4], state[7]],
            [state[2], state[5], state[8]],
            [state[0], state[4], state[8]],
            [state[2], state[4], state[6]]
        ]
        return [player, player, player] in win_conditions

    def check_draw(self, state):
        return all(cell != '' for cell in state)

    def generate_possible_states(self, state):
        possible_states = []
        for i in range(len(state)):
            if state[i] == '':
                new_state = state[:]
                new_state[i] = 'X' if state.count('X') <= state.count('O') else 'O'
                possible_states.append(new_state)
        return possible_states

    def print_board(self, state):
        for i in range(0, 9, 3):
            row = state[i:i+3]
            row_str = [' ' if cell == '' else cell for cell in row]
            print(f"{row_str[0]} | {row_str[1]} | {row_str[2]}")
            if i < 6:
                print("---------")
        print()

test_game_states = [
    ['', '', '', '', '', '', '', '', ''],
    ['X', '', 'X', '', 'O', 'O', '', '', ''],
    ['X', 'X', 'O', 'O', 'O', 'X', 'X', '', ''],
    ['X', 'X', 'X', 'O', 'O', '', '', '', ''],
    ['O', 'O', 'O', 'X', 'X', '', '', '', '']
]

for state in test_game_states:
    minimax_solver = Minimax(state)
    print("Current Board State:")
    minimax_solver.print_board(state)

    if not minimax_solver.is_terminal(state):
        best_move = minimax_solver.best_move(state)
        print("Best move found:")
        minimax_solver.print_board(best_move)
    else:
        print("Game already finished for this state.\n")
