class AlphaBetaPruning:
    def __init__(self, depth, game_state, player):
        self.depth = depth
        self.game_state = game_state
        self.player = player  # 'X' for maximizer, 'O' for minimizer

    def is_terminal(self, state):
        # Check if there's a winner or if it's a draw
        winning_combinations = [
            [state[0], state[1], state[2]],
            [state[3], state[4], state[5]],
            [state[6], state[7], state[8]],
            [state[0], state[3], state[6]],
            [state[1], state[4], state[7]],
            [state[2], state[5], state[8]],
            [state[0], state[4], state[8]],
            [state[2], state[4], state[6]],
        ]
        if ['X', 'X', 'X'] in winning_combinations:
            return True, 'X'  # Maximizer wins
        if ['O', 'O', 'O'] in winning_combinations:
            return True, 'O'  # Minimizer wins
        if not any(s == '' for s in state):
            return True, 'Draw'  # Draw
        return False, None  # Game continues

    def utility(self, state):
        # Returns +1 if maximizer wins, -1 if minimizer wins, 0 for a draw
        is_terminal, winner = self.is_terminal(state)
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        return 0  # Draw or ongoing game

    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        is_terminal, _ = self.is_terminal(state)
        if depth == 0 or is_terminal:
            return self.utility(state)

        if maximizing_player:
            max_eval = -float('inf')
            for i in range(len(state)):
                if state[i] == '':  # Check for empty spot
                    state[i] = 'X'  # Maximizer move
                    eval = self.alphabeta(state, depth - 1, alpha, beta, False)
                    state[i] = ''  # Undo move
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(len(state)):
                if state[i] == '':
                    state[i] = 'O'  # Minimizer move
                    eval = self.alphabeta(state, depth - 1, alpha, beta, True)
                    state[i] = ''  # Undo move
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
            return min_eval

    def best_move(self, state):
        best_value = -float('inf')
        best_move = None
        for i in range(len(state)):
            if state[i] == '':  # Check for empty spot
                state[i] = 'X'  # Maximizer move
                move_value = self.alphabeta(state, self.depth, -float('inf'), float('inf'), False)
                state[i] = ''  # Undo move
                if move_value > best_value:
                    best_value = move_value
                    best_move = i
        return best_move


# Example usage:
if __name__ == "__main__":
    game_state = ['', '', '', '', '', '', '', '', '']  # Empty Tic-Tac-Toe board
    game = AlphaBetaPruning(depth=9, game_state=game_state, player='X')

    # Find the best move for player 'X' (Maximizer)
    move = game.best_move(game_state)
    print(f"Best move for 'X': {move}")
