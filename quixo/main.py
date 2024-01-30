from typing import Tuple
import random
import numpy as np
from copy import deepcopy
from tabulate import tabulate

from game import Game, Move, Player

N = 5

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> Tuple[Tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyRandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game):
        state = game.get_board()
        player_id = game.get_current_player()

        state = state.reshape((5, 5))
        available_actions = self.get_available_actions(state,player_id)
        return random.choice(available_actions)
    
    def get_available_actions(self, state, player_id):
        available_actions = []

        for i in range(5):
            if state[4][i] == player_id or state[4][i] == -1:
                available_actions.append(((4, i), Move.TOP))
        
        for i in range(5):
            if state[0][i] == player_id or state[0][i] == -1:
                available_actions.append(((0, i), Move.BOTTOM))
        
        for i in range(5):
            if state[i][4] == player_id or state[i][4] == -1:
                available_actions.append(((i, 4), Move.LEFT))

        for i in range(5):
            if state[i][0] == player_id or state[i][0] == -1:
                available_actions.append(((i, 0), Move.RIGHT))
        
        return available_actions







class QuixoGame:
    def __init__(self, state=None) -> None:
        if state is None:
            self.state = np.ones((N, N), dtype=np.uint8) * -1
        else:
            self.state = deepcopy(state.reshape((N, N)))

    def get_current_state(self):
        return deepcopy(self.state.flatten())

    def print_state(self):
        board = np.zeros((N, N), dtype=str)
        for ii in range(N):
            for jj in range(N):
                if self.state[ii, jj] == -1:
                    board[ii, jj] = "-"
                elif self.state[ii, jj] == 0:
                    board[ii, jj] = "X"
                elif self.state[ii, jj] == 1:
                    board[ii, jj] = "O"

        board = tabulate(board, tablefmt="fancy_grid")
        print(board)

    def step(self, action, player_id):
        from_pos, move = action

        if not self._acceptable_move(from_pos, move, player_id):
            done = True
            res = "illegal"
            return deepcopy(self.state.flatten()), done, res

        self._move(from_pos, move, player_id)

        lose = self._is_winner(1-player_id)
        win = self._is_winner(player_id)

        done = win or lose
        res = None

        if lose:
            res = "lose"
        elif win:
            res = "win"

        return deepcopy(self.state.flatten()), done, res


    def _acceptable_move(self, from_pos: Tuple[int, int], slide: Move, player_id: int) -> bool:

        SIDES = [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1)]

        # Check if in border and the cube is -1 or player id
        acceptable = (
            # check if it is in the first row
            (from_pos[0] == 0 and 0 <= from_pos[1] < N)
            # check if it is in the last row
            or (from_pos[0] == N-1 and 0 <= from_pos[1] < N)
            # check if it is in the first column
            or (from_pos[1] == 0 and 0 <= from_pos[0] < N)
            # check if it is in the last column
            or (from_pos[1] == N-1 and 0 <= from_pos[0] < N)
            # and check if the piece can be moved by the current player
        ) and (self.state[from_pos] < 0 or self.state[from_pos] == player_id)

        if not acceptable:
            return acceptable


        # Check if the slide is possible given the position

        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top = from_pos[0] == 0 and slide == Move.BOTTOM
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom = from_pos[0] == N-1 and slide == Move.TOP
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left = from_pos[1] == 0 and slide == Move.RIGHT
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right = from_pos[1] == N-1 and slide == Move.LEFT

        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top = from_pos == (0, 0) and (slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left = from_pos == (N-1, 0) and (slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right = from_pos == (0, N-1) and (slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom = from_pos == (N-1, N-1) and (slide == Move.TOP or slide == Move.LEFT)

        acceptable = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right

        return acceptable


    def _move(self, from_pos: Tuple[int, int], slide: Move, player_id: int):
        '''Perform a move'''
        self.last_state = deepcopy(self.state)
        self._take(from_pos, player_id)
        self._slide(from_pos, slide)


    def _take(self, from_pos: Tuple[int, int], player_id: int):
        '''Take piece'''
        self.state[from_pos] = player_id


    def _slide(self, from_pos: Tuple[int, int], slide: Move):
        '''Slide the other pieces'''
        # take the piece
        piece = self.state[from_pos]
        # if the player wants to slide it to the left
        if slide == Move.LEFT:
            # for each column starting from the column of the piece and moving to the left
            for i in range(from_pos[1], 0, -1):
                # copy the value contained in the same row and the previous column
                self.state[(from_pos[0], i)] = self.state[(
                    from_pos[0], i - 1)]
            # move the piece to the left
            self.state[(from_pos[0], 0)] = piece
        # if the player wants to slide it to the right
        elif slide == Move.RIGHT:
            # for each column starting from the column of the piece and moving to the right
            for i in range(from_pos[1], self.state.shape[1] - 1, 1):
                # copy the value contained in the same row and the following column
                self.state[(from_pos[0], i)] = self.state[(
                    from_pos[0], i + 1)]
            # move the piece to the right
            self.state[(from_pos[0], self.state.shape[1] - 1)] = piece
        # if the player wants to slide it upward
        elif slide == Move.TOP:
            # for each row starting from the row of the piece and going upward
            for i in range(from_pos[0], 0, -1):
                # copy the value contained in the same column and the previous row
                self.state[(i, from_pos[1])] = self.state[(
                    i - 1, from_pos[1])]
            # move the piece up
            self.state[(0, from_pos[1])] = piece
        # if the player wants to slide it downward
        elif slide == Move.BOTTOM:
            # for each row starting from the row of the piece and going downward
            for i in range(from_pos[0], self.state.shape[0] - 1, 1):
                # copy the value contained in the same column and the following row
                self.state[(i, from_pos[1])] = self.state[(
                    i + 1, from_pos[1])]
            # move the piece down
            self.state[(self.state.shape[0] - 1, from_pos[1])] = piece

    def _is_winner(self, player_id: int) -> bool:
        # for each row
        for x in range(self.state.shape[0]):
            if all(self.state[x, :] == player_id):
                return True
        # for each column
        for y in range(self.state.shape[1]):
            if all(self.state[:, y] == player_id):
                return True
        # if a player has completed the principal diagonal
        if all(np.array([self.state[x, x] for x in range(self.state.shape[0])]) == player_id):
            return True
        # if a player has completed the secondary diagonal
        if all(np.array([self.state[x, -(x + 1)] for x in range(self.state.shape[0])]) == player_id):
            return True

        return False
    


class MiniMaxNode:
    def __init__(self, state, node_player_id, action=None) -> None:
        self.state = deepcopy(state)
        self.node_player_id = node_player_id
        self.action = action
        self.children = []

    def has_children(self):
        return len(self.children)>0

    def get_children(self):
        return self.children

    def get_action(self):
        return self.action



    def expand(self, depth_left):
        if depth_left == 0:
            return

        for action in self._get_available_actions(self.state):
            game = QuixoGame(self.state.flatten())
            next_state, done, _ = game.step(action, self.node_player_id)
            new_node = MiniMaxNode(next_state.reshape((N, N)), 1-self.node_player_id, action)
            if not done:
                new_node.expand(depth_left-1)
            self.children.append(new_node)

    def evaluate(self, player_id):
        player_count = []
        opponent_count = []

        for i in range(self.state.shape[0]):
            row = self.state[i]
            player_count.append(np.count_nonzero(row==player_id))
            opponent_count.append(np.count_nonzero(row==1-player_id))

        for j in range(self.state.shape[1]):
            col = self.state[:, j].transpose()
            player_count.append(np.count_nonzero(col==player_id))
            opponent_count.append(np.count_nonzero(col==1-player_id))

        player_count.append(np.count_nonzero(np.array([self.state[i, i] for i in range(self.state.shape[0])]) == player_id))
        player_count.append(np.count_nonzero(np.array([self.state[i, -(i + 1)] for i in range(self.state.shape[0])]) == player_id))

        opponent_count.append(np.count_nonzero(np.array([self.state[i, i] for i in range(self.state.shape[0])]) == 1-player_id))
        opponent_count.append(np.count_nonzero(np.array([self.state[i, -(i + 1)] for i in range(self.state.shape[0])]) == 1-player_id))

        scoremax = N ** max(player_count)
        scoremin = N ** max(opponent_count)

        return scoremax - scoremin

    def _get_available_actions(self, state):
        available_actions = []
        for i in range(N):
            if state[N-1][i] == self.node_player_id or state[N-1][i] == -1:
                available_actions.append(((N-1, i), Move.TOP))

        for i in range(N):
            if state[0][i] == self.node_player_id or state[0][i] == -1:
                available_actions.append(((0, i), Move.BOTTOM))

        for i in range(N):
            if state[i][N-1] == self.node_player_id or state[i][N-1] == -1:
                available_actions.append(((i, N-1), Move.LEFT))

        for i in range(N):
            if state[i][0] == self.node_player_id or state[i][0] == -1:
                available_actions.append(((i, 0), Move.RIGHT))

        return available_actions

class MiniMaxTree:
    def __init__(self, state, player_id, depth) -> None:
        self.player_id = player_id
        self.root = MiniMaxNode(state, self.player_id)
        self.depth = depth
        self.root.expand(self.depth)

    def max_value(self, node, alpha, beta):
        if not node.has_children():
            return node.evaluate(self.player_id)

        val = -float("inf")

        for child in node.get_children():
            val = max(val, self.min_value(child, alpha, beta))
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_value(self, node, alpha, beta):
        if not node.has_children():
            return node.evaluate(self.player_id)

        val = float("inf")

        for child in node.get_children():
            val = min(val, self.max_value(child, alpha, beta))
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val

    def alpha_beta_search(self):
        alpha = -float("inf")
        beta = float("inf")

        best_node = None
        for child in self.root.get_children():
            val = self.min_value(child, alpha, beta)
            if val > alpha:
                alpha = val
                best_node = child

        return best_node


class MiniMaxPlayer(Player):
    def __init__(self, depth) -> None:
        super().__init__()
        self. depth = depth
    
    def make_move(self, game):
        state = game.get_board().reshape((N, N))
        player_id = game.get_current_player()
        tree = MiniMaxTree(state, player_id, self.depth)
        best_node = tree.alpha_beta_search()
        action = best_node.get_action()
        action = ((action[0][1], action[0][0]), action[1])
        return action


if __name__ == '__main__':
    g = Game()
    g.print()
    player1 = MiniMaxPlayer(depth=2)
    player2 = MiniMaxPlayer(depth=3)
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")
