import gymnasium as gym
import numpy as np
from tabulate import tabulate

from typing import Tuple, List


class TicTacToeEnv(gym.Env):
    def __init__(self, opponent_strategies, initial_turn=-1) -> None:
        self.n_actions = 9
        self.n_states = 8953  # 8953 = 3**9 possible combinations (not all feasable)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.colors = [1, 2]
        self.fields_per_side = 3
        self.opponent_strategies = opponent_strategies
        if initial_turn != -1:
            self.initial_turn = initial_turn
        else:
            self.initial_turn = np.random.choice([1, 2])
        self.reset()

    def reset(self) -> Tuple[np.ndarray, dict]:
        
        self.state: np.ndarray = np.zeros(
            (self.fields_per_side, self.fields_per_side), dtype=int
        )

        if self.initial_turn == 2:
            opponent_strategy = np.random.choice(self.opponent_strategies)
            action = opponent_strategy(self.state.flatten())
            (row, col) = self.decode_action(action)
            self.state[row, col] = 2


        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten().copy(), self.info
    


    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:

        if not self.action_space.contains(action):
            raise ValueError(f"action '{action}' is not in action_space")

        (row, col) = self.decode_action(action)

        if self.state[row, col] != 0:
            reward = -100
            done = True
            res = "illegal"
            return self.state.flatten().copy(), reward, done, res, self.info

        self.state[row, col] = 1
        self.info["players"][1]["actions"].append(action)

        win = self._is_winner(1)
        draw = self._is_full()
        done = win or draw

        res = None
        reward = 0
        
        if done:
            if win:
                res = "win"
                reward = 10
            elif draw:
                res = "draw"
                reward = 3
    
            return self.state.flatten().copy(), reward, done, res, self.info


        opponent_strategy = np.random.choice(self.opponent_strategies)
        action = opponent_strategy(self.state.flatten())
        (row, col) = self.decode_action(action)
        self.state[row, col] = 2
        self.info["players"][2]["actions"].append(action)

        lose = self._is_winner(2)
        draw = self._is_full()
        done = lose or draw

        if lose:
            res = "lose"
            reward = -10
        elif draw:
            res = "draw"
            reward = 3

        return self.state.flatten().copy(), reward, done, res, self.info

    def _is_winner(self, color: int) -> bool:
        done = False
        bool_matrix = self.state == color

        for ii in range(3):
            if (
                np.all(bool_matrix[:, ii])
                or np.all(bool_matrix[ii, :])
            ):
                done = True
                break

        if np.all([bool_matrix[i, i] for i in range(3)]) or np.all([bool_matrix[i, 2-i] for i in range(3)]):
            done = True
        
        return done
    
    def _is_full(self):
        return np.all(self.state != 0)

    def decode_action(self, action: int) -> List[int]:
        col = action % 3
        row = action // 3
        assert 0 <= col < 3
        return [row, col]

    def render(self) -> None:
        board = np.zeros((3, 3), dtype=str)
        for ii in range(3):
            for jj in range(3):
                if self.state[ii, jj] == 0:
                    board[ii, jj] = "-"
                elif self.state[ii, jj] == 1:
                    board[ii, jj] = "X"
                elif self.state[ii, jj] == 2:
                    board[ii, jj] = "O"

        board = tabulate(board, tablefmt="fancy_grid")
        return board
