import argparse
from typing import Tuple
from enum import Enum
import numpy as np
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from tabulate import tabulate
import math
from bitarray import bitarray

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Params
N = 5


# Game logic
    
class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


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
    



# Test function
    
def test(agent, opponent_agents, episode_num, max_episode_steps, init_player=None, render=False, verbose=False):
    win_count = 0
    draw_count = 0
    lose_count = 0
    illegal_count = 0

    for ep in range(episode_num):
        game = QuixoGame()
        state = game.get_current_state()
        if init_player is not None:
            current_player = init_player
        else:
            current_player = random.choice([0, 1])

        done = False
        step = 0
        while not done and step < max_episode_steps:
            if current_player == 0:
                action = agent.choose_action(state)
                next_state, done, res = game.step(action, current_player)

            elif current_player == 1:
                opponent_agent = random.choice(opponent_agents)
                action = opponent_agent.choose_action(state)
                next_state, done, res = game.step(action, current_player)

            if render:
                print(f"Player: {current_player}, Action: {action}")
                game.print_state()
                input()

            if not done:
                step += 1
                state = next_state
                current_player = 1 - current_player


        if step >= max_episode_steps:
            res = "draw"
        elif (current_player == 0 and res == "win") or (current_player == 1 and res == "lose"):
            res = "win"
        elif (current_player == 0 and res == "lose") or (current_player == 1 and res == "win"):
            res = "lose"

        if res == "win":
            win_count += 1
        elif res == "draw":
            draw_count += 1
        elif res == "lose":
            lose_count += 1
        elif res == "illegal":
            illegal_count += 1

        if verbose:
            print(f"Episode {ep} - Result: {res.upper()}")

    print(f"Win rate {100*win_count/episode_num:.2f}%")
    print(f"Draw rate {100*draw_count/episode_num:.2f}%")
    print(f"Lose rate {100*lose_count/episode_num:.2f}%")
    print(f"Illegal rate {100*illegal_count/episode_num:.2f}%")



# Agent abstact class

class Agent(ABC):
    def __init__(self, player_id) -> None:
        self.player_id = player_id

    @abstractmethod
    def choose_action(self, state) -> Tuple[Tuple[int, int], Move]:
        pass


# Random agent

class RandomAgent(Agent):
    def __init__(self, player_id) -> None:
        super().__init__(player_id)


    def _get_available_actions(self, state):
        available_actions = []
        for i in range(N):
            if state[N-1][i] == self.player_id or state[N-1][i] == -1:
                available_actions.append(((N-1, i), Move.TOP))

        for i in range(N):
            if state[0][i] == self.player_id or state[0][i] == -1:
                available_actions.append(((0, i), Move.BOTTOM))

        for i in range(N):
            if state[i][N-1] == self.player_id or state[i][N-1] == -1:
                available_actions.append(((i, N-1), Move.LEFT))

        for i in range(N):
            if state[i][0] == self.player_id or state[i][0] == -1:
                available_actions.append(((i, 0), Move.RIGHT))

        return available_actions

    def choose_action(self, state):
        state = state.reshape((N, N))
        available_actions = self._get_available_actions(state)
        return random.choice(available_actions)



# MiniMax agent

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


class MiniMaxAgent(Agent):
    def __init__(self, player_id, depth) -> None:
        super().__init__(player_id)
        self.depth = depth

    def choose_action(self, state):
        state = state.reshape((N, N))
        tree = MiniMaxTree(state, self.player_id, self.depth)
        best_node = tree.alpha_beta_search()
        action = best_node.get_action()
        return action






# Deep Q-Learning agent

class DeepQNetwork(nn.Module):
    def __init__(self, lr, in_features, h1, h2, out_features):
        super().__init__()
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.out(x)
        return actions
    


class DeepQLearningAgent(Agent):
    def __init__(self, player_id, gamma, lr, state_dim, batch_size, actions_dim, mem_size=100000):
        self.player_id = player_id
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(actions_dim)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, state_dim, 256, 256, actions_dim)

        self.state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)

        self.action_memory = np.zeros(mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def store_transition(self, state, action, reward, state_, done):
        state = np.array(state, dtype=np.float32)

        index = self.mem_cntr % self.mem_size
        # self.state_memory[index] = self.reformat_state(state)
        # self.new_state_memory[index] = self.reformat_state(state_)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = self.encode_action(action)
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    # def reformat_state(self, state):
    #     state_player_0 = np.where(state!=0, 0, 1)
    #     state_player_1 = np.where(state!=1, 0, 1)
    #     state_empty = np.where(state!=-1, 0, 1)

    #     if self.player_id == 1:
    #         state_player_0, state_player_1 = state_player_1, state_player_0

    #     return np.concatenate([state_empty, state_player_0, state_player_1], axis=0, dtype=np.float32)

    def decode_action(self, action: int) -> Tuple[Tuple[int, int], Move]:
        decoded_action = None

        i = (action - N*Move.TOP.value)
        if 0 <= i < N:
            decoded_action = ((N-1, i), Move.TOP)

        i = (action - N*Move.BOTTOM.value)
        if 0 <= i < N:
            decoded_action = ((0, i), Move.BOTTOM)

        i = (action - N*Move.LEFT.value)
        if 0 <= i < N:
            decoded_action = ((i, N-1), Move.LEFT)

        i = (action - N*Move.RIGHT.value)
        if 0 <= i < N:
            decoded_action = ((i, 0), Move.RIGHT)

        return decoded_action

    def encode_action(self, action):
        from_pos, move = action
        encoded_action = None

        if move == Move.TOP:
            encoded_action = (from_pos[1] + N*Move.TOP.value)
        if move == Move.BOTTOM:
            encoded_action = (from_pos[1] + N*Move.BOTTOM.value)
        if move == Move.LEFT:
            encoded_action = (from_pos[0] + N*Move.LEFT.value)
        if move == Move.RIGHT:
            encoded_action = (from_pos[0] + N*Move.RIGHT.value)

        return encoded_action


    def choose_action(self, state):
        state = np.array(state, dtype=np.float32)
        available_actions = self.get_available_actions(state.reshape((N, N)))
        decoded_action = None

        with torch.no_grad():
            # state = self.reformat_state(state)
            state = T.tensor(state).to(self.Q_eval.device)
            actions = self.Q_eval(state)

            sorted_actions = T.argsort(actions, descending=True).tolist()

            for action in sorted_actions:
                decoded_action = self.decode_action(action)
                if decoded_action in available_actions:
                    break

        return decoded_action


    def choose_action_train(self, state, epsilon):
        state = np.array(state, dtype=np.float32)
        available_actions = self.get_available_actions(state.reshape((N, N)))
        decoded_action = None
        if np.random.random() > epsilon:
            # state = self.reformat_state(state)
            state = T.tensor(state).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            #action = T.argmax(actions).item()

            sorted_actions = T.argsort(actions, descending=True).tolist()

            for action in sorted_actions:
                decoded_action = self.decode_action(action)
                if decoded_action in available_actions:
                    break

        else:
            while decoded_action not in available_actions:
                action = np.random.choice(self.action_space)
                decoded_action = self.decode_action(action)
        return decoded_action



    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

    def get_available_actions(self, state):
        available_actions = []
        for i in range(N):
            if state[N-1][i] == self.player_id or state[N-1][i] == -1:
                available_actions.append(((N-1, i), Move.TOP))

        for i in range(N):
            if state[0][i] == self.player_id or state[0][i] == -1:
                available_actions.append(((0, i), Move.BOTTOM))

        for i in range(N):
            if state[i][N-1] == self.player_id or state[i][N-1] == -1:
                available_actions.append(((i, N-1), Move.LEFT))

        for i in range(N):
            if state[i][0] == self.player_id or state[i][0] == -1:
                available_actions.append(((i, 0), Move.RIGHT))

        return available_actions


    def save_model(self, filename="model.pt"):
        torch.save(self.Q_eval.state_dict(), filename)

    def load_model(self, filename="model.pt"):
        self.Q_eval.load_state_dict(torch.load(filename))



# Training function

def reward_1(res, step, current_player):
    reward = 0
    if (current_player == 0 and res== "win") or (current_player == 1 and res== "lose"):
        reward = 200 - step
    elif (current_player == 0 and res== "lose") or (current_player == 1 and res== "win"):
        reward = - (200 - step)
    return reward


def evaluate(state, player_id):
    state = state.reshape((N, N))
    player_count = []
    opponent_count = []

    for i in range(N):
        row = state[i]
        player_count.append(np.count_nonzero(row==player_id))
        opponent_count.append(np.count_nonzero(row==1-player_id))

    for j in range(N):
        col = state[:, j].transpose()
        player_count.append(np.count_nonzero(col==player_id))
        opponent_count.append(np.count_nonzero(col==1-player_id))

    player_count.append(np.count_nonzero(np.array([state[i, i] for i in range(state.shape[0])]) == player_id))
    player_count.append(np.count_nonzero(np.array([state[i, -(i + 1)] for i in range(state.shape[0])]) == player_id))

    opponent_count.append(np.count_nonzero(np.array([state[i, i] for i in range(state.shape[0])]) == 1-player_id))
    opponent_count.append(np.count_nonzero(np.array([state[i, -(i + 1)] for i in range(state.shape[0])]) == 1-player_id))

    scoremax = N ** max(player_count)
    scoremin = N ** max(opponent_count)

    return scoremax - scoremin

def reward_2(res, next_state, current_player):
    reward = 0
    if (current_player == 0 and res == "win") or (current_player == 1 and res== "lose"):
        reward = N**(N+1)
    elif current_player == 1:
        reward = evaluate(next_state, 0)

    return reward

def train(agent, opponent_agents, episode_num, max_episode_steps, init_player=None, render=False, verbose=False):
    win_count = 0
    draw_count = 0
    lose_count = 0
    illegal_count = 0

    reward_history = []
    avg_reward_hist = []

    save_ep = 1000

    for ep in range(episode_num):
        game = QuixoGame()
        state = game.get_current_state()
        if init_player:
            current_player = init_player
        else:
            current_player = random.choice([0, 1])

        reward_sum = 0

        # GLIE epsilon
        final_epsilon = 0.1
        b = 0.8*episode_num*final_epsilon/(1-final_epsilon)
        epsilon = max(b/(b+ep), final_epsilon)


        done = False
        step = 0
        while not done and step < max_episode_steps:
            if current_player == 0:
                action = agent.choose_action_train(state, epsilon)
                next_state, done, res = game.step(action, current_player)

                if done:
                    reward = reward_1(res, step, current_player)
                    # reward = reward_2(res, next_state, current_player)

                    reward_sum += reward

                    agent.store_transition(state, action, reward, next_state, done)
                    agent.learn()



            elif current_player == 1:
                opponent_agent = random.choice(opponent_agents)
                action = opponent_agent.choose_action(state)
                next_state, done, res = game.step(action, current_player)

                reward = reward_1(res, step, current_player)
                # reward = reward_2(res, next_state, current_player)
                
                reward_sum += reward

                agent.store_transition(state, action, reward, next_state, done)
                agent.learn()

            if render:
                print(f"Player: {current_player}, Action: {action}")
                game.print_state()
                input()

            if not done:
                step += 1
                state = next_state
                current_player = 1 - current_player


        reward_history.append(reward_sum)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_hist.append(avg_reward)

        if ep%save_ep == 0:
            agent.save_model()

        if step >= max_episode_steps:
            res = "draw"
        elif (current_player == 0 and res == "win") or (current_player == 1 and res == "lose"):
            res = "win"
        elif (current_player == 0 and res == "lose") or (current_player == 1 and res == "win"):
            res = "lose"

        if res == "win":
            win_count += 1
        elif res == "draw":
            draw_count += 1
        elif res == "lose":
            lose_count += 1
        elif res == "illegal":
            illegal_count += 1

        if verbose:
            print(f"Episode {ep} - Total reward: {reward_sum:.3g} - Avg reward: {avg_reward:.3g} - Epsilon: {epsilon:.2f} - Result: {res.upper()}")


    print(f"Win rate {100*win_count/episode_num:.2f}%")
    print(f"Draw rate {100*draw_count/episode_num:.2f}%")
    print(f"Lose rate {100*lose_count/episode_num:.2f}%")
    print(f"Illegal rate {100*illegal_count/episode_num:.2f}%")



        
if __name__ == "__main__":

    agent = DeepQLearningAgent(player_id=0, gamma=0.98, lr=0.01, state_dim=N*N, batch_size=64, actions_dim=N*4)

    print("Training with RandomAgent (100_000 episodes)")

    opponent_agents = (RandomAgent(player_id=1), )

    train(
        agent=agent,
        opponent_agents=opponent_agents,
        episode_num=100_000,
        max_episode_steps=500,
        verbose=True
    )

    agent.save_model(f"model_{N}x{N}_1.pt")
    print(f"Model saved as model_{N}x{N}_1.pt")


    print("Training with RandomAgent and MiniMaxAgent (100_000 episodes)")

    opponent_agents = (RandomAgent(player_id=1), MiniMaxAgent(player_id=1, depth=2))

    train(
        agent=agent,
        opponent_agents=opponent_agents,
        episode_num=100_000,
        max_episode_steps=500,
        verbose=True
    )

    agent.save_model(f"model_{N}x{N}_2.pt")
    print(f"Model saved as model_{N}x{N}_2.pt")


    print("Training with MiniMaxAgent (100_000 episodes)")

    opponent_agents = (MiniMaxAgent(player_id=1, depth=2), )

    train(
        agent=agent,
        opponent_agents=opponent_agents,
        episode_num=100_000,
        max_episode_steps=500,
        # verbose=True
    )

    agent.save_model(f"model_{N}x{N}_3.pt")
    print(f"Model saved as model_{N}x{N}_3.pt")






    print("Testing DeepQLearningAgent vs RandomAgent (init_player=0):")
    test(
        agent=agent,
        opponent_agents=(RandomAgent(player_id=1), ),
        episode_num=100,
        max_episode_steps=500,
        # init_player=0
    )
    print("-"*40)

    print("Testing DeepQLearningAgent vs RandomAgent (init_player=1):")
    test(
        agent=agent,
        opponent_agents=(RandomAgent(player_id=1), ),
        episode_num=100,
        max_episode_steps=500,
        init_player=1
    )
    print("-"*40)

    print("Testing DeepQLearningAgent vs MiniMaxAgent(2) (init_player=0):")
    test(
        agent=agent,
        opponent_agents=(MiniMaxAgent(player_id=1, depth=2), ),
        episode_num=100,
        max_episode_steps=500,
        # init_player=0
    )
    print("-"*40)

    print("Testing DeepQLearningAgent vs MiniMaxAgent(2) (init_player=1):")
    test(
        agent=agent,
        opponent_agents=(MiniMaxAgent(player_id=1, depth=2), ),
        episode_num=100,
        max_episode_steps=500,
        init_player=1
    )
    print("-"*40)


