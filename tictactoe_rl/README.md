# Tic Tac Toe - Reinforcement learning

The objective is to find an optimal agent for playing tic tac toe with reinforcement learning techniques.

## Q-learning
Q-learning is a model-free reinforcement learning algorithm.
This algorithm deals with learning a policy, which is a strategy for making decisions, in a Markov decision process (MDP).
In a MDP the future state depends only on the current state and action, not on the sequence of events that preceded them.
A reward is given every time, given a state, an action is selected to go to the next state.
The goal of Q-learning is to learn an optimal policy maximising the cumulative reward.


## Environment
The `gymnasium` library is exploited to create a custom rl environment.

### State
The state is represented as a numpy array with shape 3x3 filled with 0 (empty), 1 (agent player), 2 (opponent player). 

State example:

`[[0 1 0], [1 0 0], [0 2 2]]`

### Action

An action is represented by an integer number indicating the cell in which the agent wants to put its X.

Action space:

`[[0 1 2], [3 4 5], [6 7 8]]`

### Rewards
Given a state and an action the following rewards are assigned:
- Win reward: 10
- Lose reward: -10
- Draw reward: 3

### Opponent strategies and first turn

The environment class takes an argument to pass a pool of opponent startegies which are chosen randomly and an optional argument to decide the whose is the first turn (random if not set).


## Opponent startegies
### Random strategy
It does what you think.

### Magic strategy (optimal)
It exploit a magic matrix with the property of having the sum of each row column and diagonal equal to 15.

It act randomly until the state contains two cells of the optimal agent.

Then tries to find an action that allow the agent to win, evaluating for each couple of cells of the agent the sum of the corresponding values in the magic matrix subtracted from 15 and then checking, if the values is in the range [1, 9], if the corresponding cell in magic matrix is free and filling that cell.

If no winning moves are possible then tries to find an action that block the opponent player, performing the same operation but considering opponent cells.

### Agent strategy
The agent strategy is used during training to make the agent learn playing against itself.

## Q-learning agent
The agent class holds q_table values which contain the action-value function values for each couple of action-state.
Since not all states are feasible for tic tac toe the q_table is a defaultdict with state as key and an array of 0 values for each action.

### Q table update
In the training phase the q_table is updated with the following rule:

$
Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha \cdot \left[(R_{t+1} + \gamma \cdot \max_{a'} Q(s_{t+1}, a')) -  Q(s_{t}, a_{t})\right]
$
        
where $Q(s_{t}, a_{t})$ is the value of taking action $a$ in state $s_{t}$, $R_{t+1}$ is the immediate reward, $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $s_{t+1}$ is the next state.

The following hyperparameters are user:
- $\alpha$ = 0.1
- $\gamma$ = 0.9

### Action selection
Baed on the exploration rate $\epsilon$ two action selection methods are used:
- Random selection
- Best action selection (action which maximises the future reward)


## Training
The following training steps have been done:
- Training with opponent strategies = (random_strategy)
- Retraining with opponent strategies = (random_strategy, agent_strategy, strategy)
- Retraining with opponent strategies = (agent_strategy, magic_strategy)
  
Each training step is performed evaluating 500_000 episodes.

The $\epsilon$ parameter is evaluated for each episode in compliance with GLIE (Greedy in the Limit with Infinite Exploration) theorem.

$
\epsilon_{k} = \frac{b}{b+k}
$

where $ep$ is the current episode number and $b$ is tuned to have $\epsilon$ = 0.1 at 90% of the training and leaving that value for the remaining 10%.


## Testing and results
Tests are evaluated on 10_000 episodes.

| Opponent startegy | Win rate | Lose rate | Draw rate |
|--|--|--|--|
| random_strategy | 98.12% | 0.00% | 1.88% |
| random_strategy (first move) | 98.13% | 0.00% | 1.87% |
| random_strategy (opponent first move) | 91.49% | 0.00% | 8.51% |
| magic_strategy | 74.89% | 0.00% | 25.11% |
| magic_strategy (first move) | 75.15% | 0.00% | 24.85% |
| magic_strategy (opponent first move) | 6.02% | 0.00% | 93.98% |


# Peer review

## Done
<!-- - [Udrea Florentin](https://github.com/florentin1304/computational-intelligence/issues/1) -->
## Received
<!-- - [Michelangelo Caretto](https://github.com/SimoneBorella/computational-intelligence/issues/2) -->
