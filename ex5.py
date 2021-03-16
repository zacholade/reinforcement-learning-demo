# Please write your code for Exercise 1 in this cell or in as many cells as you want ABOVE this cell.
# You should implement your MC agent and plot your average learning curve here.
# Do NOT delete this cell.

# YOUR CODE HERE
import random
import abc
import numpy as np
from racetrack_env import RacetrackEnv
from typing import List, Union
from collections import defaultdict


def argmax(values: list) -> int:
    # np.argmax returns the first element in a list with the max value.
    # We dont want this. we want to get a random choice of this max val.
    max_val = max(values)
    actions = [a for a in range(len(values)) if values[a] == max_val]
    return np.random.choice(actions)


class Epsilon:
    def __init__(self, initial: Union[int, float]):
        self.value = initial

    @abc.abstractmethod
    def update(self, reward: Union[int, float]) -> Union[int, float]:
        ...


class RewardBasedDecayEpsilon(Epsilon):
    def __init__(self,
                 initial: Union[int, float],
                 minimum: Union[int, float],
                 decay_factor: Union[int, float],
                 reward_threshold: int,
                 reward_increment: int) -> None:
        super().__init__(initial)
        self.minimum = minimum
        self.decay_factor = decay_factor
        self.reward_threshold = reward_threshold
        self.reward_increment = reward_increment

    def update(self, reward: Union[int, float]) -> Union[int, float]:
        if self.value > self.minimum and reward >= self.reward_threshold:
            self.value *= self.decay_factor
            self.reward_threshold += self.reward_increment
        return self.value


class EpsilonGreedyPolicy:
    def __init__(self, Q: dict, actions: List[int], epsilon: Epsilon) -> None:
        self._actions = actions
        self.Q = Q
        self.epsilon = epsilon

    def argmax(self, state: int) -> int:
        # np.argmax returns the first element in a list with the max value.
        # We dont want this. we want to get a random choice of this max val.
        max_val = max(self.Q[state])
        actions = [a for a in self._actions if self.Q[state][a] == max_val]
        return np.random.choice(actions)

    def __call__(self, state: int) -> int:
        """
        Returns the optimal choice with probability of 1-epsilon. Else random.
        """
        probabilities = [(self.epsilon.value / len(self._actions)) for _ in self._actions]
        greedy_action = argmax(self.Q[state])
        probabilities[greedy_action] += (1 - self.epsilon.value)
        return np.random.choice(self._actions, p=probabilities)


class QLearning:
    def __init__(self, env, epsilon: Epsilon, gamma: float, alpha: float, num_episodes: int) -> None:
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.Q = defaultdict(lambda: [0 for _ in self.env.get_actions()])
        self.policy = EpsilonGreedyPolicy(self.Q, env.get_actions(), epsilon)
        self.model = defaultdict(dict)

    def _do_episode(self):
        state = self.env.reset()
        episode = []
        while True:
            action = self.policy(state)
            next_state, reward, is_terminal = self.env.step(action)
            episode.append((state, action, reward))
            best_action = max(self.Q[next_state])
            self.Q[state][action] += self.alpha * (reward + self.gamma * best_action - self.Q[state][action])
            self.model[state][action] = (reward, next_state)

            for i in range(10):
                s = random.choice(list(self.model))
                a = random.choice(list(self.model[s]))
                r, s_ = self.model[s][a]
                a1 = max(self.Q[s_])
                self.Q[s][a] += self.alpha * (r + self.gamma * a1 - self.Q[s][a])

            self.epsilon.update(reward)
            state = next_state
            if is_terminal:
                break
        return episode

    def start(self):
        undiscounted_rewards = []
        for episode_num in range(self.num_episodes):
            episode = self._do_episode()
            undiscounted_rewards.append(sum([step[2] for step in episode]))
            print(undiscounted_rewards)
        return undiscounted_rewards


epsilon = RewardBasedDecayEpsilon(initial=0.15,
                                  minimum=0.01,
                                  decay_factor=0.9,
                                  reward_threshold=-40,
                                  reward_increment=1
                                  )
num_agents = 20
alpha = 0.2
gamma = 0.9  # discount
num_episodes = 150
optimal_policies = []
agents_rewards_list = []
for i in range(num_agents):
    env = RacetrackEnv()
    agent = QLearning(env, epsilon, gamma, alpha, num_episodes)
    rewards = agent.start()
    agents_rewards_list.append(rewards)

mean_agent_rewards = []
for i in range(num_episodes):
    sum_i = 0
    for agent_rewards in agents_rewards_list:
        sum_i += agent_rewards[i]
    reward_avg = sum_i / num_agents
    mean_agent_rewards.append(reward_avg)

print('-----------------')
print(mean_agent_rewards)
import matplotlib.pyplot as plt

plt.plot(mean_agent_rewards)
plt.title(f"Q Learning - {num_agents} agents")
plt.xlabel("Episode Number")
plt.ylabel("Undiscounted Return")
plt.show()
