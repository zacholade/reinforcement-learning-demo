# Please write your code for Exercise 1 in this cell or in as many cells as you want ABOVE this cell.
# You should implement your MC agent and plot your average learning curve here.
# Do NOT delete this cell.

# YOUR CODE HERE
import numpy as np
from racetrack_env import RacetrackEnv
from typing import List
from collections import defaultdict


def argmax(values: list) -> int:
    # np.argmax returns the first element in a list with the max value.
    # We dont want this. we want to get a random choice of this max val.
    max_val = max(values)
    actions = [a for a in range(len(values)) if values[a] == max_val]
    return np.random.choice(actions)


class EpsilonGreedyPolicy:
    def __init__(self, Q: dict, actions: List[int], epsilon: float) -> None:
        self._actions = actions
        self.Q = Q
        self.epsilon = epsilon

    def __call__(self, state: int) -> int:
        """
        Returns the optimal choice with probability of 1-epsilon. Else random.
        """
        probabilities = [(self.epsilon / len(self._actions)) for _ in self._actions]
        greedy_action = argmax(self.Q[state])
        probabilities[greedy_action] += (1 - self.epsilon)
        return np.random.choice(self._actions, p=probabilities)


class SarsaOnPolicyTDControl:
    def __init__(self, env, epsilon: float, gamma: float, alpha: float, num_episodes: int) -> None:
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.Q = defaultdict(lambda: [0 for _ in self.env.get_actions()])
        self.policy = EpsilonGreedyPolicy(self.Q, env.get_actions(), epsilon)

    def _do_episode(self):
        state = self.env.reset()
        episode = []
        action = self.policy(state)
        while True:
            next_state, reward, is_terminal = self.env.step(action)
            next_action = self.policy(next_state)
            episode.append((state, action, reward))
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
            state = next_state
            action = next_action
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


num_agents = 20
alpha = 0.2
epsilon = 0.15  # chance of doing something random
gamma = 0.9  # discount
num_episodes = 150
optimal_policies = []
agents_rewards_list = []
for i in range(num_agents):
    env = RacetrackEnv()
    agent = SarsaOnPolicyTDControl(env, epsilon, gamma, alpha, num_episodes)
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
plt.title(f"Sarsa - Averaged {num_agents} agents")
plt.xlabel("Episode Number")
plt.ylabel("Undiscounted Return")
plt.show()
