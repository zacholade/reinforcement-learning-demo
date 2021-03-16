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
    def __init__(self, actions: List[int], epsilon: float) -> None:
        self._actions = actions
        self.mapping = defaultdict(lambda: [0 for _ in self._actions])
        self.epsilon = epsilon

    def __call__(self, state: int) -> int:
        """
        Returns the optimal choice with probability of 1-epsilon. Else random.
        """
        probabilities = [(self.epsilon / len(self._actions)) for _ in self._actions]
        greedy_action = argmax(self.mapping[state])
        probabilities[greedy_action] += (1 - self.epsilon)
        return np.random.choice(self._actions, p=probabilities)


class OnPolicyMCControl:
    def __init__(self, env, epsilon, gamma, num_episodes):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.positions = len(env.track) * len(env.track[0])
        self.Q = defaultdict(lambda: [0 for _ in self.env.get_actions()])
        self.policy = EpsilonGreedyPolicy(env.get_actions(), epsilon)

    def _do_episode(self, episode_num):
        state = self.env.reset()
        episode = []
        while True:  # Loop forever for each episode.
            action = self.policy(state)
            next_state, reward, is_terminal = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if is_terminal:
                break

        return episode

    def start(self):
        returns = defaultdict(lambda: [[] for _ in self.env.get_actions()])
        undiscounted_returns = []
        for episode_num in range(self.num_episodes):
            episode = self._do_episode(episode_num)
            G = 0
            undiscounted_return = 0
            s_a_episode = [(s, a) for s, a, _ in episode]
            for t, (s, a, r) in enumerate(reversed(episode)):
                G = self.gamma * G + r
                undiscounted_return += r
                if (s, a) in s_a_episode[:t]:
                    continue

                returns[s][a].append(G)
                self.Q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                a1 = argmax(self.Q[s])

                for a_ in self.env.get_actions():
                    if a_ == a1:
                        self.policy.mapping[s][a_] = 1 - self.epsilon + self.epsilon / len(self.env.get_actions())
                    else:
                        self.policy.mapping[s][a_] = self.epsilon / len(self.env.get_actions())
            undiscounted_returns.append(undiscounted_return)
            print(undiscounted_returns)
        return undiscounted_returns


num_agents = 20
epsilon = 0.15  # chance of doing something random
gamma = 0.9  # discount
num_episodes = 150
optimal_policies = []
agents_rewards_list = []
for i in range(num_agents):
    env = RacetrackEnv()
    agent = OnPolicyMCControl(env, epsilon, gamma, num_episodes)
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
plt.title(f"Monte Carlo Control - {num_agents} agents")
plt.xlabel("Episode Number")
plt.ylabel("Undiscounted Return")
plt.show()

