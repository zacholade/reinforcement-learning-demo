# Please write your code for Exercise 1 in this cell or in as many cells as you want ABOVE this cell.
# You should implement your MC agent and plot your average learning curve here.
# Do NOT delete this cell.

# YOUR CODE HERE
import numpy as np
import random
from racetrack_env import RacetrackEnv
import typing
from collections import defaultdict

class UniformSoftPolicy:
    def __init__(self, env, epsilon):
        self._actions = env.get_actions()
        self.policy = defaultdict(int)
        self.epsilon = epsilon

    def get(self, s):
        probabilities = [self.policy[s, a] for a in self._actions]
        greedy_prob = max(probabilities)
        greedy_action = np.argmax(probabilities)

        if greedy_prob == 0:
            # We dont have an E-greedy choice yet.
            return np.random.choice(self._actions)
        else:
            rand = np.random.uniform(0, 1)
            if rand < greedy_prob:
                return greedy_action
            return np.random.choice(self._actions)

        # probabilities = [self.policy[s, a] for a in self._actions]
        # return np.random.choice(self._actions, p=probabilities)


class OnPolicyMCControl:
    def __init__(self, env, epsilon, gamma, num_episodes):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.positions = len(env.track) * len(env.track[0])
        self.policy = UniformSoftPolicy(env, epsilon)

    def _do_episode(self, episode_num):
        state = self.env.reset()
        episode = []
        while True:  # Loop forever for each episode.
            action = self.policy.get(state)
            next_state, reward, is_terminal = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if is_terminal:
                break

        return episode

    def start(self):
        Q = defaultdict(float)
        returns = defaultdict(list)
        undiscounted_rewards = []
        for episode_num in range(self.num_episodes):
            episode = self._do_episode(episode_num)
            G = 0
            undiscounted_reward = 0
            s_a_episode = [(s, a) for s, a, _ in episode]
            for t, (s, a, r) in enumerate(reversed(episode)):
                G = self.gamma * G + r
                undiscounted_reward += r
                if (s, a) in s_a_episode[:t]:
                    continue

                returns[s, a].append(G)
                Q[s, a] = sum(returns[s, a]) / len(returns[s, a])
                a1 = np.argmax([Q[s, a_] for a_ in self.env.get_actions()])

                for a_ in self.env.get_actions():
                    if a_ == a1:
                        self.policy.policy[s, a_] = 1 - self.epsilon + self.epsilon / len(self.env.get_actions())
                    else:
                        self.policy.policy[s, a_] = self.epsilon / len(self.env.get_actions())
            undiscounted_rewards.append(undiscounted_reward)
            print(undiscounted_rewards)
        return undiscounted_rewards


num_agents = 10
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

