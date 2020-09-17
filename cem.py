import gym
import numpy as np
from collections import deque
import math

env = gym.make('MountainCarContinuous-v0')

class Agent():
    def __init__(self, s_size=2, h_size=16, a_size=1):
        self.w1 = 1e-4*np.random.randn(s_size, h_size)
        self.w2 = 1e-4*np.random.randn(h_size, a_size)
        self.b1 = 1e-4*np.random.randn(h_size)
        self.b2 = 1e-4*np.random.randn(a_size)

    def set_weights(self, weights):
        # establish splices
        w1 = self.w1.size
        b1 = self.b1.size
        w2 = self.w2.size
        b2 = self.b2.size
        # set weights
        self.w1 = weights[ : w1].reshape(self.w1.shape)
        self.b1 = weights[w1 : w1 + b1]
        self.w2 = weights[w1 + b1 : w1 + b1 + w2].reshape(self.w2.shape)
        self.b2 = weights[w1 + b1 + w2 : ]

    def get_weights_dim(self):
        return self.w1.size + self.w2.size + self.b1.size + self.b2.size

    def relu(self, matrix):
        return np.maximum(matrix, 0)

    def forward(self, state):
        x = self.relu(self.b1 + (np.dot(state.T, self.w1)))
        return np.tanh(self.b2 + (np.dot(x, self.w2)))

    #def act(self, state, weights):
    #    magnitudes = self.forward(state, weights)
    #    return magnitudes

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        for t in range(max_t):
            action = self.forward(state)
            state, reward, done, _ = env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return
    
def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    """Implementation of a cross-entropy method.

    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    agent = Agent()
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma*np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100,np.mean(scores_deque)))
            break
    return scores, agent

if __name__ == "__main__":
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    print('  - low:', env.action_space.low)
    print('  - high:', env.action_space.high)
    
    scores, agent = cem()
    
    state = env.reset()
    for t in range(1000):
        env.render()
        action = agent.forward(state)
        state, reward, done, _ = env.step(action)
        if done:
            env.close()
            break