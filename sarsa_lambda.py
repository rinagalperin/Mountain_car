import gym
import numpy as np
from numpy.random import random, choice


class SarsaLambda:
    def __init__(self, gamma=1, lambda_p=0.5, alpha=0.02, steps=100000, epsilon_decay=0.99):
        self.env = gym.make("MountainCar-v0")
        self.gamma = gamma
        self.steps = steps
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay  # decay per episode
        self.lambda_p = lambda_p
        self.w = None
        self.name = 'Sarsa Lambda'
        self.centers = self.get_gaussian()

    def get_gaussian(self,  n_feature_p=8,  n_feature_v=4):
        """
        given the number of features for the position and the number of features for the velocity,
        creates a gaussian in equal intervals  [min_p, max_p] and [min_v, max_v] ,
        using Numpy’s function landscape.
        """
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        c_p, c_v = np.linspace(low[0], high[0], n_feature_p), np.linspace(low[1], high[1], n_feature_v)
        return np.array([(c_p[i], c_v[j]) for j in range(n_feature_v) for i in range(n_feature_p)])

    def epsilon_greedy(self, Q_s, epsilon):
        """
        in probability epsilon, draws random action. otherwise, chooses the action that maximizes Q_s.
        Note that if Q_s contains equal values - we draw randomly among them (unlike the default behavior of np.argmax)
        """
        if random() < epsilon:
            return choice(len(Q_s))
        else:
            return choice(np.flatnonzero(Q_s == Q_s.max()))

    def get_action(self, s_t):
        """
        once we learned a policy using Learn function, greedily selects the best action
        """
        theta = self.comp_theta(s_t)
        q_s = self.compute_q(self.w, theta)
        return np.argmax(q_s)

    def monte_carlo_policy_evaluation(self, w, n=150):
        """
        Monte Carlo policy estimation algorithm, runs n times to evaluate the value of the initial state only.
        """
        v_start = 0
        for _ in range(n):
            s_t = self.env.reset()
            done = False
            g_t = 0
            gamma_t = 1
            while not done:
                theta = self.comp_theta(s_t)
                q_s = self.compute_q(w, theta)
                a_t = np.argmax(q_s)
                s_t, r_t, done, info = self.env.step(a_t)
                g_t += gamma_t * r_t
                gamma_t *= self.gamma

            v_start += g_t

        return v_start / n

    def comp_theta(self, s_t):
        """
        computes theta value as suggested in the formula in the task’s instructions.
        """
        x = s_t - self.centers
        x_t = [np.reshape(v, (1, 2)) for v in x]
        A = np.linalg.inv(np.diag((0.04, 0.0004)))
        var = np.array([np.dot(np.dot(x_t[i], A), x[i])[0] for i in range(len(x))])
        return np.exp(-var/2)

    def compute_q(self, w, theta):
        return np.dot(w, theta)

    def learn(self, interval=5000):
        """sarsa lambda learning algorithm. once per interval evaluates the value of the initial state."""
        n_actions, n_rbf = self.env.action_space.n, len(self.centers)
        w = np.zeros((n_actions, n_rbf))
        x, y = [], []
        t, epsilon = 0, 1
        max_v = -float('inf')

        while t < self.steps:
            s_prev = self.env.reset()
            E = np.zeros((n_actions, n_rbf))
            theta_prev = self.comp_theta(s_prev)
            q_prev = self.compute_q(w, theta_prev)
            a_prev = self.epsilon_greedy(q_prev, epsilon)
            done = False

            while not done:
                E[a_prev, :] += theta_prev

                s_t, r, done, _ = self.env.step(a_prev)
                theta = self.comp_theta(s_t)
                q_s = self.compute_q(w, theta)
                a_t = self.epsilon_greedy(q_s, epsilon)
                delta = r + self.gamma * q_s[a_t] - q_prev[a_prev]

                w += self.alpha * delta * E
                E = self.lambda_p * self.gamma * E

                q_prev, a_prev, theta_prev = q_s, a_t, theta
                t += 1

                if t % interval == 0:
                    v = self.monte_carlo_policy_evaluation(w)
                    if max_v <= v:
                        self.w = w
                        max_v = v

                    y.append(v)
                    x.append(t)

            epsilon *= self.epsilon_decay

        return x, y
