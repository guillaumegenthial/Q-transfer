import numpy as np
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import gym
import random
import sys
import env_interaction

class FeatureExtractor(object):
    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for x in range(100000)])

        # Fit feature scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Fit feature extractor
        self.feature_map = FeatureUnion([("rbf1", RBFSampler(n_components=50, gamma=1., random_state=1)),
                                         ("rbf01", RBFSampler(n_components=50, gamma=0.1, random_state=1)),
                                         ("rbf10", RBFSampler(n_components=50, gamma=10, random_state=1))])

        self.feature_map.fit(self.scaler.transform(observation_examples))
        self.dim = 150 + 1

    def __call__(self, state, action):
        state = np.array([state])
        state = np.array(state).astype(np.float32)
        state = self.scaler.transform(state)
        state = self.feature_map.transform(state)
        state = state.astype(np.float32)[0]

        return np.append(state, action)


class GlobalApproxValue(object):
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.dim = feature_extractor.dim
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=self.dim)

    def update(self, s, a, t, lr):
        self.weights -= lr*(self.predict(s, a) - t)*self.predict(s, a)

    def predict(self, s, a):
        feature_vector = self.feature_extractor(s, a)
        return np.sum(self.weights * feature_vector)


class Agent(object):
    def __init__(self, env, value_function, actions_nb=3, eps=0.2):
        self.env = env
        self.value_function = value_function
        self.actions_nb = actions_nb
        self.eps = eps

    def best_action(self, s):
        l = [self.value_function.predict(s, a) for a in xrange(self.actions_nb)]
        return np.max(l), np.argmax(l)

    def action(self, state):
        # returns an action
        if np.random.random() < self.eps:
            return random.choice(xrange(self.actions_nb))
        else:
            q, a = self.best_action(state)
            return int(a)

    def target(self, r, sp):
        # returns r + max_a (Q(sp, a))
        q, a = self.best_action(sp)
        return r + q

    def get_policy(self):
        self.eps = 0.
        return lambda s : self.best_action(s)[1]

    def learn(self, max_iter, max_trial, lr=0.001, render=False):
        for e in xrange(max_trial):
            s = env.reset()
            done = False
            i = 0
            while not done and i < max_iter:
                if render: env.render()
                a = self.action(s)
                # print self.value_function.predict(s, a)
                sp, r, done, info = env.step(a)
                t = self.target(r, sp)
                self.value_function.update(s, a, t, lr)
                s = sp
                i += 1

            print('episode {} finished in {} steps'.format(e, i))



# In[]
env_name = 'MountainCar-v0'
env = gym.make(env_name)
feature_extractor = FeatureExtractor(env)
global_value_function = GlobalApproxValue(feature_extractor)
agent = Agent(env, global_value_function)
agent.learn(1000, 100, 0.001, False)
