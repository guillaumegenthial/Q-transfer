"""
Reinforcement learning
"""

import random, math, pickle
from collections import defaultdict
from copy import deepcopy

class SimpleQLearning:
    def __init__(self, actions, discount, featureExtractor, explorationProb = 0.2, weights = None):
        """
        `actions` is the list of possible actions at any state
        `featureExtractor` takes a (state, action) and returns a feature dictionary
        `weights` is an optional file containing pre-computed weights
        """
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        
        if weights:
            with open(weights, "rb") as fin:
                weights_ = pickle.load(fin)
                self.weights = weights_
        else:
            self.weights = defaultdict(float)

    def makeGreedy(self):
        """
        Set explorationProb to 0, ie. greedily chooses best actions
        """
        self.explorationProb = 0.

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        Best action from `state`, given current estimation of Q.
        With probability `explorationProb` take a random action (epsilon-greedy).
        """
        self.numIters += 1
        if len(self.actions) == 0:
            return None
        
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return max((self.evalQ(state, action), action) for action in self.actions)[1]

    def getStepSize(self):
        """
        Get the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)

    def updateQ(self, state, action, reward, newState):
        if newState is None:
            return
        
        phi = self.featureExtractor(state, action)
        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions)
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v

    def train(self, env, num_trials = 100, max_iter = 1000, verbose=False):
        """
        Learn the weights by running simulations
        """
        totalRewards = []  # The rewards we get on each trial
        for trial in xrange(num_trials):
            # init
            totalDiscount = 1
            totalReward = 0
            done = False
            state = env.reset() # start state

            for it in xrange(max_iter):
                if done:
                    if verbose: print("Episode finished after {} timesteps".format(it+1))
                    break

                action = self.getAction(state)
                newState, reward, done, info = env.step(action)
                self.updateQ(state, action, reward, newState)

                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            totalRewards.append(totalReward)

        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards


############################################################

def simpleFeatures(state, action):
    """
    Return a list of (feature-name, value)
    """
    pos, vel = state
    features = [((pos * 100, vel * 1000, action), 1), (('pos', pos * 100, action), 1), (('vel', vel * 1000, action), 1), (action, 1)]
    return features

############################################################

def rl_policy(env, featureExtractor, num_trials = 1000, max_iter = 500, filename = "weights.p", verbose = False):
    actions = range(env.action_space.n)
    rl = SimpleQLearning(actions, discount = .99, featureExtractor = featureExtractor, explorationProb = 0.2)
    rl.train(env, num_trials = num_trials, max_iter = max_iter, verbose = verbose)
    rl.makeGreedy()
    policy = lambda s : rl.getAction(s)

    # save learned weights
    with open(filename, "wb") as fout:
        weights = dict(rl.weights)
        pickle.dump(weights, fout)
    
    return policy

def load_policy(filename, env, featureExtractor):
    actions = range(env.action_space.n)
    rl = SimpleQLearning(actions, discount = .99, featureExtractor = featureExtractor, explorationProb = 0., weights = filename)
    policy = lambda s : rl.getAction(s)    
    return policy