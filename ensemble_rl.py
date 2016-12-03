"""
Ensemble method for Q-learning
"""

import random, math, pickle
from collections import defaultdict
from copy import deepcopy
from streamplot import PlotManager
from base_rl import SimpleQLearning

class EnsembleQLearning(SimpleQLearning):
    def __init__(self, sources, actions, discount, explorationProb = 0.2):
        """
        `actions` is the list of possible actions at any state
        `sources` a list of source SimpleQLearning
        """
        self.sources = sources
        self.n_sources = len(sources)
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.numIters = 0
        self.plt_mgr = PlotManager(title="reward")
        self.coefs = [1./self.n_sources] * self.n_sources

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        return sum(self.coefs[i] * sources[i].evalQ(state, action) for i in xrange(self.n_sources))

    def updateQ(self, state, action, reward, newState):
        if newState is None:
            return
        
        pred = self.evalQ(state, action)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions)
        except:
            print "error"
            v_opt = 0.
        target = reward + self.discount * v_opt

        for i in xrange(self.n_sources):
            self.coefs[i] = self.coefs[i] - self.getStepSize() * (pred - target) * self.sources[i].evalQ(state, action)
