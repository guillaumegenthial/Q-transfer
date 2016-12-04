"""
Ensemble method for Q-learning
"""

import random, math, pickle
from collections import defaultdict
from copy import deepcopy
from streamplot import PlotManager
from base_rl import SimpleQLearning
import numpy as np

class EnsembleQLearning(SimpleQLearning):
    def __init__(self, name, sources, actions, discount, explorationProb = 0.2):
        """
        `actions` is the list of possible actions at any state
        `sources` a list of source SimpleQLearning
        """
        self.name = name
        self.sources = sources
        self.n_sources = len(sources)
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.numIters = 0
        self.coefs = [1./self.n_sources] * self.n_sources

    def preliminaryCheck(self, state, action):
        sources = self.sources
        index_to_remove = []
        for i in xrange(self.n_sources):
            if sources[i].evalQ(state, action) > 10**2:
                print("ERROR : task {} has Q value too large".format(sources[i].name))
                index_to_remove += [(i, sources[i].name)]
        for i, name in index_to_remove:
            print("Removing task {} from sources".format(name))
            del sources[i]

        self.n_sources -= len(index_to_remove)
        self.coefs = [1./self.n_sources] * self.n_sources

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        sources = self.sources
        return sum(self.coefs[i] * sources[i].evalQ(state, action) for i in xrange(self.n_sources))

    def getStepSize(self):
        return 0.000001

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

        print self.coefs
        for i in xrange(self.n_sources):
            self.coefs[i] = self.coefs[i] - self.getStepSize() * (pred - target) * self.sources[i].evalQ(state, action)

    def dump(self, file_name):
        with open(file_name, "wb") as fout:
            pickle.dump(self.coefs, fout)

####################################################

def target_train(env, name, sources, num_trials=1, max_iter=10, filename="weights.p", verbose=False, reload_weights=True, discount=1, explorationProb=0.1):
    weights = filename if reload_weights else None
    actions = range(env.action_space.n)

    rl_ens = EnsembleQLearning(
        name, 
        sources, 
        range(env.action_space.n), 
        discount
        )

    rl_ens.preliminaryCheck(np.array([-0.5, 0]),0)

    rl_ens.train(
        env, 
        num_trials=num_trials, 
        max_iter=max_iter, 
        verbose=verbose
        )

    rl_ens.dump("weights/" + filename)

    return rl_ens
