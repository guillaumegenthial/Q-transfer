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
    def __init__(
        self, 
        name, 
        sources, 
        actions, 
        discount, 
        weights=None, 
        exploration_start=1.,
        exploration_end=0.1,
        eligibility=0.9, 
        reload_weights=False):
        """
        `actions` is the list of possible actions at any state
        `sources` a list of source SimpleQLearning
        """
        self.name = name
        self.sources = sources
        self.n_sources = len(sources)
        self.actions = actions
        self.discount = discount
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.explorationProb = exploration_start
        self.numIters = 0
        self.eligibility = eligibility

        if weights and reload_weights:
            self.load(weights)
        else:
            self.default_load()

        # self.plt_mgr = PlotManager(title="reward")

    def default_load(self):
        self.coefs = [1./self.n_sources] * self.n_sources        

    def progress(self, compute=False):
        try:
            if compute:
                return np.sum(np.square(self.coefs - self.coefs_bak))
            else:
                self.coefs_bak = deepcopy(self.coefs)
        except Exception, e:
            pass

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

    def normalize(self):
        s = sum(self.coefs)
        self.coefs /= s

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        sources = self.sources
        return sum(self.coefs[i] * sources[i].evalQ(state, action) for i in xrange(self.n_sources))

    def getStepSize(self):
        return 0.001

    def updateQ(self, state, action, gradient):
        for i in xrange(self.n_sources):
            self.coefs[i] = self.coefs[i] - self.getStepSize() * gradient * self.sources[i].evalQ(state, action)

        self.print_coefs()
        self.normalize()

    def print_coefs(self):
        pp = []
        for i, source in enumerate(self.sources):
            pp += ["%s: %.4f"% (source.name, self.coefs[i])]
        print(", ".join(pp))

    def dump(self, file_name):
        print("Saving coefs to file {}".format(file_name))
        self.print_coefs()
        with open(file_name, "wb") as fout:
            pickle.dump(self.coefs, fout)

    def load(self, file_name):
        """
        Load weights from pickle file dict into the default dict
        """
        with open(file_name, "rb") as fin:
            print("Loading coefs from file {}".format(file_name))
            self.coefs = pickle.load(fin)
