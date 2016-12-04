"""
Reinforcement learning
"""

import random, math, pickle
from collections import defaultdict
from copy import deepcopy
from streamplot import PlotManager
import env_interaction
import sys
import numpy as np


class SimpleQLearning:
    def __init__(self, 
        name, 
        actions, 
        discount, 
        discreteExtractor, 
        featureExtractor, 
        explorationProb=0.2, 
        weights=None, 
        eligibility=0.9):
        """
        `actions` is the list of possible actions at any state
        `featureExtractor` takes a (state, action) and returns a feature dictionary
        `weights` is an optional file containing pre-computed weights
        """
        self.name = name
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.discreteExtractor = discreteExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        self.eligibility = eligibility

        if weights:
            self.load(weights)
        else:
            self.weights = defaultdict(float)

        self.el_traces = defaultdict(float)

    def normalize(self):
        M = 1
        for k, v in self.weights.iteritems():
            if np.abs(v) > M:
                M = np.abs(v)
        for k in self.weights.iterkeys():
            self.weights[k] /= M

    def progress(self, compute=False):
        """
        if compute == True
            Compute square 2-norm of the difference between weights_bak and weights
        else:
            stores a snapshot of the weights in weigts.bak
        """
        if compute:
            progress = 0
            for k, v in self.weights.iteritems():
                try:
                    progress += (self.weights_bak[k] - v)**2
                except Exception, e:
                    progress += v**2

            return progress
        else:
            self.weights_bak = deepcopy(self.weights)


    def load(self, file_name):
        """
        Load weights from pickle file dict into the default dict
        """
        with open(file_name, "rb") as fin:
            print("Loading weights from file {}".format(file_name))
            weights_ = pickle.load(fin)
            self.weights = defaultdict(float)
            for k, v in weights_.iteritems():
                self.weights[k] = v

    def dump(self, file_name):
        """
        Dumps weights in pickle file dict
        """
        print("Saving weights to file {}".format(file_name))
        with open(file_name, "wb") as fout:
            weights = dict(self.weights)
            pickle.dump(weights, fout)

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

    def getPolicy(self):
        self.makeGreedy()
        return lambda s : self.getAction(s)

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
        return 0.1 
        # return 1.0 / math.sqrt(self.numIters)

    def resetElTraces(self):
        """
        Reinitialize counts
        """
        self.el_traces = defaultdict(float)

    def filterElTraces(self):
        keys = []
        for key, trace in self.el_traces.iteritems():
            if trace < 0.01:
                keys += [key]
        for key in keys:
            del self.el_traces[key]

    def updateElTraces(self, state, action, gradient):
        """
        Update for Q(lambda) algorithm
        """
        self.el_traces[self.discreteExtractor[0](state, action)] += 1
        for key, trace in self.el_traces.iteritems():
            s, a = self.discreteExtractor[1](key)
            g = gradient * trace
            self.updateQ(s, a, g)
            self.el_traces[key] *= self.discount * self.eligibility
        
        self.filterElTraces()

    def updateQ(self, state, action, gradient):
        phi = self.featureExtractor(state, action)
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (gradient) * v

    def _update(self, state, action, reward, newState, eligibility=False):
        # Compute gradient (update)
        pred = self.evalQ(state, action)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions)
        except:
            print "error"
            v_opt = 0.

        target = reward + self.discount * v_opt
        gradient = (pred - target)

        # perform update
        if eligibility:
            self.updateElTraces(state, action, gradient)
        else:
            self.updateQ(state, action, gradient)


    def train(self, env, num_trials=100, max_iter=1000, verbose=False, eligibility=False):
        """
        Learn the weights by running simulations
        """
        plt_mgr = PlotManager(title="reward")
        totalRewards = []  # The rewards we get on each trial
        for trial in xrange(num_trials):
            # init
            totalDiscount = 1
            totalReward = 0
            done = False
            state = env.reset() # start state
            self.resetElTraces() # el traces to zero
            self.progress(False)

            for it in xrange(max_iter):
                if done:
                    if verbose: 
                        print("Episode finished after {} timesteps".format(it+1))
                    break
                # if verbose:
                #     env.render()
                action = self.getAction(state)
                newState, reward, done, info = env.step(action)
                self._update(state, action, reward, newState, eligibility)
                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            totalRewards.append(totalReward)
            progress = self.progress(True)

            # plotting and printing
            plt_mgr.add(name="Task {}, Reward".format(self.name), x=trial, y=totalReward)
            plt_mgr.update()
            sys.stdout.write("\rTrial nb {}, total reward {}, progress {}".format(trial, totalReward, progress))
            sys.stdout.flush()

        print("\nAverage reward: {}".format(sum(totalRewards)/num_trials))
        plt_mgr.export("plots")
        plt_mgr.close(force=True)
        return totalRewards



############################################################

def rl_train(
    name, 
    env, 
    discreteExtractor, 
    featureExtractor, 
    num_trials=1, 
    max_iter=10000, 
    filename="weights.p", 
    verbose = False, 
    reload_weights=True, 
    discount=1, 
    explorationProb=0.1, 
    eligibility=False):

    weights = filename if reload_weights else None
    actions = range(env.action_space.n)

    rl = SimpleQLearning(
        name=name, 
        actions=actions, 
        discount=discount, 
        discreteExtractor=discreteExtractor, 
        featureExtractor=featureExtractor, 
        explorationProb=explorationProb, 
        weights=weights
        )
    rl.train(
        env=env, 
        num_trials=num_trials, 
        max_iter=max_iter, 
        verbose=verbose, 
        eligibility=eligibility,
        )

    rl.dump("weights/"+filename)
    
    return rl

def rl_load(name, discreteExtractor, featureExtractor, filename, env, discount=1):
    actions = range(env.action_space.n)

    rl = SimpleQLearning(
        name=name, 
        actions=actions, 
        discount=discount, 
        discreteExtractor=discreteExtractor, 
        featureExtractor=featureExtractor, 
        explorationProb=0., 
        weights="weights/" + filename
        )
     
    return rl

def train_task(
    env, 
    discreteExtractor, 
    featureExtractor, 
    name, 
    param, 
    num_trials, 
    max_iter, 
    verbose, 
    reload_weights, 
    discount, 
    explorationProb, 
    eligibility):
    """
    perform task training

    saves plot of performance during training in /plots
    saves weights in /weights
    writes policy evaluation in file result.txt
    """
    print("Task {}".format(name))
    file_name = param["file_name"]
    slope = param["slope"]
    reward_modes = param["reward_modes"]
    max_speed = param["max_speed"]
    power = param["power"]

    env.set_task(reward_modes, slope, max_speed, power)

    rl = rl_train(
        name=name, 
        env=env, 
        discreteExtractor=discreteExtractor, 
        featureExtractor=featureExtractor, 
        num_trials=num_trials, 
        max_iter=max_iter,
        filename=file_name, 
        verbose=verbose,
        reload_weights=reload_weights, 
        discount=discount, 
        explorationProb=explorationProb,
        eligibility=eligibility)

    rl.normalize()

    evaluation = env_interaction.policy_evaluation(
        env=env, 
        policy=rl.getPolicy(), 
        discount=discount)


    with open("results.txt", "a") as f:
        f.write("{} {}\n".format(name, evaluation))