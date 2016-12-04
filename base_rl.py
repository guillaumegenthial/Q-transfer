"""
Reinforcement learning
"""

import random, math, pickle
from collections import defaultdict
from copy import deepcopy
from streamplot import PlotManager
import env_interaction
import sys

class SimpleQLearning:
    def __init__(self, name, actions, discount, featureExtractor, explorationProb = 0.2, weights = None):
        """
        `actions` is the list of possible actions at any state
        `featureExtractor` takes a (state, action) and returns a feature dictionary
        `weights` is an optional file containing pre-computed weights
        """
        self.name = name
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0

        if weights:
            self.load(weights)
        else:
            self.weights = defaultdict(float)


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

    def updateQ(self, state, action, reward, newState):
        if newState is None:
            return
        
        phi = self.featureExtractor(state, action)
        pred = self.evalQ(state, action)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions)
        except:
            print "error"
            v_opt = 0.
        target = reward + self.discount * v_opt
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v

    def train(self, env, num_trials=100, max_iter=1000, verbose=False):
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

            for it in xrange(max_iter):
                if done:
                    if verbose: 
                        print("Episode finished after {} timesteps".format(it+1))
                    break
                # if verbose:
                #     env.render()
                action = self.getAction(state)
                newState, reward, done, info = env.step(action)
                self.updateQ(state, action, reward, newState)
                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState

            totalRewards.append(totalReward)

            # plotting and printing
            plt_mgr.add(name="Task {}, Reward".format(self.name), x=trial, y=totalReward)
            plt_mgr.update()
            sys.stdout.write("\rTrial nb {}, total reward {}".format(trial, totalReward))
            sys.stdout.flush()

        print("\nAverage reward:", sum(totalRewards)/num_trials)
        plt_mgr.export("plots")
        plt_mgr.close(force=True)
        return totalRewards



############################################################

def rl_train(name, env, featureExtractor, num_trials=1, max_iter=10000, filename="weights.p", verbose = False, reload_weights=True, discount=1, explorationProb=0.1):
    weights = filename if reload_weights else None
    actions = range(env.action_space.n)

    rl = SimpleQLearning(
        name, 
        actions, 
        discount=discount, 
        featureExtractor=featureExtractor, 
        explorationProb=explorationProb, 
        weights=weights
        )
    rl.train(
        env, 
        num_trials=num_trials, 
        max_iter=max_iter, 
        verbose=verbose
        )

    rl.dump("weights/"+filename)
    
    return rl

def rl_load(name, filename, env, featureExtractor, discount=1):
    actions = range(env.action_space.n)

    rl = SimpleQLearning(
        name, 
        actions, 
        discount=discount, 
        featureExtractor=featureExtractor, 
        explorationProb=0., 
        weights="weights/" + filename
        )
     
    return rl

def train_task(env, name, param, num_trials, max_iter, verbose, reload_weights, discount, explorationProb):
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
        name, 
        env, 
        env_interaction.simpleFeatures(env), 
        num_trials=num_trials, 
        max_iter=max_iter,
        filename=file_name, 
        verbose=verbose,
        reload_weights=reload_weights, 
        discount=discount, 
        explorationProb=explorationProb)

    evaluation = env_interaction.policy_evaluation(
        env, 
        rl.getPolicy(), 
        discount=discount)

    with open("results.txt", "a") as f:
        f.write("{} {}\n".format(name, evaluation))