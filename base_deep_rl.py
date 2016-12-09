import numpy as np
import collections
import pickle
import copy
import random
from streamplot import PlotManager
import theano
import theano.tensor as T
import base_rl
import env_interaction
from base_rl import SimpleQLearning

floatX = theano.config.floatX


class DeepQLearning(SimpleQLearning):
    def __init__(self, 
            name, 
            actions, 
            discount, 
            exploration_start=1.,
            exploration_end=0.1, 
            weights=None, 
            eligibility=0.9, 
            reload_freq=1000,
            experience_replay_size=5000,
            state_size=4):

        self.name = name
        self.actions = actions
        self.discount = discount
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.explorationProb = exploration_start
        self.numIters = 0
        self.eligibility = eligibility
        self.experience_replay_size = experience_replay_size
        self.experience_replay = []
        self.state_size = state_size
        self.batch_size = 32

        # handle update from time to time for params for eval
        self.reload_freq = reload_freq
        # counts the number of calls to evalQ without reloading params
        self.steps = 0

        if weights:
            self.load(weights)
        else:
            # weights of network
            self.params = collections.OrderedDict()
            # copy updated from time to time
            self.params_bak = collections.OrderedDict()

        self.add_Q()

        # self.plt_mgr = PlotManager(title="Plots")


    def load(self, file_name):
        """
        Load weights from pickle file dict into the default dict
        """
        with open(file_name, "rb") as fin:
            print("Loading params from file {}".format(file_name))
            self.params = pickle.load(fin)
            self.params_bak = copy.deepcopy(self.params)

    def makeGreedy(self):
        """
        Set explorationProb to 0, ie. greedily chooses best actions
        """
        self.reload_params()
        self.explorationProb = 0.

    def update_experience_replay(self, s, a, r, sp, d):
        """
        Adds transition to memory and remove first inserted element
        """
        if len(self.experience_replay) == self.experience_replay_size:
            self.experience_replay.pop(0)
        elif len(self.experience_replay) > self.experience_replay_size:
            print "ERROR : memory replay should have size less than {}".format(self.experience_replay_size)
        else:
            pass
            
        self.experience_replay.append((s, a, r, sp, d))

        return len(self.experience_replay) >= 2*self.batch_size

    def sample_experience_replay(self, batch_size=2):
        """
        Returns samples from memory replay
        """
        n = len(self.experience_replay)
        sample_size = min(n, batch_size)
        return [self.experience_replay[i] for i in random.sample(xrange(n), sample_size)]

    def reload_params(self):
        """
        Update params_bak with param
        """
        self.steps = 0
        self.params_bak = copy.deepcopy(self.params)


    def dump(self, file_name):
        """
        Dumps weights in pickle file dict
        """
        print("Saving params to file {}".format(file_name))
        with open(file_name, "wb") as fout:
            pickle.dump(self.params, fout)

    def init(self, size, init="uniform"):
        if init == "uniform":
            return np.random.uniform(low=-0.05, high=0.05, size=size)
        elif init == "ones":
            return 0.05*np.ones(size)

    def parameter_matrix(self, name, size=None, bak=False, weights=None, init="uniform"):
        if name not in self.params:
            if weights:
                self.params[name] = theano.shared(weights.astype(floatX), name)
                self.params_bak[name] = theano.shared(weights.astype(floatX), name)
            else:
                weights = self.init(size, init)
                self.params[name] = theano.shared(weights.astype(floatX), name)
                self.params_bak[name] = theano.shared(weights.astype(floatX), name)

        if bak:
            return self.params_bak[name]
        else:
            return self.params[name]

    def activation(self, activation, x):
        if activation == "softmax":
            return T.nnet.softmax(x)
        elif activation == "sigmoid":
            return T.nnet.sigmoid(x)
        elif activation == "tanh":
            return T.tanh(x)
        elif activation == "relu":
            return T.nnet.relu(x, 0)
        else:
            print("ERROR, {} is an unknown activation".format(activation))
            raise

    def fully_connected(self, name, x, x_dim, y_dim, bak=False, activation=None, offset=True, init="uniform"):
        M = self.parameter_matrix("M_{}".format(name), (x_dim, y_dim), bak=bak, init=init)

        z = T.dot(x, M)

        if offset:
            b = self.parameter_matrix("b_{}".format(name), (y_dim,), bak=bak, init=init)
            z += b


        if activation:
            return self.activation(activation, z)
        else:
            return z

    def output(self, state, bak=False):
       h1 = self.fully_connected("h1", state, self.state_size, 10, bak, "relu")
       h2 = self.fully_connected("h2", h1, 10, 10, bak, "relu")

       output = self.fully_connected("out", h2, 10, len(self.actions), bak, None)

       return output

    def add_Q(self):
        """
        Adds 2 functions to the class
            - Q_eval : to evaluate Q value (state, action)
            - Q_update : to update parameters of Q-transfert
        """
        # 1. Q eval
        # inputs
        state = T.vector("state", floatX)
        action = T.iscalar("action")
        target = T.scalar("target", floatX)

        # compute output with bak params
        output_eval = self.output(state, True)[action]
        self.Q_eval = theano.function(
            [state, action], 
            output_eval, 
            on_unused_input='warn', 
            allow_input_downcast=True)

        # 2. Q_update 
        # inputs
        states = T.matrix("state", floatX)
        actions = T.ivector("action")
        targets = T.vector("target", floatX)

        # compute output with regular params
        outputs = self.output(states, False)
        outputs = outputs[T.arange(outputs.shape[0], dtype="int32"), actions]
        # get gradients
        loss = T.mean(T.sqr(targets - outputs))
        grads = T.grad(loss, self.params.values())

        # # get updates
        updates = []
        for p, g in zip(self.params.values(), grads):
            updates.append((p, p - 0.01 * g))

        # define function
        self.Q_update = theano.function(
            [states, actions, targets],
            loss,
            updates=updates, 
            on_unused_input='warn', 
            allow_input_downcast=True
            )



    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        q = self.Q_eval(state, action)
        return q

    def prepare_data(self, samples):
        """
        Given list of (s, a, r, sp), prepare data for network
        """
        states = []
        actions = []
        targets = []
        for (s, a, r, sp, d) in samples:
            states.append(s)
            actions.append(a)
            targets.append(self.target(r, sp, d))

        return np.asarray(states), np.asarray(actions), np.asarray(targets)

    def target(self, r, sp, d):
        """
        Computes r + discount * max_a (Q(sp, a))
        """
        # compute target
        try:
            v_opt = max(self.evalQ(sp, new_a) for new_a in self.actions)
        except:
            print "error computing evalQ for target"
            v_opt = 0.

        return np.array(r + self.discount * v_opt)


    def _update(self, state, action, reward, newState, done, eligibility=False):
        """
        Update parameters
        0. keep count of calls to update without reloading weights for double Q learning
        1. adds t = (s, a, r, s') to the memory replay
        2. sample t_1, ..., t_n from the memory replay
        3. prepare array of data from samples
        4. update Q
        """

        # 0. update number of calls to evalQ
        self.steps += 1
        if self.steps > self.reload_freq:
            self.reload_params()

        # 1. adds t = (s, a, r, s', done) to the memory replay
        ready = self.update_experience_replay(state, action, reward, newState, done)

        if ready:
            # 2. sample t_1, ..., t_n from the memory replay
            samples = self.sample_experience_replay(self.batch_size)

            # 3. prepare array of data from samples
            states, actions, targets = self.prepare_data(samples)
            
            # 4. update Q
            loss = self.Q_update(states, actions, targets)
            # self.plt_mgr.add("loss", loss)
            # self.plt_mgr.update()



