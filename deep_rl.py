import numpy as np
import collections
import pickle
import copy
import theano
import theano.tensor as T
import base_rl
from base_rl import SimpleQLearning

floatX = theano.config.floatX


class DeepQTransfer(SimpleQLearning):
    def __init__(self, name, sources, actions, discount, weights=None, explorationProb = 0.2, eligibility=0.9, reload_freq=10, mode=0):
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
        self.eligibility = eligibility
        # handle update from time to time for params for eval
        self.reload_freq = reload_freq
        # counts the number of calls to evalQ without reloading params
        self.steps = 0
        # set a choice for the way we compute ouput
        self.mode = mode

        if weights:
            self.load(weights)
        else:
            # weights of network
            self.params = collections.OrderedDict()
            # copy updated from time to time
            self.params_bak = collections.OrderedDict()

    def load(self, file_name):
        """
        Load weights from pickle file dict into the default dict
        """
        with open(file_name, "rb") as fin:
            print("Loading params from file {}".format(file_name))
            self.params = pickle.load(fin)
            self.params_bak = copy.deepcopy(self.params)

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
            return np.random.uniform(low=0, high=0.0005, size=size)
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

    def output(self, q_values, state_action, bak=False):
        """
        Computes theano expression for output = f(q_values, state_action)
        """
        if self.mode == 0:
            # simple linear combination like ensemble_rl
            output = self.fully_connected("alpha", q_values, self.n_sources, 1, bak, None, False).sum()

        elif self.mode == 1:
            # computes coefficients as functions of state_action
            h1 = self.fully_connected("h1", state_action, 3, self.n_sources*2, bak, "sigmoid")
            h2 = self.fully_connected("h2", h1, self.n_sources*2, self.n_sources*2, bak, "sigmoid")
            alpha = self.fully_connected("alpha", h2, self.n_sources*2, self.n_sources, bak, "sigmoid")

            output = (q_values*alpha).sum()

        elif self.mode == 2:
            # computes attention with coefficient as for mode 1
            # computes coefficients as functions of state_action
            h1 = self.fully_connected("h1", state_action, 3, self.n_sources*2, bak, "sigmoid")
            h2 = self.fully_connected("h2", h1, self.n_sources*2, self.n_sources*2, bak, "sigmoid")
            alpha = self.fully_connected("alpha", h2, self.n_sources*2, self.n_sources, bak, "sigmoid")
            # weight each input q value
            q_values_pond = q_values * alpha
            h3 = self.fully_connected("h3", q_values_pond, self.n_sources, self.n_sources*2, bak, "sigmoid")
            output = self.fully_connected("out", h3, self.n_sources*2, 1, bak, "sigmoid").sum()

        else:
            print("ERROR mode {} is unknown".format(self.mode))

        return output

    def add_Q(self):
        """
        Adds 2 functions to the class
            - Q_eval : to evaluate Q value
            - Q_update : to update parameters of Q-transfert
        """
        # inputs
        q_values = T.vector("q_values", floatX)
        state_action = T.vector("state", floatX)
        target = T.scalar("target", floatX)

        # compute output with bak params
        output_eval = self.output(q_values, state_action, True)

        # 1. Q eval
        self.Q_eval = theano.function(
            [q_values, state_action], 
            output_eval, 
            on_unused_input='warn', 
            allow_input_downcast=True)

        # 2. Q_update 
        # compute output with regular params
        output = self.output(q_values, state_action, False)
        # get gradients
        loss = T.mean(T.sqr(target - output))
        grads = T.grad(loss, self.params.values())

        # get updates
        updates = []
        for p, g in zip(self.params.values(), grads):
            updates.append((p, p - 0.000001 * g))

        # define function
        self.Q_update = theano.function(
            [q_values, state_action, target],
            output,
            updates=updates, 
            on_unused_input='warn', 
            allow_input_downcast=True
            )

    def process_data(self, state, action):
        q_values = np.zeros(self.n_sources, floatX)
        for i in xrange(self.n_sources):
            q_values[i] = self.sources[i].evalQ(state, action)

        return q_values, np.append(state, action)

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        # update number of calls to evalQ
        self.steps += 1
        if self.steps > self.reload_freq:
            self.reload_params()
        # prepare data as np arrays
        q_values, state_action = self.process_data(state, action)
        # call Q_eval
        return self.Q_eval(q_values, state_action)

    def _update(self, state, action, reward, newState, eligibility=False):
        """
        Update parameters for a transition
        """
        # compute target
        # TODO : use old params for evalQ in target
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions)
        except:
            print "error"
            v_opt = 0.

        target = np.array(reward + self.discount * v_opt)

        # compute input data
        q_values, state_action = self.process_data(state, action)

        # update
        self.Q_update(q_values, state_action, target)
        # M_q = self.params["M_q"].get_value()
        # print M_q / M_q.sum()

####################################################

def target_train(env, name, sources, num_trials=1, max_iter=10, filename="weights.p", verbose=False, reload_weights=True, discount=1, explorationProb=0.1, eligibility=False, mode=0):
    filename = "weights/"+filename
    weights = filename if reload_weights else None
    actions = range(env.action_space.n)

    rl_deep = DeepQTransfer(
        name=name, 
        sources=sources, 
        actions=range(env.action_space.n), 
        discount=discount, 
        weights=weights,
        mode=mode,
        explorationProb=explorationProb,
        eligibility=eligibility
    )

    rl_deep.add_Q()

    rl_deep.train(
        env, 
        num_trials=num_trials, 
        max_iter=max_iter, 
        verbose=verbose
    )

    rl_deep.dump(filename)

    return rl_deep


