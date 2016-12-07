import numpy as np
import collections
import theano
import theano.tensor as T
import base_rl
from base_rl import SimpleQLearning

floatX = theano.config.floatX


class DeepQTransfer(SimpleQLearning):
    def __init__(self, name, sources, actions, discount, explorationProb = 0.2, eligibility=0.9):
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

        # weights of network
        self.params = collections.OrderedDict()

    def init(self, size, init="uniform"):
        if init == "uniform":
            return np.random.uniform(low=-0.05, high=0.05, size=size)
        elif init == "ones":
            return 0.05*np.ones(size)

    def parameter_matrix(self, name, size=None, weights=None, init="uniform"):
        if name not in self.params:
            if weights:
                self.params[name] = theano.shared(weights.astype(floatX), name)
            else:
                weights = self.init(size, init)
                self.params[name] = theano.shared(weights.astype(floatX), name)

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

    def fully_connected(self, name, x, x_dim, y_dim, activation=None):
        M = self.parameter_matrix("M_{}".format(name), (x_dim, y_dim))
        z = T.dot(x, M)

        b = self.parameter_matrix("b_{}".format(name), (y_dim,), init="ones")

        if activation:
            return self.activation(activation, z) + b
        else:
            return z

    def add_Q(self):
        """
        Adds 2 functions to the class
            - Q_eval : to evaluate Q value
            - Q_update : to update parameters of Q-transfert
        """
        # inputs
        q_values = T.vector("q_values", floatX)
        state_action = T.vector("state", floatX)

        # compute output Q
        # M_q = self.parameter_matrix("M_q", (self.n_sources,))
        # M_s = self.parameter_matrix("M_s", (3, self.n_sources))
        # output = T.dot(q_values, M_q)

        z1 = self.fully_connected("h1", state_action, 3, self.n_sources*2, "relu")
        alpha = self.fully_connected("alpha", z1, self.n_sources*2, self.n_sources, "relu")

        # q_ = q_values * z2
        # output = self.fully_connected("q", q_, self.n_sources, 1)

        output = T.dot(alpha, q_values).sum()


        # Q eval
        self.Q_eval = theano.function(
            [q_values, state_action], 
            output, 
            on_unused_input='warn', 
            allow_input_downcast=True)

        # target
        target = T.scalar("target", floatX)
        loss = T.mean(T.sqr(target - output))

        # get gradients
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

def target_train(env, name, sources, num_trials=1, max_iter=10, filename="weights.p", verbose=False, reload_weights=True, discount=1, explorationProb=0.1):
    weights = filename if reload_weights else None
    actions = range(env.action_space.n)

    rl_deep = DeepQTransfer(
        name, 
        sources, 
        range(env.action_space.n), 
        discount
        )

    rl_deep.add_Q()

    rl_deep.train(
        env, 
        num_trials=num_trials, 
        max_iter=max_iter, 
        verbose=verbose
        )

    return rl_deep


