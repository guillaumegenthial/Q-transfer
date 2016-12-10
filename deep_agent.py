import gym
import numpy as np
import theano
import theano.tensor as T
import sklearn.preprocessing
import collections
import env_interaction

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

floatX = theano.config.floatX


class ValueFunctionApproximator(object):
    def __init__(self, env, batch_size, learning_rate, transform=False):
        self.nA = env.action_space.n
        self.sS = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.lr = theano.shared(np.float32(learning_rate))
        self.env = env
        self.input_size = 300 if transform else len(env.observation_space.sample())
        self.params = collections.OrderedDict()
        self.transform = transform

        if transform:
            # Fit feature scaler
            observation_examples = np.array([env.observation_space.sample() for x in range(100000)])
            self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler.fit(observation_examples)
            # Fit feature extractor
            self.feature_map = FeatureUnion([("rbf1", RBFSampler(n_components=100, gamma=1., random_state=1)),
                                             ("rbf01", RBFSampler(n_components=100, gamma=0.1, random_state=1)),
                                             ("rbf10", RBFSampler(n_components=100, gamma=10, random_state=1))])
            self.feature_map.fit(self.scaler.transform(observation_examples))

        self._init_model()


    def _init_model(self):
        nn_x, nn_z = T.matrices('x', 'z')

        # theano implementation
        nn_lh2 = self.fully_connected("nn_lh2", nn_x, self.input_size, 512, "relu")
        nn_lh3 = self.fully_connected("nn_lh3", nn_lh2, 512, 256, "relu")
        nn_y = self.fully_connected("nn_y", nn_lh3, 256, self.nA,  None)

        self.f_predict = theano.function([nn_x], nn_y)

        nn_cost = T.sum(T.sqr(nn_y - nn_z))

        nn_updates = self.rmsprop(nn_cost)

        self.f_train = theano.function([nn_x, nn_z],
                                       [nn_y, nn_cost],
                                       updates=nn_updates)

    def rmsprop(self, cost):
        grads = T.grad(cost, self.params.values())
        nn_updates = []    

        self.accu = collections.OrderedDict()
        for name, shared_variable in self.params.iteritems():
            w = shared_variable.get_value(borrow=True)
            w = np.zeros_like(w)
            self.accu[name] = theano.shared(w.astype(floatX), name)

        for p, g, acc, in zip(self.params.values(), grads, self.accu.values()):
            new_acc = 0.9*acc + 0.1*g**2
            nn_updates.append((acc, new_acc))
            nn_updates.append((p, p - self.lr * g / T.sqrt(new_acc + 1e-6)))

        return nn_updates


    def dump(self, file_name):
        import pickle
        d = {"params": self.params, "scaler": self.scaler, "feature_map": self.feature_map}
        with open(file_name, "wb") as fout:
            pickle.dump(d, fout)
        print("Dumped value function weights in {}".format(file_name))

    def load(self, file_name):
        import pickle
        with open(file_name, "rb") as fin:
            d = pickle.load(fin)
        self.params = d["params"]
        self.scaler = d["scaler"]
        self.feature_map = d["feature_map"]

        print("Loaded value function weights from {}".format(file_name))

    def _scale_state(self, s_float32):
        return self.scaler.transform(s_float32)

    def _process_state(self, s_float32):
        if len(s_float32.shape) == 1:
            s_float32 = np.expand_dims(s_float32, axis=0)
        if len(s_float32.shape) != 2:
            raise RuntimeError('Input should be an 2d-array or row-vector.')
        if self.transform:
            s_float32 = self._scale_state(s_float32)
            s_float32 = self.feature_map.transform(s_float32)
            s_float32 = s_float32.astype(np.float32)
        return s_float32

    def predict(self, s):
        s_float32 = np.array(s)
        s_float32 = self._process_state(s_float32)
        return self.f_predict(s_float32)

    def train(self, states, actions, rewards):
        s_float32 = np.array(states).astype(np.float32)
        s_float32 = self._process_state(states)
        a_float32 = np.array(actions).astype(np.float32)
        result = self.f_train(s_float32, a_float32)
        return result

    def init(self, size, init="uniform"):
        if init == "uniform":
            return np.random.uniform(low=-0.5, high=0.5, size=size)
        elif init == "ones":
            return 0.05*np.ones(size)
        elif init == "glorot":
            return self.glorot_normal(size)
        elif init == "zeros":
            return np.zeros(size)
        else:
            print "unknown initialization method"
            raise

    def parameter_matrix(self, name, size=None, weights=None, init="uniform"):
        print "Initializing {} with shape {}".format(name, size)
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
            return T.nnet.relu(x, 0.01)
        else:
            print("ERROR, {} is an unknown activation".format(activation))
            raise

    def glorot_normal(self, size, dim_ordering='th'):
        fan_in, fan_out = size[0], size[1]
        s = np.sqrt(2. / (fan_in + fan_out))
        return self.normal(size, s)

    def normal(self, size, scale=0.05):
        if scale is None:
            scale = 0.05
        return np.random.normal(loc=0.0, scale=scale, size=size)

    def fully_connected(self, name, x, x_dim, y_dim, activation=None, offset=True, init="glorot"):
        M = self.parameter_matrix("M_{}".format(name), (x_dim, y_dim), init=init)
        z = T.dot(x, M)

        if offset:
            b = self.parameter_matrix("b_{}".format(name), (y_dim,), init="zeros")
            z += b
        if activation:
            return self.activation(activation, z)
        else:
            return z
    

class ReplayMemory(object):
    def __init__(self, agent, capacity):
        self.agent = agent
        self.capacity = capacity
        self.memory = []

    def append(self, s, a, r, sp):
        self.memory.append([s, a, r, sp])

        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size, discount=1.0):
        batch_size = min(batch_size, len(self.memory))
        choices = np.random.choice(len(self.memory), batch_size)
        s = np.array([self.memory[i][0] for i in choices])
        a = np.array([self.memory[i][1] for i in choices])
        r = np.array([self.memory[i][2] for i in choices])
        sp = np.array([self.memory[i][3] for i in choices])

        q_vals = self.agent.q_values(s)
        target = r + (r <= 0).astype(int) * discount * np.amax(self.agent.q_values(sp), axis=1)
        for i in range(len(choices)):
            q_vals[i, a[i]] = target[i]

        return s, q_vals


class DeepAgent(object):
    def __init__(self, env, eps=1.0, learning_rate=0.1, transform=False, value_function=None):
        self.env = env
        self.nA = env.action_space.n
        self.eps = eps
        if value_function:
            self.value_function = value_function
        else:
            self.value_function = ValueFunctionApproximator(env, 32, learning_rate, transform)
        self.memory = ReplayMemory(self, 100000)
        self.discount = 1.0

    def q_values(self, s):
        return self.value_function.predict(s)

    def act(self, s):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.value_function.predict(s))

    def estimate(self, s, a):
        prediction = self.value_function.predict(s)[0]
        return prediction[a]

    def learn(self, s, targets):
        self.value_function.train(s, targets, [])

    def dump(self, file_name):
        self.value_function.dump(file_name)

    def load(self, file_name):
        self.value_function.load(file_name)

    def get_policy(self):
        return lambda s : np.argmax(self.value_function.predict(s))

    def train(self, n_episodes=100, max_steps_per_episode=20000):
        env = self.env
        discount = self.discount
        memory = self.memory

        for episode in range(n_episodes):
            steps = 0
            s = self.env.reset()
            done = False
            while not done:
                a = self.act(s)
                q_vals = self.q_values(s)

                sp, r, done, info = env.step(a)
                memory.append(s, a, r, sp)

                if len(memory.memory) > 128:
                    mem_states, mem_targets = memory.sample(64, discount)
                    mem_states = np.array(mem_states)
                    mem_targets = np.array(mem_targets)
                    self.learn(mem_states, mem_targets)

                if steps % 50 == 0:
                    print('Episode {}, Step {}, eps {}'.format(episode, steps, self.eps))

                if done or steps > max_steps_per_episode:
                    print("Episode finished after {} timesteps".format(steps))
                    print()
                    break
                s = sp
                steps += 1

                if self.eps >= 0.01 and steps % 10000 == 0:
                    self.eps *= 0.9

            if self.eps >= 0.0:
                self.eps *= 0.9

if file.__name__ == "__main__":
    env_name = 'MountainCar-v0'
    file_name = "deep_agent_transform.p"
    env = gym.make(env_name)
    agent = DeepAgent(env, eps=0.2, learning_rate=0.0001, transform=True)
    agent.load(file_name)
    agent.train(n_episodes=100, max_steps_per_episode=20000)
    agent.dump(file_name)
    evaluation, se = env_interaction.policy_evaluation(
                    env=env, 
                    policy=agent.get_policy())

    print evaluation
