import gym
import numpy as np
import pickle

# evaluation
def rollout(env, pol, gamma, max_t=100):
    reward_total = 0
    observation = env.reset()
    for t in range(max_t):
        action = pol(observation)
        observation, reward, done, info = env.step(action)
        reward_total = reward + gamma*reward_total
        if done:
            break

    return reward_total

def policy_evaluation(env, pol, gamma=1, n_episode=1000, max_t=100):
    r = 0
    for i_episode in xrange(n_episode):
        env.reset()
        r += rollout(env, pol, gamma, max_t)

    return r/n_episode

# global approximation
def beta(env, observation, N):
    # returns indices of discretized states
    features = []
    for i, obs in enumerate(observation):
        radius = (env.high[i] - env.low[i])
        features += [int((obs-env.low[i])/radius*N)]
    return features

def explore(env, observation, N, theta level=0.2):
    # must return an action
    eps = np.random.random()
    if eps < level:
        return env.action_space.sample()
    else:
        return np.argmax(Q(env, observation, N, theta))

def Q(env, observation, N, theta):
    # must return a vector q[action] = Q(observation, action)
    b = beta(env, observation, N)
    q = theta
    for i in b:
        q = q[i]
    return q

def global_approximation(env, alpha=0.1, gamma=1, N=10, theta=None):
    # get number of features
    observation = env.reset()
    action = env.action_space.sample()
    b = beta(env, observation, N)
    # theta np array(a, N, N) (action, pos, speed)
    if theta is None:
        theta = np.random.random( [env.action_space.n]+[N]*len(b))
    for i_episode in range(100):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = explore(env, beta, theta, observation)
            _observation, reward, done, info = env.step(action)
            if t > 0:
                r = reward + gamma*np.max(np.dot(theta, beta(_observation))) - np.dot(theta, beta(observation))[action]
                theta[action] = theta[action] + alpha*r*beta(observation)
            observation = _observation
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    return theta


# playing
def play(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = 0
            observation, reward, done, info = env.step(action)
            print beta(env, observation, 10)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

def plot(env, theta, beta):
    # plotting the Q function just to see
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    n = 10
    pos = env.min_position
    for i in range(n):
        pos += (env.max_position - env.min_position)/n
        vel = -env.max_speed
        for i in range(n):
            vel += 2*env.max_speed/n
            q = np.dot(theta, beta(np.asarray([pos, vel])))[1]
            x += [pos]
            y += [vel]
            z += [q]
    ax.scatter(x, y, z)
    ax.set_xlabel('Pos')
    ax.set_ylabel('Speed')
    ax.set_zlabel('Q value')
    plt.show()

# env
ENV = 'MountainCar-v0'
env = gym.make(ENV)
env.set_mode(0)
gamma = 1
# test policy
# pol = lambda observation: 2
# with open("theta.pkl") as f:
#     theta = pickle.load(f)
theta = global_approximation(env)
# with open("theta.pkl", "w") as f:
#     pickle.dump(theta, f)
# plot(env, theta, beta)
# pol = lambda observation: np.argmax(np.dot(theta, beta(observation)))
# evaluate policy
# print policy_evaluation(env, pol)

