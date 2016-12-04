import gym
import numpy as np

def rollout(env, policy, discount=1, max_iter = 500):
    """
    Run a trial with policy
    """
    totalReward = 0
    state = env.reset()
    totalDiscount = 1
    for _ in xrange(max_iter):
        action = policy(state)
        state, reward, done, info = env.step(action)
        totalReward += totalDiscount * reward
        totalDiscount *= discount
        if done:
            break

    return totalReward

# playing
def play(env, policy, max_iter=1000):
    state = env.reset()
    for t in range(max_iter):
        env.render()
        action = policy(state)
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

def policy_evaluation(env, policy, discount=1, num_trials=100, max_iter=500):
    """
    Simulate many trials to evaluate `policy`
    """
    r = 0.
    for _ in xrange(num_trials):
        env.reset()
        r += rollout(env, policy, discount, max_iter)
    print("Policy evaluation : {}".format(r/num_trials))
    return r/num_trials

def discreteFeatures(env):
    """
    Return a function so that f(state, action) = list of (feature-name, value) 
    """
    def f(state, action):
        pos, vel = state
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos_index = int((pos - env.low[0])/r_pos * 1000)
        vel_index = int((vel - env.low[1])/r_vel * 1000)
        features = [((pos_index, vel_index, action), 1)]
        return features
   
    return f

def discreteExtractor(env):
    def f(state, action):
        pos, vel = state
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos_index = int((pos - env.low[0])/r_pos * 1000)
        vel_index = int((vel - env.low[1])/r_vel * 1000)
        features = (pos_index, vel_index, action)
        return features

    def fm1(features):
        pos_index, vel_index, action = features
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos = pos_index/float(1000)*r_pos + env.low[0]
        vel = vel_index/float(1000)*r_vel + env.low[1]
        return np.array([pos, vel]), action

    return f, fm1

def simpleFeatures(env):
    """
    Return a function so that f(state, action) = list of (feature-name, value) 
    """
    def f(state, action):
        pos, vel = state
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos_index = int((pos - env.low[0])/r_pos * 100)
        vel_index = int((vel - env.low[1])/r_vel * 100)
        features = [
            ((pos_index, vel_index), 1), 
            (('pos', pos_index, action), 1), 
            (('vel', vel_index, action), 1), 
            (action, 1)
            ]
        return features
   
    return f

def linearSimpleFeatures(env):
    """
    Return a function so that f(state, action) = list of (feature-name, value) 
    """
    def f(state, action):
        pos, vel = state
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos_index = int((pos - env.low[0])/r_pos * 100)
        vel_index = int((vel - env.low[1])/r_vel * 100)
        features = [
            (('pos_', pos_index, vel_index), pos), 
            (('vel_', pos_index, vel_index), vel), 
            (('pos', pos_index, action), pos), 
            (('vel', vel_index, action), vel), 
            (action, 1)
            ]
        return features
   
    return f


def linearFeatures(env):
    """
    Return a function so that f(state, action) = list of (feature-name, value) 
    """
    def f(state, action):
        pos, vel = state
        r_pos = float(env.high[0] - env.low[0])
        r_vel = float(env.high[1] - env.low[1])
        pos_index = int((pos - env.low[0])/r_pos * 100)
        vel_index = int((vel - env.low[1])/r_vel * 100)
        features = [
            (('1', pos_index, vel_index, action), 1), 
            (('pos', pos_index, vel_index, action), pos), 
            (('vel', pos_index, vel_index, action), vel), 
            ]
        return features
   
    return f

def plotQ(env, evalQ, action=0):
    """
    plotting the Q function for an action
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    n = 50
    pos = env.low[0]
    for i in range(n):
        pos += (env.high[0] - env.low[0])/n
        vel = env.low[1]
        for j in range(n):
            vel += (env.high[1] - env.low[1])/n
            q = evalQ(np.asarray([pos, vel]), action)
            x += [pos]
            y += [vel]
            z += [q]
    ax.scatter(x, y, z)
    ax.set_xlabel('Pos')
    ax.set_ylabel('Speed')
    ax.set_zlabel('Q value')
    plt.show()
