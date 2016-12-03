import gym

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

def policy_evaluation(env, policy, discount=1, num_trials=10, max_iter=500):
    """
    Simulate many trials to evaluate `policy`
    """
    r = 0.
    for _ in xrange(num_trials):
        env.reset()
        r += rollout(env, policy, discount, max_iter)
    print("Policy evaluation : {}".format(r/num_trials))
    return r/num_trials

def simpleFeatures(state, action):
    """
    Return a list of (feature-name, value)
    """
    pos, vel = state
    features = [((int(pos * 100), int(vel * 1000), action), 1), (('pos', int(pos * 100), action), 1), (('vel', int(vel * 1000), action), 1), (action, 1)]
    return features
