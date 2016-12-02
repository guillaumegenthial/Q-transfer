import gym
import base_rl

ENV = 'MountainCar-v0'
FILE_NAME = "weights.p"
DICSCOUNT = 1

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
def play(env, policy, max_iter=500):
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

if __name__ == "__main__":
    env = gym.make(ENV)
    policy = base_rl.rl_policy(
        env, 
        base_rl.simpleFeatures, 
        num_trials=1000, 
        max_iter=5000,
        filename=FILE_NAME, 
        verbose=False,
        reload_weights=True, 
        discount=DICSCOUNT, 
        explorationProb=0.1)

    policy = base_rl.load_policy(
        FILE_NAME, 
        env, 
        base_rl.simpleFeatures,
        discount=DICSCOUNT)

    play(env, policy)
    policy_evaluation(
        env, 
        policy, 
        discount=DICSCOUNT)