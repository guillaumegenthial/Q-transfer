import gym
import base_rl

ENV = 'MountainCar-v0'


def rollout(env, policy, discount, max_iter = 100):
    """
    Run a trial with policy
    """
    reward_total = 0
    state = env.reset()
    for _ in xrange(max_iter):
        action = policy(state)
        state, reward, done, info = env.step(action)
        reward_total = reward + discount * reward_total
        if done:
            break

    return reward_total

def policy_evaluation(env, policy, discount = 0.99, num_trials = 1000, max_iter = 100):
    """
    Simulate many trials to evaluate `policy`
    """
    r = 0.
    for _ in xrange(num_trials):
        env.reset()
        r += rollout(env, policy, discount, max_iter) / num_trials

    return r

if __name__ == "main":
    env = gym.make(ENV)
    policy = base_rl.rl_policy(env, base_rl.simpleFeatures)
    print policy_evaluation(env, policy)