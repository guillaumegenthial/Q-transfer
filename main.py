import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl

ENV = 'MountainCar-v0'
VERBOSE = False
EXPLORATION_PROBA = 0.2
MAX_ITER = 100
NUM_TRIALS = 1000
RELOAD_WEIGHTS = False
DISCOUNT = 1

if __name__ == "__main__":
    env = gym.make(ENV)

    # 1. train each task separately
    with open("results.txt", "a") as f:
        f.write("#"*20 + "\n")
    for name, param in tasks.SOURCES.iteritems():
        base_rl.train_task(
            env, 
            name, 
            param, 
            NUM_TRIALS, 
            MAX_ITER, 
            VERBOSE, 
            RELOAD_WEIGHTS, 
            DISCOUNT, 
            EXPLORATION_PROBA)

        # env_interaction.plotQ(env, rl.evalQ)
        # env_interaction.play(env, rl.getPolicy())

    # 2. learn combination of tasks for full
    sources = []
    for name, param in tasks.SOURCES.iteritems():
        sources += [base_rl.rl_load(
            name,
            param["file_name"], 
            env, 
            env_interaction.simpleFeatures(env),
            discount=DISCOUNT
            )]


    param = tasks.TARGET["full"]
    file_name = param["file_name"]
    slope = param["slope"]
    reward_modes = param["reward_modes"]
    max_speed = param["max_speed"]
    power = param["power"]

    env.set_task(reward_modes, slope, max_speed, power)

    rl_ens = ensemble_rl.target_train(
        env, 
        "full", 
        sources, 
        num_trials=1, 
        max_iter=1000, 
        filename="weights.p", 
        verbose = False, 
        reload_weights=True, 
        discount=1, 
        explorationProb=0.1
        )

    evaluation = env_interaction.policy_evaluation(
        env, 
        rl_ens.getPolicy(), 
        discount=DISCOUNT)



