import gym
import base_rl
import env_interaction
import tasks
import time

ENV = 'MountainCar-v0'
VERBOSE = False
EXPLORATION_PROBA = 0.2
MAX_ITER = 5000
NUM_TRIALS = 20
RELOAD_WEIGHTS = False
DISCOUNT = 1

if __name__ == "__main__":
    env = gym.make(ENV)
    with open("results.txt", "a") as f:
        f.write("#"*10 + "\n")
    for name, param in tasks.TASKS.iteritems():
        print("Task {}".format(name))
        file_name = param["file_name"]
        slope = param["slope"]
        reward_modes = param["reward_modes"]
        max_speed = param["max_speed"]
        power = param["power"]

        env.set_task(reward_modes, slope, max_speed, power)

        rl = base_rl.rl_train(
            name, 
            env, 
            env_interaction.simpleFeatures(env), 
            num_trials=NUM_TRIALS, 
            max_iter=MAX_ITER,
            filename=file_name, 
            verbose=VERBOSE,
            reload_weights=RELOAD_WEIGHTS, 
            discount=DISCOUNT, 
            explorationProb=EXPLORATION_PROBA)

        # rl = base_rl.rl_load(
        #     name,
        #     file_name, 
        #     env, 
        #     env_interaction.simpleFeatures(env),
        #     discount=DISCOUNT)

        # env_interaction.plotQ(env, rl.evalQ)
        # env_interaction.play(env, rl.getPolicy())

        evaluation = env_interaction.policy_evaluation(
            env, 
            rl.getPolicy(), 
            discount=DISCOUNT)

        with open("results.txt", "a") as f:
            f.write("{} {}\n".format(name, evaluation))