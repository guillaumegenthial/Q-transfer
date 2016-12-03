import gym
import base_rl
import env_interaction
import tasks

ENV = 'MountainCar-v0'
VERBOSE = False
EXPLORATION_PROBA = 0.2
MAX_ITER = 5000
NUM_TRIALS = 100
RELOAD_WEIGHTS = True
DICSCOUNT = 1

if __name__ == "__main__":
    env = gym.make(ENV)
    for name, param in tasks.TASKS.iteritems():
        print("Training task {}".format(name))
        file_name = param["file_name"]
        slope = param["slope"]
        reward_modes = param["reward_modes"]
        max_speed = param["max_speed"]
        power = param["power"]

        env.set_task(reward_modes, slope, max_speed, power)
        policy = base_rl.rl_policy(
            env, 
            env_interaction.simpleFeatures, 
            num_trials=NUM_TRIALS, 
            max_iter=MAX_ITER,
            filename=file_name, 
            verbose=VERBOSE,
            reload_weights=RELOAD_WEIGHTS, 
            discount=DICSCOUNT, 
            explorationProb=EXPLORATION_PROBA)

        # policy = base_rl.load_policy(
        #     FILE_NAME, 
        #     env, 
        #     base_rl.simpleFeatures,
        #     discount=DICSCOUNT)

        env_interaction.play(env, policy)
        env_interaction.policy_evaluation(
            env, 
            policy, 
            discount=DICSCOUNT)