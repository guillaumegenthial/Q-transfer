import sys, json
import gym
import base_rl
import base_deep_rl
import env_interaction
import tasks
import ensemble_rl
import deep_rl
import numpy as np
import utils

# 0. Get config and arguments for experiment
if len(sys.argv) > 1:
    config = __import__(sys.argv[1].replace(".py", ""))
else:
    config = __import__("config")

EXP_NAME                = config.EXP_NAME
ENV                     = config.ENV
TARGET_NAMES            = config.TARGET_NAMES
SOURCE_NAMES            = config.SOURCE_NAMES
VERBOSE                 = config.VERBOSE
EXPLORATION_PROBA_START = config.EXPLORATION_PROBA_START
EXPLORATION_PROBA_END   = config.EXPLORATION_PROBA_END
MAX_ITER                = config.MAX_ITER
NUM_TRIALS_SOURCES      = config.NUM_TRIALS_SOURCES
NUM_TRIALS_TARGETS      = config.NUM_TRIALS_TARGETS
NUM_TRIALS_EVAL         = config.NUM_TRIALS_EVAL
RELOAD_WEIGHTS          = config.RELOAD_WEIGHTS
DISCOUNT                = config.DISCOUNT
ELIGIBILITY             = config.ELIGIBILITY
TRAIN                   = config.TRAIN
DEEP_MODES              = config.DEEP_MODES


env = gym.make(config.ENV)
fout = open("results/{}_deep.txt".format(EXP_NAME), "wb", 0)

# 1. train each source task separately
SOURCES, TARGETS = tasks.SOURCES, tasks.TARGETS

if TRAIN:
    fout.write("# Sources performance\n")
    for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            print("\nTask {}".format(name))
            env.set_task_params(param)
            file_name = "weights/{}_{}.p".format(name, NUM_TRIALS_SOURCES)

            rl = base_deep_rl.DeepQLearning(
            name, 
            range(env.action_space.n), 
            DISCOUNT, 
            exploration_start=.5,
            exploration_end=0.1, 
            weights=None, 
            eligibility=0.9, 
            reload_freq=20,
            experience_replay_size=10000)

            training_rewards = rl.train(
                env=env, 
                num_trials=NUM_TRIALS_SOURCES, 
                max_iter=MAX_ITER, 
                verbose=VERBOSE, 
                eligibility=ELIGIBILITY,
                )

            rl.dump(file_name)

            evaluation, se = env_interaction.policy_evaluation(
                env=env, 
                policy=rl.getPolicy(), 
                discount=DISCOUNT,
                num_trials=NUM_TRIALS_EVAL,
                max_iter=MAX_ITER
            )

            fout.write("\t{}\t{}\t+/-{}\n".format(name, evaluation, se))
