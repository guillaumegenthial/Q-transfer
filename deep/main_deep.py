import sys, json
import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl
import deep_rl
import numpy as np
import utils
from deep_agent import ValueFunctionApproximator, DeepAgent, deepTransferAgent, Extractor

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
AVERAGE_TIMES           = config.AVERAGE_TIMES

env = gym.make(config.ENV)
fout = open("../results/{}_deep.txt".format(EXP_NAME), "wb", 0)
print "run exp", EXP_NAME

# 1. train each source task separately
SOURCES, TARGETS = tasks.SOURCES, tasks.TARGETS
LEARNING_RATE = 0.0001
TRANSFORM = True
MAX_TIME = 1800

# 1. train sources agents
if TRAIN:
    # use the same value function for every deep agent
    extractor = Extractor(env)
    fout.write("# Sources performance\n")
    for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            print("\nTask {}".format(name))
            env.set_task_params(param)
            file_name = "../weights/{}_{}_deep_transform.p".format(name, NUM_TRIALS_SOURCES)

            agent = DeepAgent(env, eps=0.5, learning_rate=0.0001, transform=True,
                extractor=extractor)

            agent.train(n_episodes=NUM_TRIALS_SOURCES, 
                max_steps_per_episode=20000, max_time=MAX_TIME)

            agent.dump(file_name)

            evaluation, se = env_interaction.policy_evaluation(
                            env=env, 
                            policy=agent.get_policy(),
                            discount=DISCOUNT,
                            num_trials=NUM_TRIALS_EVAL,
                            # max_iter=MAX_ITER
                            )


            fout.write("\t{}\t{}\t+/-{}\n".format(name, evaluation, se))

# 2. load sources agents
sources = []
for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            file_name = "../weights/{}_{}_deep_transform.p".format(name, NUM_TRIALS_SOURCES)
            agent = DeepAgent(env, eps=0.5, learning_rate=0.0001, transform=True)

            agent.load(file_name)

            sources.append(agent)


# 3. train Targets from sources with different values of num_trials
for target_name in TARGET_NAMES:
    fout.write("\n# target: {}\n".format(target_name))
    param = TARGETS[target_name]
    env.set_task_params(param)

    file_name = "../weights/{}_deep_transfer_agent.p".format(target_name)
    deep_transfer_agent = deepTransferAgent(env, sources, eps=0.1, 
        learning_rate=0.0001, transform=True)

    deep_transfer_agent.train(n_episodes=NUM_TRIALS_SOURCES, 
                    max_steps_per_episode=20000, max_time=MAX_TIME)

    # deep_transfer_agent.dump(file_name)

    evaluation, se = env_interaction.policy_evaluation(
                                env=env, 
                                policy=deep_transfer_agent.get_policy(),
                                discount=DISCOUNT,
                                num_trials=NUM_TRIALS_EVAL,
                                # max_iter=MAX_ITER
                                )


    fout.write("\t{}\t{}\t+/-{}\n".format("deep transfert agent", evaluation, se))


