import sys, json
import gym
import base_rl
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
AVERAGE_TIMES           = config.AVERAGE_TIMES

env = gym.make(config.ENV)

discreteExtractor = env_interaction.discreteExtractor(env)
featureExtractor = env_interaction.simpleFeatures(env)

# 1. train each source task separately
SOURCES, TARGETS = tasks.SOURCES, tasks.TARGETS

# 2. load SimpleQLearning objects in sourcess
sources = []
for name, param in SOURCES.iteritems():
    if name in SOURCE_NAMES:
        file_name = "weights/{}_{}.p".format(name, NUM_TRIALS_SOURCES)
        sources.append(base_rl.SimpleQLearning(
            name=name, 
            actions=range(env.action_space.n), 
            discount=DISCOUNT, 
            discreteExtractor=discreteExtractor, 
            featureExtractor=featureExtractor, 
            exploration_start=0.,
            exploration_end=0., 
            weights=file_name, 
            reload_weights=True
        ))

# 3. train Targets from sources with different values of num_trials
num_trials = 10
n = 2
for target_name in TARGET_NAMES:
    param = TARGETS[target_name]
    env.set_task_params(param)

    print "\n\n{} trials".format(num_trials)


    training_rewards_all = {}

    for av in xrange(n):
        ########## NEURAL NETWORK IMPLEMENTATION #########
        print "\nDeep transfer"
        for deep_mode in DEEP_MODES:
            name = "{}_deep_{}".format(target_name, deep_mode)
            file_name = "weights/{}_{}.p".format(name, num_trials)

            rl_deep = deep_rl.DeepQTransfer(
                name=name, 
                sources=sources, 
                actions=range(env.action_space.n), 
                discount=DISCOUNT, 
                weights=file_name,
                mode=deep_mode,
                exploration_start=EXPLORATION_PROBA_START,
                exploration_end=EXPLORATION_PROBA_END, 
                eligibility=ELIGIBILITY,
                reload_weights=RELOAD_WEIGHTS
            )

            training_rewards = rl_deep.train(
                env, 
                num_trials=num_trials, 
                max_iter=MAX_ITER, 
                verbose=VERBOSE
            )
            
            training_rewards = np.array(training_rewards)

            if av == 0:
                training_rewards_all[name] = training_rewards
            else:
                training_rewards_all[name] += training_rewards

        ########## LINEAR TRANSFER IMPLEMENTATION #########
        print "\nLinear transfer"
        name = "{}_linear".format(target_name)
        file_name = "weights/{}_{}.p".format(name, num_trials)

        rl_ens = ensemble_rl.EnsembleQLearning(
            name=name, 
            sources=sources, 
            actions=range(env.action_space.n), 
            discount=DISCOUNT,
            weights=file_name,
            exploration_start=EXPLORATION_PROBA_START,
            exploration_end=EXPLORATION_PROBA_END, 
            eligibility=ELIGIBILITY,
            reload_weights=RELOAD_WEIGHTS, 
        )

        rl_ens.preliminaryCheck(np.array([-0.5, 0]),0)

        training_rewards = rl_ens.train(
            env, 
            num_trials=num_trials, 
            max_iter=MAX_ITER, 
            verbose=VERBOSE
        )

        training_rewards = np.array(training_rewards)

        if av == 0:
            training_rewards_all[name] = training_rewards
        else:
            training_rewards_all[name] += training_rewards

        ########## LEARNING FROM SCRATCH #########
        print "\nLearning from scratch"
        name = "{}_direct".format(target_name)
        file_name = "weights/{}_{}.p".format(name, num_trials)

        rl = base_rl.SimpleQLearning(
            name=name, 
            actions=range(env.action_space.n), 
            discount=DISCOUNT, 
            discreteExtractor=discreteExtractor, 
            featureExtractor=featureExtractor, 
            exploration_start=EXPLORATION_PROBA_START,
            exploration_end=EXPLORATION_PROBA_END, 
            weights=file_name, 
            reload_weights=RELOAD_WEIGHTS
            )

        training_rewards = rl.train(
            env=env, 
            num_trials=num_trials, 
            max_iter=MAX_ITER, 
            verbose=VERBOSE, 
            eligibility=ELIGIBILITY,
            )

        training_rewards = np.array(training_rewards)

        if av == 0:
            training_rewards_all[name] = training_rewards
        else:
            training_rewards_all[name] += training_rewards


    training_rewards_list = [(name, training_rewards*1/float(n)) for name, training_rewards in training_rewards_all.iteritems()]
    utils.plot_rewards(training_rewards_list, file_name="plots/{}_{}_transfer_plots_averaged.png".format(EXP_NAME, target_name))


