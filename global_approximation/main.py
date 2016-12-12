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
LR_ENSEMBLE             = config.LR_ENSEMBLE
LR_DEEP                 = config.LR_DEEP

env = gym.make(config.ENV)
fout = open("../results/{}.txt".format(EXP_NAME), "wb", 0)
print "run exp", EXP_NAME

discreteExtractor = env_interaction.discreteExtractor(env)
featureExtractor = env_interaction.simpleFeatures(env)

# 1. train each source task separately
SOURCES, TARGETS = tasks.SOURCES, tasks.TARGETS

if TRAIN:
    fout.write("# Sources performance\n")
    for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            print("\nTask {}".format(name))
            env.set_task_params(param)
            file_name = "../weights/{}_{}.p".format(name, NUM_TRIALS_SOURCES)

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
                num_trials=NUM_TRIALS_SOURCES, 
                max_iter=MAX_ITER, 
                verbose=VERBOSE, 
                eligibility=ELIGIBILITY,
                )

            rl.normalize()
            rl.dump(file_name)

            evaluation, se = env_interaction.policy_evaluation(
                env=env, 
                policy=rl.getPolicy(), 
                discount=DISCOUNT,
                num_trials=NUM_TRIALS_EVAL,
                max_iter=MAX_ITER
            )

            fout.write("\t{}\t{}\t+/-{}\n".format(name, evaluation, se))

# 2. load SimpleQLearning objects in sourcess
sources = []
for name, param in SOURCES.iteritems():
    if name in SOURCE_NAMES:
        file_name = "../weights/{}_{}.p".format(name, NUM_TRIALS_SOURCES)
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
for target_name in TARGET_NAMES:
    fout.write("\n# target: {}\n".format(target_name))
    param = TARGETS[target_name]
    env.set_task_params(param)

    for i, num_trials in enumerate(NUM_TRIALS_TARGETS):
        fout.write("\n# learning trials: {}\n".format(num_trials))
        print "\n\n{} trials".format(num_trials)
        if i == len(NUM_TRIALS_TARGETS)-1:
            training_rewards_list = []
            n_av = AVERAGE_TIMES
        else:
            n_av = 1

        for av in xrange(n_av):

            ########## NEURAL NETWORK IMPLEMENTATION #########
            print "\nDeep transfer"
            for deep_mode in DEEP_MODES:
                name = "{}_deep_{}".format(target_name, deep_mode)
                file_name = "../weights/{}_{}.p".format(name, num_trials)

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
                    reload_weights=RELOAD_WEIGHTS,
                    lr=LR_DEEP
                )

                training_rewards = rl_deep.train(
                    env, 
                    num_trials=num_trials, 
                    max_iter=MAX_ITER, 
                    verbose=VERBOSE
                )
                if i == len(NUM_TRIALS_TARGETS)-1:
                    training_rewards_list.append((name, training_rewards))

                rl_deep.dump(file_name)

                evaluation, se = env_interaction.policy_evaluation(
                    env=env, 
                    policy=rl_deep.getPolicy(), 
                    discount=DISCOUNT,
                    num_trials=NUM_TRIALS_EVAL,
                    # max_iter=MAX_ITER
                )

                fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(name, evaluation, se, np.mean(training_rewards)))


            ########## LINEAR TRANSFER IMPLEMENTATION #########
            print "\nLinear transfer"
            name = "{}_linear".format(target_name)
            file_name = "../weights/{}_{}.p".format(name, num_trials)

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
                lr=LR_ENSEMBLE
            )

            rl_ens.preliminaryCheck(np.array([-0.5, 0]),0)

            training_rewards = rl_ens.train(
                env, 
                num_trials=num_trials, 
                max_iter=MAX_ITER, 
                verbose=VERBOSE
            )

            if i == len(NUM_TRIALS_TARGETS)-1:
                training_rewards_list.append((name, training_rewards))


            rl_ens.dump(file_name)

            evaluation, se = env_interaction.policy_evaluation(
                env=env, 
                policy=rl_ens.getPolicy(), 
                discount=DISCOUNT,
                num_trials=NUM_TRIALS_EVAL,
                # max_iter=MAX_ITER
            )


            fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(name, evaluation, se, np.mean(training_rewards)))

            ########## LEARNING FROM SCRATCH #########
            print "\nLearning from scratch"
            name = "{}_direct".format(target_name)
            file_name = "../weights/{}_{}.p".format(name, num_trials)

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

            if i == len(NUM_TRIALS_TARGETS)-1:
                training_rewards_list.append((name, training_rewards))

            rl.normalize()
            rl.dump(file_name)

            evaluation, se = env_interaction.policy_evaluation(
                env=env, 
                policy=rl.getPolicy(), 
                discount=DISCOUNT,
                num_trials=NUM_TRIALS_EVAL,
                # max_iter=MAX_ITER
            )

            fout.write("\t{}_direct\t{}\t+/-{}\t({} at training time)\n\n".format(name, evaluation, se, np.mean(training_rewards)))

    # utils.plot_rewards(training_rewards_list, file_name="plots/{}_{}_transfer_plots.png".format(EXP_NAME, target_name))

# save config parameters
fout.write("\n# Config:\n" + "\n".join(
    ["{} = {}".format(k,v) for k,v in config.__dict__.iteritems() if not k.startswith('__')]
))

# save sources and target
fout.write("\n\n# Sources:\n")
sources = {name: param for name, param in SOURCES.items() if name in SOURCE_NAMES}
fout.write(json.dumps(sources, indent=4))

fout.write("\n\n# Target:\n")
targets = {name: param for name, param in TARGETS.items() if name in TARGET_NAMES}
fout.write(json.dumps(targets, indent=4))
fout.close()


