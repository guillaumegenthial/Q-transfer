import sys, json
import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl
import deep_rl
import numpy as np

# 0. Get config and arguments for experiment
if len(sys.argv) > 1:
    config = __import__(sys.argv[1].replace(".py", ""))
else:
    config = __import__("config")

EXP_NAME            = config.EXP_NAME
ENV                 = config.ENV
N_SOURCES           = config.N_SOURCES
TARGET_NAME         = config.TARGET_NAME
SOURCE_NAMES        = config.SOURCE_NAMES
VERBOSE             = config.VERBOSE
EXPLORATION_PROBA   = config.EXPLORATION_PROBA
MAX_ITER            = config.MAX_ITER
NUM_TRIALS_SOURCES  = config.NUM_TRIALS_SOURCES
NUM_TRIALS_EVAL     = config.NUM_TRIALS_EVAL
NUM_TRIALS          = config.NUM_TRIALS
RELOAD_WEIGHTS      = config.RELOAD_WEIGHTS
DISCOUNT            = config.DISCOUNT
ELIGIBILITY         = config.ELIGIBILITY
TRAIN               = config.TRAIN
DEEP_MODE           = config.DEEP_MODE

env = gym.make(config.ENV)
fout = open("results/{}.txt".format(EXP_NAME), "wb", 0)

discreteExtractor = env_interaction.discreteExtractor(env)
featureExtractor = env_interaction.simpleFeatures(env)

# 1. train each source task separately
SOURCES, TARGET = tasks.SOURCES, tasks.TARGET

if TRAIN:
    fout.write("# Sources performance\n")
    for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            print("\nTask {}".format(name))
            env.set_task_params(param)
            file_name = "weights/{}_{}.p".format(name, NUM_TRIALS_SOURCES)

            rl = base_rl.SimpleQLearning(
                name=name, 
                actions=range(env.action_space.n), 
                discount=DISCOUNT, 
                discreteExtractor=discreteExtractor, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBA, 
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
            # env_interaction.plotQ(env, rl.evalQ)
            # env_interaction.play(env, rl.getPolicy())

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
            explorationProb=0., 
            weights=file_name, 
            reload_weights=True
        ))

# 3. train Target from sources with different values of num_trials
param = TARGET[TARGET_NAME]
env.set_task_params(param)

for num_trials in NUM_TRIALS:
    fout.write("\n# learning trials: {}\n".format(num_trials))
    print "\n\n{} trials".format(num_trials)

    ########## NEURAL NETWORK IMPLEMENTATION #########
    print "\nDeep transfer"
    file_name = "weights/{}_deep_{}.p".format(TARGET_NAME, num_trials)

    rl_deep = deep_rl.DeepQTransfer(
        name=TARGET_NAME, 
        sources=sources, 
        actions=range(env.action_space.n), 
        discount=DISCOUNT, 
        weights=file_name,
        mode=DEEP_MODE,
        explorationProb=EXPLORATION_PROBA,
        eligibility=ELIGIBILITY,
        reload_weights=RELOAD_WEIGHTS
    )

    training_rewards = rl_deep.train(
        env, 
        num_trials=num_trials, 
        max_iter=MAX_ITER, 
        verbose=VERBOSE
    )

    rl_deep.dump(file_name)

    evaluation, se = env_interaction.policy_evaluation(
        env=env, 
        policy=rl_deep.getPolicy(), 
        discount=DISCOUNT,
        num_trials=NUM_TRIALS_EVAL,
        max_iter=MAX_ITER
    )

    fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(TARGET_NAME+"_target_deep", evaluation, se, np.mean(training_rewards)))


    ########## LINEAR TRANSFER IMPLEMENTATION #########
    print "\nLinear transfer"
    file_name = "weights/{}_coefs_{}.p".format(TARGET_NAME, num_trials)

    rl_ens = ensemble_rl.EnsembleQLearning(
        name=TARGET_NAME, 
        sources=sources, 
        actions=range(env.action_space.n), 
        discount=DISCOUNT,
        weights=file_name,
        explorationProb=EXPLORATION_PROBA,
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

    rl_ens.dump(file_name)

    evaluation, se = env_interaction.policy_evaluation(
        env=env, 
        policy=rl_ens.getPolicy(), 
        discount=DISCOUNT,
        num_trials=NUM_TRIALS_EVAL,
        max_iter=MAX_ITER
    )

    fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(TARGET_NAME+"_target_linear", evaluation, se, np.mean(training_rewards)))

    ########## LEARNING FROM SCRATCH #########
    print "\nLearning from scratch"
    file_name = "weights/{}_direct_{}.p".format(TARGET_NAME, num_trials)

    rl = base_rl.SimpleQLearning(
        name=TARGET_NAME, 
        actions=range(env.action_space.n), 
        discount=DISCOUNT, 
        discreteExtractor=discreteExtractor, 
        featureExtractor=featureExtractor, 
        explorationProb=EXPLORATION_PROBA, 
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

    rl.normalize()
    rl.dump(file_name)

    evaluation, se = env_interaction.policy_evaluation(
        env=env, 
        policy=rl.getPolicy(), 
        discount=DISCOUNT,
        num_trials=NUM_TRIALS_EVAL,
        max_iter=MAX_ITER
    )

    fout.write("\t{}_direct\t{}\t+/-{}\t({} at training time)\n\n".format(TARGET_NAME, evaluation, se, np.mean(training_rewards)))


# save config parameters
fout.write("\n# Config:\n" + "\n".join(
    ["{} = {}".format(k,v) for k,v in config.__dict__.iteritems() if not k.startswith('__')]
))

# save sources and target
fout.write("\n\n# Sources:\n")
sources = {name: param for name, param in SOURCES.items() if name in SOURCE_NAMES}
fout.write(json.dumps(sources, indent=4))

fout.write("\n\n# Target:\n" + json.dumps({TARGET_NAME: TARGET[TARGET_NAME]}, indent=4))

fout.close()


