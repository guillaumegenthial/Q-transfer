import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl
import deep_rl

ENV = 'MountainCar-v0'
N_SOURCES = 10
TARGET_NAME = "full"
VERBOSE = False
EXPLORATION_PROBA = 0.2
MAX_ITER = 1000
NUM_TRIALS = 300
RELOAD_WEIGHTS = False
DISCOUNT = 1
ELIGIBILITY = False
TRAIN = False

if __name__ == "__main__":
    env = gym.make(ENV)
    with open("results.txt", "r") as f:
        run_no = 1
        for line in f:
            if line[0] == "#":
                run_no += 1

    with open("results.txt", "a") as f:
        f.write("#"*10 + " run {} ".format(run_no) + "#"*10 + "\n")

    # 1. train each source task separately
    # SOURCES = tasks.SOURCES
    # TARGET = tasks.TARGET
    # SOURCES, TARGET = tasks.generate_tasks(N_SOURCES)
    SOURCES, TARGET = tasks.SOURCES, tasks.TARGET
    
    if TRAIN:
        for name, param in SOURCES.iteritems():
            base_rl.train_task(
                discreteExtractor=env_interaction.discreteExtractor(env), 
                featureExtractor=env_interaction.simpleFeatures(env), 
                env=env, 
                name=name, 
                param=param, 
                num_trials=NUM_TRIALS, 
                max_iter=MAX_ITER, 
                verbose=VERBOSE, 
                reload_weights=RELOAD_WEIGHTS, 
                discount=DISCOUNT, 
                explorationProb=EXPLORATION_PROBA,
                eligibility=ELIGIBILITY)

            # env_interaction.plotQ(env, rl.evalQ)
            # env_interaction.play(env, rl.getPolicy())

    # 2. learn combination of tasks for full
    sources = []
    for name, param in SOURCES.iteritems():
        sources += [base_rl.rl_load(
            name=name,
            discreteExtractor=env_interaction.discreteExtractor(env), 
            featureExtractor=env_interaction.simpleFeatures(env),
            filename=param["file_name"], 
            env=env, 
            discount=DISCOUNT
            )]

    param = TARGET[TARGET_NAME]
    file_name = param["file_name"]
    slope = param["slope"]
    reward_modes = param["reward_modes"]
    max_speed = param["max_speed"]
    power = param["power"]

    env.set_task(reward_modes, slope, max_speed, power)


    ##################################################
    ########## NEURAL NETWORK IMPLEMENTATION #########

    rl_deep = deep_rl.target_train(
        env, 
        TARGET_NAME, 
        sources, 
        num_trials=NUM_TRIALS, 
        max_iter=MAX_ITER, 
        filename="weights.p", 
        verbose = VERBOSE, 
        reload_weights=RELOAD_WEIGHTS, 
        discount=DISCOUNT, 
        explorationProb=EXPLORATION_PROBA
        )

    evaluation = env_interaction.policy_evaluation(
        env=env, 
        policy=rl_deep.getPolicy(), 
        discount=DISCOUNT)

    with open("results.txt", "a") as f:
        f.write("{} {}\n".format(TARGET_NAME+"_target_deep", evaluation))


    ##################################################
    rl_ens = ensemble_rl.target_train(
        env, 
        TARGET_NAME, 
        sources, 
        num_trials=NUM_TRIALS, 
        max_iter=MAX_ITER, 
        filename="weights.p", 
        verbose = VERBOSE, 
        reload_weights=RELOAD_WEIGHTS, 
        discount=DISCOUNT, 
        explorationProb=EXPLORATION_PROBA
        )

    evaluation = env_interaction.policy_evaluation(
        env=env, 
        policy=rl_ens.getPolicy(), 
        discount=DISCOUNT)

    with open("results.txt", "a") as f:
        f.write("{} {}\n".format(TARGET_NAME+"_target", evaluation))

    # 3. compare with direct learning
    base_rl.train_task(
        discreteExtractor=env_interaction.discreteExtractor(env), 
        featureExtractor=env_interaction.simpleFeatures(env), 
        env=env, 
        name=TARGET_NAME+"_direct", 
        param=param, 
        num_trials=NUM_TRIALS, 
        max_iter=MAX_ITER, 
        verbose=VERBOSE, 
        reload_weights=RELOAD_WEIGHTS, 
        discount=DISCOUNT, 
        explorationProb=EXPLORATION_PROBA,
        eligibility=False)

    with open("results.txt", "a") as f:
        f.write("\n")


