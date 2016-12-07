import sys, json
import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl
import deep_rl



if __name__ == "__main__":
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
    fout = open("results/{}.txt".format(EXP_NAME), "wb")

    # 1. train each source task separately
    # SOURCES = tasks.SOURCES
    # TARGET = tasks.TARGET
    # SOURCES, TARGET = tasks.generate_tasks(N_SOURCES)
    SOURCES, TARGET = tasks.SOURCES, tasks.TARGET
    
    if TRAIN:
        for name, param in SOURCES.iteritems():
            if name in SOURCE_NAMES:
                evaluation, se, _ = base_rl.train_task(
                    discreteExtractor=env_interaction.discreteExtractor(env), 
                    featureExtractor=env_interaction.simpleFeatures(env), 
                    env=env, 
                    name=name, 
                    param=param, 
                    num_trials=NUM_TRIALS_SOURCES, 
                    max_iter=MAX_ITER, 
                    verbose=VERBOSE, 
                    reload_weights=RELOAD_WEIGHTS, 
                    discount=DISCOUNT, 
                    explorationProb=EXPLORATION_PROBA,
                    eligibility=ELIGIBILITY
                )
                fout.write("{}\t{}\t+/-{}\n".format(name, evaluation, se))
                # env_interaction.plotQ(env, rl.evalQ)
                # env_interaction.play(env, rl.getPolicy())

    # 2. learn combination of tasks for full
    sources = []
    for name, param in SOURCES.iteritems():
        if name in SOURCE_NAMES:
            sources.append(base_rl.SimpleQLearning(
                name=name, 
                actions=range(env.action_space.n), 
                discount=DISCOUNT, 
                discreteExtractor=env_interaction.discreteExtractor(env), 
                featureExtractor=env_interaction.simpleFeatures(env), 
                explorationProb=0., 
                weights="weights/{}{}.p".format(param["file_name"][:-2], NUM_TRIALS_SOURCES)
            ))

    param = TARGET[TARGET_NAME]

    env.set_task_params(param)

    # env.set_task(
    #     modes=[
    #     ("time", 1),
    #     ("energy", 0),
    #     ("distance", 0),
    #     ("center", 0),
    #     ("height", 0),
    #     ("still", 0)
    #     ], 
    #     slope=0.0025, 
    #     speed=0.07, 
    #     power=0.001, 
    #     min_position=-1.2,
    #     low=-0.6,
    #     high=-0.4,
    #     obstacles=[
    #     (-.5, .1, .01), 
    #     (0, .1, .05)], 
    #     actions_nb=3,
    #     neutral=1
    #     )


    ##################################################

    # Train on target with different values of num_trials
    for num_trials in NUM_TRIALS:
        fout.write("# learning trials: {}\n".format(num_trials))
        print "\n\n{} trials".format(num_trials)

        ########## NEURAL NETWORK IMPLEMENTATION #########
        print "\nDeep transfer"
        rl_deep, training_reward = deep_rl.target_train(
            env, 
            TARGET_NAME, 
            sources, 
            num_trials=num_trials, 
            max_iter=MAX_ITER, 
            filename="params_deep.p", 
            verbose=VERBOSE, 
            reload_weights=RELOAD_WEIGHTS, 
            discount=DISCOUNT, 
            explorationProb=EXPLORATION_PROBA,
            eligibility=False,
            mode=DEEP_MODE
        )

        evaluation, se = env_interaction.policy_evaluation(
            env=env, 
            policy=rl_deep.getPolicy(), 
            discount=DISCOUNT,
            num_trials=NUM_TRIALS_EVAL,
            max_iter=MAX_ITER
        )

        fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(TARGET_NAME+"_target_deep", evaluation, se, training_reward))


        ##################################################
        print "\nLinear transfer"
        rl_ens, training_reward = ensemble_rl.target_train(
            env, 
            TARGET_NAME, 
            sources, 
            num_trials=num_trials, 
            max_iter=MAX_ITER, 
            filename="coefs.p", 
            verbose = VERBOSE, 
            reload_weights=RELOAD_WEIGHTS, 
            discount=DISCOUNT, 
            explorationProb=EXPLORATION_PROBA,
            eligibility=False
        )
        evaluation, se = env_interaction.policy_evaluation(
            env=env, 
            policy=rl_ens.getPolicy(), 
            discount=DISCOUNT,
            num_trials=NUM_TRIALS_EVAL,
            max_iter=MAX_ITER
        )

        fout.write("\t{}\t{}\t+/-{}\t({} at training time)\n".format(TARGET_NAME+"_target_linear", evaluation, se, training_reward))

        # 3. compare with direct learning
        evaluation, se, training_reward = base_rl.train_task(
            discreteExtractor=env_interaction.discreteExtractor(env), 
            featureExtractor=env_interaction.simpleFeatures(env), 
            env=env, 
            name=TARGET_NAME+"_direct", 
            param=param, 
            num_trials=num_trials, 
            max_iter=MAX_ITER, 
            verbose=VERBOSE, 
            reload_weights=RELOAD_WEIGHTS, 
            discount=DISCOUNT, 
            explorationProb=EXPLORATION_PROBA,
            eligibility=False
        )

        fout.write("\t{}_direct\t{}\t+/-{}\t({} at training time)\n\n".format(TARGET_NAME, evaluation, se, training_reward))


    # save config parameters
    fout.write("\nConfig:\n" + "\n".join(
        ["{}\t\t{}".format(k,v) for k,v in config.__dict__.iteritems() if not k.startswith('__')]
    ))

    # save sources and target
    fout.write("\n\nTarget:\n" + json.dumps(param, indent=4))
    fout.write("\n\nSources:\n" + json.dumps(SOURCES, indent=4))

    fout.close()


