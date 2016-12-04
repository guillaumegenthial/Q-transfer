import gym
import base_rl
import env_interaction
import tasks
import ensemble_rl

ENV = 'MountainCar-v0'
VERBOSE = False
EXPLORATION_PROBA = 0.2
MAX_ITER = 1000
NUM_TRIALS = 5000
RELOAD_WEIGHTS = False
DISCOUNT = 1
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
    if TRAIN:
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

    name = "full"
    param = tasks.TARGET[name]
    file_name = param["file_name"]
    slope = param["slope"]
    reward_modes = param["reward_modes"]
    max_speed = param["max_speed"]
    power = param["power"]

    env.set_task(reward_modes, slope, max_speed, power)

    rl_ens = ensemble_rl.target_train(
        env, 
        name, 
        sources, 
        num_trials=10, 
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

    with open("results.txt", "a") as f:
        f.write("{} {}\n".format(name+"_target", evaluation))

    # 3. compare with direct learning
    base_rl.train_task(
        env, 
        name+"_direct", 
        param, 
        NUM_TRIALS, 
        MAX_ITER, 
        VERBOSE, 
        RELOAD_WEIGHTS, 
        DISCOUNT, 
        EXPLORATION_PROBA)

    with open("results.txt", "a") as f:
        f.write("\n")


