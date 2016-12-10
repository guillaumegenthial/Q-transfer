EXP_NAME = "moreActions500_cleanSources"
ENV = 'MountainCar-v0'
TARGET_NAMES = [
# "more_actions", 
# "full_clean",
"full",
# "full_energy"
]
SOURCE_NAMES = [
"standard", 
"bananas", 
"bumps", 
"easy slope", 
"more_power", 
"energy", 
"distance", 
"center", 
"height", 
"still",
"more_actions1"]
VERBOSE = False
EXPLORATION_PROBA_START = 1.
EXPLORATION_PROBA_END = 0.1
MAX_ITER = 5000
NUM_TRIALS_SOURCES = 1000
NUM_TRIALS_TARGETS = [2, 5, 10, 100]
NUM_TRIALS_EVAL = 1000
RELOAD_WEIGHTS = False
DISCOUNT = 1
ELIGIBILITY = False
TRAIN = True
DEEP_MODES = [1, 2]
AVERAGE_TIMES = 1