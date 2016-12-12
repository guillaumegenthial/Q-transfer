EXP_NAME = "all_target_all_sources"
ENV = 'MountainCar-v0'
TARGET_NAMES = [
"more_actions", 
"full_clean",
"full",
"full_energy",
"time"
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
"more_actions1",
"more_actions2"]
VERBOSE = False
EXPLORATION_PROBA_START = 0.5
EXPLORATION_PROBA_END = 0.05
MAX_ITER = 5000
NUM_TRIALS_SOURCES = 1000
NUM_TRIALS_TARGETS = [2, 5, 10, 100]
NUM_TRIALS_EVAL = 100
RELOAD_WEIGHTS = False
DISCOUNT = 1
ELIGIBILITY = False
TRAIN = False
DEEP_MODES = [1, 2, 3]
AVERAGE_TIMES = 1
LR_ENSEMBLE = 0.001
LR_DEEP = 0.001
