#import torch
CUDA = False

# 'event_type'
GOAL = 0 # 'announcement' is never used
SHOOT = 1

NB_EVENT_TYPES = 12

# 'event_type2'
OWN_GOAL = 15

########## Our own events list ##########
GOAL_HOME = 0
GOAL_AWAY = NB_EVENT_TYPES
SHOT_HOME = 1
SHOT_AWAY = NB_EVENT_TYPES + 1
CORNER_HOME = 2
CORNER_AWAY = NB_EVENT_TYPES + 2
FOUL_HOME = 3
FOUL_AWAY = NB_EVENT_TYPES + 3
YELLOW_CARD_HOME = 4
YELLOW_CARD_AWAY = NB_EVENT_TYPES + 4
SUBSTITUTION_HOME = 7
SUBSTITUTION_AWAY = NB_EVENT_TYPES + 7
OFFSIDE_HOME = 9
OFFSIDE_AWAY = NB_EVENT_TYPES + 9
NO_EVENT = NB_EVENT_TYPES * 2
GAME_OVER = NB_EVENT_TYPES * 2 + 1
GAME_STARTING = NB_EVENT_TYPES * 2 + 2
# Contains all event types (for home and for away team) + no event + match over
NB_ALL_EVENTS = NB_EVENT_TYPES * 2 + 3

########## Our time list (after events list) ##########
SAME_TIME_THAN_PREV = 0
DIFF_TIME_THAN_PREV = 1
GAME_NOT_RUNNING_TIME = 2
# Contains all times possibilities
NB_ALL_TIMES = 3

# Parameters
K_FOLD = 1
MAX_EPOCH = 40
BATCH_SIZE = 15
SAMPLE_VALID_AND_TEST = True
NB_GAMES_TO_SAMPLE = 10

# Some constants from previous models
# Commented ones are for full dataset, others are for a quarter of it

#BOOKMAKER_CE_LOSS = 1.0252809092279713
BOOKMAKER_CE_LOSS = 1.0288130576418444
#BOOKMAKER_ACCURACY = 0.4216
BOOKMAKER_ACCURACY = 0.4181
#SIMPLE_NN_CE_LOSS = 1.02923257012738
SIMPLE_NN_CE_LOSS = 1.02883257012738
#SIMPLE_NN_ACCURACY = 0.41593
SIMPLE_NN_ACCURACY = 0.412578

MAX_SAME_MINUTE_EVENTS = 10

# Constants for Beta distribution
PROB_SAME_MINUTE_EVENT = 0.3516
BETA_WEIGHT = 0.35
ALPHA_FOR_BETA_DISTR = 4.0
BETA_FOR_BETA_DISTR = 6.53242321

# Files and directories
GAME_INFO_FILENAME = '../data/football-events/ginf.csv'
TRAINING_IDS_FILE = '../data/football-events/training_ids.csv'
TEST_IDS_FILE = '../data/football-events/test_ids.csv'
NEW_EVENTS_FILE = '../data/football-events/new_events.csv'
IMAGES_DIR = 'images'
EVENTS_DIR = 'events'
MODELS_DIR = 'models'
EVENTS_PROBA_DIR = 'events_proba'
DISTR_DIR = 'distributions'
SECOND_HALF_PRED_DIR = '2nd_half_pred'
