# 'event_type'
GOAL = 0 # 'announcement' is never used
SHOOT = 1

NB_EVENT_TYPES = 12

# 'event_type2'
OWN_GOAL = 15

# Our own events list
GOAL_HOME = 0
GOAL_AWAY = NB_EVENT_TYPES

SHOT_HOME = 1
SHOT_AWAY = NB_EVENT_TYPES + 1

CORNER_HOME = 2
CORNER_AWAY = NB_EVENT_TYPES + 2

FOUL_HOME = 3
FOUL_AWAY = NB_EVENT_TYPES + 3

NO_EVENT = NB_EVENT_TYPES * 2
GAME_OVER = NB_EVENT_TYPES * 2 + 1

# Contains all event types (for home and for away team) + no event + match over
NB_ALL_EVENTS = NB_EVENT_TYPES * 2 + 2

# Parameters
K_FOLD = 2
MAX_EPOCH = 30
BATCH_SIZE = 10

IMAGES_DIR = 'images'
EVENTS_DIR = 'events'
