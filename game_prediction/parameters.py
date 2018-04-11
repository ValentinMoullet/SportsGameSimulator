DEFAULT_LEAGUE = 'F1'

SEED = 42
TRAINING_SET_RATIO = 0.8
VALIDATION_SET_RATIO = 0.2
K_FOLD = 4
BOOKMAKERS_OVERVIEW = True

MAX_EPOCH = 300
MAX_EPOCH_WITHOUT_IMPROV = 50
BATCH_SIZES = [1, 10, 25]
LEARNING_RATES = [1e-4]
HIDDEN_LAYER_SIZES = [(5, 10), (10, 5), (20, 20)]
DROPOUT_RATES = [0, 0.3]

CHOSEN_EPOCH = 50
CHOSEN_BATCH_SIZE = 5
CHOSEN_LEARNING_RATE = 1e-4
CHOSEN_HIDDEN_LAYER_SIZES = (20, 20)
CHOSEN_DROPOUT_RATE = 0

IMAGES_DIR = './images'
MODELS_DIR = './models'