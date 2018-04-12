from utils import *
from game_prediction.model import *
from game_prediction.parameters import CHOSEN_LEARNING_RATE, CHOSEN_HIDDEN_LAYER_SIZES, CHOSEN_DROPOUT_RATE


MODELS_FOLDER = 'game_prediction/models'


########## Load pre-saved models ##########

models = {}

model = NN(nb_teams=26, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)
model.load_state_dict(torch.load("%s/%s/%s" % (MODELS_FOLDER, 'D1', get_hyperparams_filename('model.pt'))))
models['D1'] = model

model = NN(nb_teams=31, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)
model.load_state_dict(torch.load("%s/%s/%s" % (MODELS_FOLDER, 'E0', get_hyperparams_filename('model.pt'))))
models['E0'] = model

model = NN(nb_teams=30, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)
model.load_state_dict(torch.load("%s/%s/%s" % (MODELS_FOLDER, 'F1', get_hyperparams_filename('model.pt'))))
models['F1'] = model

model = NN(nb_teams=30, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)
model.load_state_dict(torch.load("%s/%s/%s" % (MODELS_FOLDER, 'I1', get_hyperparams_filename('model.pt'))))
models['I1'] = model

model = NN(nb_teams=30, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)
model.load_state_dict(torch.load("%s/%s/%s" % (MODELS_FOLDER, 'SP1', get_hyperparams_filename('model.pt'))))
models['SP1'] = model


def get_model(league):
    return models[league]