import sys, os, glob
from tqdm import tqdm
tqdm.monitor_interval = 0
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from utils import *
from parameters import *
from plot import *
from sample import *


game_info_df = pd.read_csv(GAME_INFO_FILENAME)
all_teams = get_teams(game_info_df)

test_ids_df = pd.read_csv('../data/football-events/test_ids.csv')
game_info_df = game_info_df[game_info_df['id_odsp'].isin(test_ids_df['test_id'].values)]
game_info_df = game_info_df[game_info_df['ht'].isin(all_teams) & game_info_df['at'].isin(all_teams)]

nb_games_test = len(game_info_df)
print("Testing %d games...\n" % nb_games_test)

target = torch.zeros(nb_games_test)
bookmaker_tensor = torch.zeros(nb_games_test, 3)
prediction_tensor = torch.zeros(nb_games_test, 3)
bookmaker_accuracy = 0
our_accuracy = 0
i = 0
for _, row in tqdm(game_info_df[-nb_games_test + 1:].iterrows()):
    idd = row['id_odsp']
    home_team = row['ht']
    away_team = row['at']

    # Add bookmaker predictions
    odd_h = row['odd_h']
    odd_d = row['odd_d']
    odd_a = row['odd_a']
    odds_tensor = torch.FloatTensor([1/odd_h, 1/odd_a, 1/odd_d])
    odds_tensor *= 1 / torch.sum(odds_tensor)
    bookmaker_tensor[i, :] = odds_tensor

    # Add our predictions
    exp_home_win, exp_away_win, exp_draw = sample_n_times([home_team, away_team], 70)
    game_pred_tensor = torch.FloatTensor([exp_home_win, exp_away_win, exp_draw])
    prediction_tensor[i, :] = game_pred_tensor

    # Add the real results
    home_goals = row['fthg']
    away_goals = row['ftag']
    if home_goals > away_goals:
        target[i] = 0
        bookmaker_accuracy += odds_tensor[0].item()
        our_accuracy += game_pred_tensor[0].item()
    elif away_goals > home_goals:
        target[i] = 1
        bookmaker_accuracy += odds_tensor[1].item()
        our_accuracy += game_pred_tensor[1].item()
    else:
        target[i] = 2
        bookmaker_accuracy += odds_tensor[2].item()
        our_accuracy += game_pred_tensor[2].item()

    i += 1

target = target.type(torch.LongTensor)
bookmaker_loss = F.cross_entropy(bookmaker_tensor, target)
our_loss = F.cross_entropy(prediction_tensor, target)
bookmaker_accuracy /= nb_games_test
our_accuracy /= nb_games_test

print("Bookmaker loss", bookmaker_loss)
print("Our loss:", our_loss)
print()
print("Bookmaker accuracy", bookmaker_accuracy)
print("Our accuracy:", our_accuracy)

