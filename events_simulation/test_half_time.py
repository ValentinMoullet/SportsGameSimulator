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


NB_SAMPLES = 10
TIME_STARTING_2ND = 46

model = load_latest_model()

events_df = pd.read_csv('../data/football-events/new_events.csv')

ids_to_df = {key: events_df.loc[value] for key, value in events_df.groupby("id_odsp").groups.items()}

nb_games = len(ids_to_df)

all_teams = get_teams(events_df, home_team_col_name='home_team', away_team_col_name='away_team')

accuracy_with_help = 0
accuracy_no_help = 0
for idd in tqdm(sorted(ids_to_df)[:nb_games]):
    df = ids_to_df[idd]

    tensor_df = df.loc[:, '0':]
    home_team = df['home_team'].iloc[0]
    away_team = df['away_team'].iloc[0]

    # Remove special cases
    if not home_team in all_teams or not away_team in all_teams:
        continue

    events = [SOG_TOKEN]
    times = [GAME_NOT_RUNNING_TIME]
    current_time = 0
    start_2nd_half = -1
    goal_home_2nd = 0
    goal_away_2nd = 0
    idx = 0
    for _, row in tensor_df.iterrows():
        # Find index of 1's
        event, time = [i for i, x in enumerate(row.values) if x == 1]
        time -= NB_ALL_EVENTS

        events.append(event)
        times.append(time)

        if time == DIFF_TIME_THAN_PREV:
            current_time += 1
            if current_time == TIME_STARTING_2ND:
                start_2nd_half = idx

        if current_time >= TIME_STARTING_2ND and event == GOAL_HOME:
            goal_home_2nd += 1

        if current_time >= TIME_STARTING_2ND and event == GOAL_AWAY:
            goal_away_2nd += 1

        idx += 1

    # Sample a few times with help
    home_win_2nd = away_win_2nd = draw_2nd = 0
    for s in range(NB_SAMPLES):
        # +2 because we init events and times with 1 element
        sampled_events, sampled_times = model.sample([[home_team, away_team]], events=events[:start_2nd_half+2], times=times[:start_2nd_half+2])

        # +1 because we init events and times with 1 element
        pred_goal_home_2nd = sampled_events[start_2nd_half+1:].count(GOAL_HOME)
        pred_goal_away_2nd = sampled_events[start_2nd_half+1:].count(GOAL_AWAY)

        if pred_goal_home_2nd > pred_goal_away_2nd:
            home_win_2nd += 1
        elif pred_goal_home_2nd < pred_goal_away_2nd:
            away_win_2nd += 1
        else:
            draw_2nd += 1

    exp_home_win = home_win_2nd / NB_SAMPLES
    exp_away_win = away_win_2nd / NB_SAMPLES
    exp_draw = draw_2nd / NB_SAMPLES
    if goal_home_2nd > goal_away_2nd:
        accuracy_with_help += exp_home_win
    elif goal_home_2nd < goal_away_2nd:
        accuracy_with_help += exp_away_win
    else:
        accuracy_with_help += exp_draw

    # Sample a few times without help
    home_win_2nd = away_win_2nd = draw_2nd = 0
    for s in range(NB_SAMPLES):
        sampled_events, sampled_times = model.sample([[home_team, away_team]])

        # +1 because we init events and times with 1 element
        pred_goal_home_2nd = sampled_events[start_2nd_half+1:].count(GOAL_HOME)
        pred_goal_away_2nd = sampled_events[start_2nd_half+1:].count(GOAL_AWAY)

        if pred_goal_home_2nd > pred_goal_away_2nd:
            home_win_2nd += 1
        elif pred_goal_home_2nd < pred_goal_away_2nd:
            away_win_2nd += 1
        else:
            draw_2nd += 1

    exp_home_win = home_win_2nd / NB_SAMPLES
    exp_away_win = away_win_2nd / NB_SAMPLES
    exp_draw = draw_2nd / NB_SAMPLES
    if goal_home_2nd > goal_away_2nd:
        accuracy_no_help += exp_home_win
    elif goal_home_2nd < goal_away_2nd:
        accuracy_no_help += exp_away_win
    else:
        accuracy_no_help += exp_draw

accuracy_with_help /= nb_games
accuracy_no_help /= nb_games

print("Accuracy with help:", accuracy_with_help)
print("Accuracy no help:", accuracy_no_help)

        
