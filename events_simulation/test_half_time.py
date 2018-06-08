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


NB_SAMPLES = 20
TIME_STARTING_2ND = 46

model = load_latest_model()

events_df = pd.read_csv('../data/football-events/new_events.csv')
test_ids_df = pd.read_csv('../data/football-events/test_ids.csv')
test_df = events_df[events_df['id_odsp'].isin(test_ids_df['test_id'].values)]

ids_to_df = {key: test_df.loc[value] for key, value in test_df.groupby("id_odsp").groups.items()}

nb_games = len(ids_to_df)

all_teams = get_teams(events_df, home_team_col_name='home_team', away_team_col_name='away_team')

loss_sampled_with_help = 0
loss_sampled_no_help = 0
accuracy_with_help = 0
accuracy_no_help = 0
unknown_teams = 0
for idd in tqdm(sorted(ids_to_df)[:nb_games]):
    df = ids_to_df[idd]

    tensor_df = df.loc[:, '0':]
    home_team = df['home_team'].iloc[0]
    away_team = df['away_team'].iloc[0]

    # Remove special cases
    if not home_team in all_teams or not away_team in all_teams:
        unknown_teams += 1
        continue

    events = [SOG_TOKEN]
    times = [GAME_NOT_RUNNING_TIME]
    current_time = 0
    start_2nd_half = -1
    goal_home_1st = 0
    goal_away_1st = 0
    goal_home_2nd = 0
    goal_away_2nd = 0
    idx = 0
    for _, row in tensor_df.iterrows():
        # Find index of 1's
        event, time = [i for i, x in enumerate(row.values) if x == 1]
        time -= NB_ALL_EVENTS

        if time == DIFF_TIME_THAN_PREV:
            current_time += 1
            if current_time == TIME_STARTING_2ND:
                start_2nd_half = idx
                #break

        events.append(event)
        times.append(time)

        if current_time < TIME_STARTING_2ND and event == GOAL_HOME:
            goal_home_1st += 1

        if current_time < TIME_STARTING_2ND and event == GOAL_AWAY:
            goal_away_1st += 1

        if current_time >= TIME_STARTING_2ND and event == GOAL_HOME:
            goal_home_2nd += 1

        if current_time >= TIME_STARTING_2ND and event == GOAL_AWAY:
            goal_away_2nd += 1

        idx += 1

    # Sample a few times with help
    home_win_2nd = away_win_2nd = draw_2nd = 0
    home_win_1st = away_win_1st = draw_1st = 0
    for s in range(NB_SAMPLES):
        # +2 because we init events and times with 1 element
        sampled_events, sampled_times = model.sample([[home_team, away_team]], events=events[:start_2nd_half+2], times=times[:start_2nd_half+2])
        #sampled_events, sampled_times = model.sample([[home_team, away_team]], events=[], times=[])

        # Find end of first half
        current_time = 0
        start_2nd_half = -1
        for idx, time in enumerate(sampled_times):
            if time == DIFF_TIME_THAN_PREV:
                current_time += 1
                if current_time == TIME_STARTING_2ND:
                    start_2nd_half = idx
                    break

            #idx += 1

        # +1 because we init events and times with 1 element
        pred_goal_home_2nd = sampled_events[start_2nd_half+1:].count(GOAL_HOME)
        pred_goal_away_2nd = sampled_events[start_2nd_half+1:].count(GOAL_AWAY)
        #pred_goal_home_1st = sampled_events[:start_2nd_half].count(GOAL_HOME)
        #pred_goal_away_1st = sampled_events[:start_2nd_half].count(GOAL_AWAY)

        if pred_goal_home_2nd > pred_goal_away_2nd:
            home_win_2nd += 1
        elif pred_goal_home_2nd < pred_goal_away_2nd:
            away_win_2nd += 1
        else:
            draw_2nd += 1

        '''
        if pred_goal_home_1st > pred_goal_away_1st:
            home_win_1st += 1
        elif pred_goal_home_1st < pred_goal_away_1st:
            away_win_1st += 1
        else:
            draw_1st += 1
        '''

    exp_home_win = home_win_2nd / NB_SAMPLES
    exp_away_win = away_win_2nd / NB_SAMPLES
    exp_draw = draw_2nd / NB_SAMPLES
    exp_sampled_with_help = torch.FloatTensor([[exp_home_win, exp_away_win, exp_draw]])
    if goal_home_2nd > goal_away_2nd:
        target = 0
    elif goal_home_2nd < goal_away_2nd:
        target = 1
    else:
        target = 2

    target = torch.LongTensor([target])

    # Compute CE loss
    loss_sampled_with_help += F.cross_entropy(exp_sampled_with_help, target)

    # Accuracy
    accuracy_with_help += exp_sampled_with_help[0][target.item()]

    '''
    exp_home_win = home_win_1st / NB_SAMPLES
    exp_away_win = away_win_1st / NB_SAMPLES
    exp_draw = draw_1st / NB_SAMPLES
    exp_sampled_with_help = torch.FloatTensor([[exp_home_win, exp_away_win, exp_draw]])
    if goal_home_1st > goal_away_1st:
        target = 0
    elif goal_home_1st < goal_away_1st:
        target = 1
    else:
        target = 2

    target = torch.LongTensor(target)

    # Compute CE loss
    loss_sampled_with_help += F.cross_entropy(exp_sampled_with_help, target)

    # Accuracy
    accuracy_with_help += exp_sampled_with_help[0][target.item()]

    '''

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
    exp_sampled_no_help = torch.FloatTensor([[exp_home_win, exp_away_win, exp_draw]])

    # Compute CE loss
    loss_sampled_no_help += F.cross_entropy(exp_sampled_no_help, target)

    # Accuracy
    accuracy_no_help += exp_sampled_no_help[0][target.item()]

loss_sampled_with_help /= (nb_games - unknown_teams)
loss_sampled_no_help /= (nb_games - unknown_teams)

accuracy_with_help /= (nb_games - unknown_teams)
accuracy_no_help /= (nb_games - unknown_teams)

print("Loss with help:", loss_sampled_with_help)
print("Loss no help:", loss_sampled_no_help)

print("Accuracy with help:", accuracy_with_help)
print("Accuracy no help:", accuracy_no_help)

        
