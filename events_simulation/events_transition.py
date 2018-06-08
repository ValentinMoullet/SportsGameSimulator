from model import *
from utils import *
from parameters import *
from plot import *
from sample import *

import sys, os, glob
from tqdm import tqdm
tqdm.monitor_interval = 0
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['figure.figsize']=(15, 8)
plt.style.use('ggplot')
import seaborn as sns
sns.set()

EVENTS = ['Goals', 'Attempts', 'Corners', 'Fouls', 'Yellow cards', 'Second yellow cards', 'Red cards', 'Substitutions', 'Free kicks', 'Offsides', 'Hand balls', 'Penaltys']
NB_SAMPLES = 10

model = load_latest_model()

events_df = pd.read_csv('../data/football-events/new_events.csv')
test_ids_df = pd.read_csv('../data/football-events/test_ids.csv')
test_df = events_df[events_df['id_odsp'].isin(test_ids_df['test_id'].values)]

ids_to_df = {key: test_df.loc[value] for key, value in test_df.groupby("id_odsp").groups.items()}

nb_games = len(ids_to_df)

all_teams = get_teams(events_df, home_team_col_name='home_team', away_team_col_name='away_team')

transition_table = [[0] * 12 for _ in range(12)]
transition_table_away = [[0] * 12 for _ in range(12)]

events_count = [0] * 12

total_nb_events = 0
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

    for s in range(NB_SAMPLES):
        sampled_events, sampled_times = model.sample([[home_team, away_team]])

        prev_event = None
        prev_side = None
        for event in sampled_events:
            if event >= NO_EVENT:
                continue

            if event >= NB_EVENT_TYPES:
                event -= NB_EVENT_TYPES
                side = 2
            else:
                side = 1

            total_nb_events += 1

            if not prev_event is None and not prev_side is None and prev_side == side:
                transition_table[prev_event][event] += 1
            
            if not prev_event is None and not prev_side is None and prev_side != side:
                transition_table_away[prev_event][event] += 1
                
            events_count[event] += 1
                
            prev_event = event
            prev_side = side

events_proba = [[0] * 12 for _ in range(12)]
away_events_proba = [[0] * 12 for _ in range(12)]

total_proba = 0
for prev_event, event_list in enumerate(transition_table):
    for new_event, count in enumerate(event_list):
        proba = count / events_count[prev_event]
        proba_normalized = proba / (events_count[new_event] / total_nb_events)
        total_proba += proba
        events_proba[prev_event][new_event] = proba_normalized
                
for prev_event, event_list in enumerate(transition_table_away):
    for new_event, count in enumerate(event_list):
        proba = count / events_count[prev_event]
        proba_normalized = proba / (events_count[new_event] / total_nb_events)
        total_proba += proba
        away_events_proba[prev_event][new_event] = proba_normalized

plt.figure(figsize=(10,5))
home_events_df = pd.DataFrame(events_proba, columns=EVENTS).set_index([EVENTS])
ax = sns.heatmap(home_events_df, cmap='YlOrRd', annot=True, vmin=0, vmax=5, fmt='.2f')
ax.set_xticklabels(EVENTS, rotation=40, ha='right')
ax.figure.savefig('events_transition/same_side_event_influence.pdf')

plt.figure(figsize=(10,5))
away_events_df = pd.DataFrame(away_events_proba, columns=EVENTS).set_index([EVENTS])
ax = sns.heatmap(away_events_df, cmap='YlOrRd', annot=True, vmin=0, vmax=5, fmt='.2f')
ax.set_xticklabels(EVENTS, rotation=40, ha='right')
ax.figure.savefig('events_transition/other_side_event_influence.pdf')

# Make difference
global_home_events_df = pd.read_pickle('events_transition/same_side_event_influence.pkl')
global_away_events_df = pd.read_pickle('events_transition/other_side_event_influence.pkl')

diff_home_events_df = global_home_events_df.replace(0, 0.01) / home_events_df.replace(0, 0.01)
diff_away_events_df = global_away_events_df.replace(0, 0.01) / away_events_df.replace(0, 0.01)

plt.figure(figsize=(10,5))
ax = sns.heatmap(diff_home_events_df, cmap='coolwarm', annot=True, vmin=0, vmax=2, center=1, fmt='.2f')
ax.set_xticklabels(EVENTS, rotation=40, ha='right')
ax.figure.savefig('events_transition/ratio_same_side_event_influence.pdf')

plt.figure(figsize=(10,5))
ax = sns.heatmap(diff_away_events_df, cmap='coolwarm', annot=True, vmin=0, vmax=2, center=1, fmt='.2f')
ax.set_xticklabels(EVENTS, rotation=40, ha='right')
ax.figure.savefig('events_transition/ratio_other_side_event_influence.pdf')

