import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from parameters import *
from utils import *
from sample import *


MAX_NB_EVENT = 40
NB_TO_SAMPLE = 100
EVENT_TYPES_TO_CHECK = [SHOT_HOME, CORNER_HOME, FOUL_HOME, SHOT_AWAY, CORNER_AWAY, FOUL_AWAY]


def get_event_distr(ids_to_df, event_type):
    event_distr = [0] * MAX_NB_EVENT
    for idd, df in ids_to_df.items():
        event_df = df[df[str(event_type)] == 1]
        event_distr[len(event_df)] += 1
        
    return [x / len(ids_to_df) for x in event_distr]

events_df = pd.read_csv(NEW_EVENTS_FILE)

# Getting global distribution in training set
print("Getting global distribution in training set...")
training_ids_df = pd.read_csv(TRAINING_IDS_FILE)[['training_id']]
training_events_df = events_df[events_df['id_odsp'].isin(training_ids_df['training_id'].values)]
events_and_times_df = training_events_df.loc[:, '0':]

training_events_distr = []
team_training_events_distr = []
ids_to_df = {key: training_events_df.loc[value] for key, value in training_events_df.groupby("id_odsp").groups.items()}
for event_type in tqdm(range(NB_ALL_EVENTS - 3)):
    event_distr = get_event_distr(ids_to_df, event_type)
    # Could plot here
    training_events_distr.append(event_distr)
    team_training_events_distr.append({})

    # Teams
    for idd, df in ids_to_df.items():
        home_team = df['home_team'].iloc[0]
        away_team = df['away_team'].iloc[0]
        if event_type < NB_EVENT_TYPES:
            team = home_team
        else:
            team = away_team

        if not team in team_training_events_distr[-1]:
            team_training_events_distr[-1][team] = [0] * MAX_NB_EVENT

        nb_events = len(df[df[str(event_type)] == 1])
        team_training_events_distr[-1][team][nb_events] += 1

    # Want to get the distribution (sum to 1)
    new_team_to_distr = {}
    for team, team_distr in team_training_events_distr[-1].items():
        total_team_games = sum(team_distr)
        new_team_to_distr[team] = [x / total_team_games for x in team_distr]
        #print(new_team_to_distr[team])

    team_training_events_distr[-1] = new_team_to_distr

# Getting sampled distribution in test set
print("Getting sampled distribution and real number of events in test set...")
test_ids_df = pd.read_csv(TEST_IDS_FILE)[['test_id']]
test_events_df = events_df[events_df['id_odsp'].isin(test_ids_df['test_id'].values)]
ids_to_df = {key: test_events_df.loc[value] for key, value in test_events_df.groupby("id_odsp").groups.items()}

nb_games_test = len(ids_to_df) // 100
ids_to_df = {k: ids_to_df[k] for k in list(ids_to_df)[:nb_games_test]}

global_accuracy = [0] * (NB_ALL_EVENTS - 3)
team_accuracy = [0] * (NB_ALL_EVENTS - 3)
sampled_accuracy = [0] * (NB_ALL_EVENTS - 3)
unknown_teams = 0
game_nb = 0
for idd, df in tqdm(ids_to_df.items()):
    home_team = df['home_team'].iloc[0]
    away_team = df['away_team'].iloc[0]
    all_sampled_events, _ = sample_n_times_events([home_team, away_team], NB_TO_SAMPLE)
    test_events_distr = []
    true_nb_events = []
    for event_type in range(NB_ALL_EVENTS - 3):
        # Store real number of event_type for this game
        real_nb_events = len(df[df[str(event_type)] == 1])
        true_nb_events.append(real_nb_events)

        # Compute distribution when sampling
        event_distr = [0] * MAX_NB_EVENT
        for sampled_events in all_sampled_events:
            nb_events = sampled_events.count(event_type)
            event_distr[nb_events] += 1

        event_distr = [x / NB_TO_SAMPLE for x in event_distr]
        test_events_distr.append(event_distr)

        if event_type in EVENT_TYPES_TO_CHECK:
            global_accuracy[event_type] += training_events_distr[event_type][real_nb_events]
            sampled_accuracy[event_type] += test_events_distr[event_type][real_nb_events]
            if home_team in team_training_events_distr[event_type]:
                team_accuracy[event_type] += team_training_events_distr[event_type][home_team][real_nb_events]
                if game_nb < 10:
                    plot_3_bars(training_events_distr[event_type], team_training_events_distr[event_type][home_team], test_events_distr[event_type], real_nb_events, "distr_%d_%s_%s.pdf" % (event_type, home_team, away_team), title="%s: %s -VS- %s" % (event_type_to_string(event_type), home_team, away_team))
            else:
                unknown_teams += 1

    game_nb += 1

print("unknown_teams:", unknown_teams)
print("len(EVENT_TYPES_TO_CHECK):", len(EVENT_TYPES_TO_CHECK))
unknown_teams /= len(EVENT_TYPES_TO_CHECK)

global_accuracy = [x / nb_games_test for x in global_accuracy]
team_accuracy = [x / (nb_games_test - unknown_teams) for x in team_accuracy]
sampled_accuracy = [x / nb_games_test for x in sampled_accuracy]

for event_type in EVENT_TYPES_TO_CHECK:
    print("global_accuracy for %s:" % event_type_to_string(event_type), global_accuracy[event_type])
    print("team_accuracy for %s:" % event_type_to_string(event_type), team_accuracy[event_type])
    print("sampled_accuracy for %s:" % event_type_to_string(event_type), sampled_accuracy[event_type])
    print()

