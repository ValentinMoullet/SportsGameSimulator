import time
import numpy as np
import pandas as pd
import random
import pathlib
from scipy.stats.distributions import poisson
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.autograd import Variable

from parameters import *
from plot import *
from game_prediction.parameters import CHOSEN_BATCH_SIZE, CHOSEN_LEARNING_RATE, CHOSEN_HIDDEN_LAYER_SIZES, CHOSEN_DROPOUT_RATE


# Create <SOG> token (start of game token)
SOG_TOKEN = [0]*(NB_ALL_EVENTS + NB_ALL_TIMES)
SOG_TOKEN[GAME_STARTING] = 1
SOG_TOKEN[NB_ALL_EVENTS + GAME_NOT_RUNNING_TIME] = 1

MAX_GOAL_FOR_TEAM = 20


########## Create events mapping ##########

event_type_to_sentence = {}
sides = ['home', 'away']
for i, side in enumerate(sides):
    for event_type in range(NB_EVENT_TYPES):
        if event_type == 0:
            sentence = "Goal %s." % side
        elif event_type == 1:
            sentence = "Shot %s." % side
        elif event_type == 2:
            sentence = "Corner %s." % side
        elif event_type == 3:
            sentence = "Foul %s." % side
        elif event_type == 4:
            sentence = "Yellow card %s." % side
        elif event_type == 5:
            sentence = "2nd yellow card %s." % side
        elif event_type == 6:
            sentence = "Red card %s." % side
        elif event_type == 7:
            sentence = "Substitution %s." % side
        elif event_type == 8:
            sentence = "Free kick %s." % side
        elif event_type == 9:
            sentence = "Offside %s." % side
        elif event_type == 10:
            sentence = "Hand ball %s." % side
        elif event_type == 11:
            sentence = "Penalty conceded %s." % side

        event_type_to_sentence[event_type + i * NB_EVENT_TYPES] = sentence

event_type_to_sentence[NB_EVENT_TYPES * 2] = "No event this minute."
event_type_to_sentence[NB_EVENT_TYPES * 2 + 1] = "Game is over."
event_type_to_sentence[NB_EVENT_TYPES * 2 + 2] = "Game is starting."

time_type_to_sentence = {}
time_type_to_sentence[SAME_TIME_THAN_PREV] = "Same time."
time_type_to_sentence[DIFF_TIME_THAN_PREV] = "Diff time."
time_type_to_sentence[GAME_NOT_RUNNING_TIME] = "Game is not running."


def event_type_to_string(event_type):
    return event_type_to_sentence[event_type]

def get_next_time(current_time, time_type, event_type, prev_event_type):
    '''
    if event_type == NO_EVENT or prev_event_type == NO_EVENT:
        return current_time + 1
    '''
    if time_type == 0:
        return current_time
    elif time_type == 1:
        return current_time + 1
    elif time_type == 2:
        return current_time

def repackage(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage(v) for v in h)

def count_events(events):
    events_count = [0]*NB_ALL_EVENTS
    for event_type in events:
        events_count[event_type] += 1

    events_count_dict = {}
    for i in range(NB_ALL_EVENTS):
        events_count_dict[event_type_to_sentence[i]] = events_count[i]

    return events_count_dict

def count_times(times):
    times_count = [0]*NB_ALL_TIMES
    for time_type in times:
        times_count[time_type] += 1

    times_count_dict = {}
    for i in range(NB_ALL_TIMES):
        times_count_dict[time_type_to_sentence[i]] = times_count[i]

    return times_count_dict

def output_events_file(event_scores, time_scores, target, teams, filename):
    #print(event_scores)
    #print(time_scores)
    #print("target:", target)

    target = target.data

    generated_events = generate_events(event_scores, time_scores)

    with open('%s/%s' % (EVENTS_DIR, filename), 'w+') as f:
        for batch_idx in range(generated_events.size(0)):
            f.write('\n\nNew game: %s -VS- %s\n' % (teams[batch_idx][0], teams[batch_idx][1]))
            f.write('--------------------\n\n')

            goal_home = 0
            goal_away = 0
            goal_home_target = 0
            goal_away_target = 0

            shot_home = 0
            shot_away = 0
            shot_home_target = 0
            shot_away_target = 0

            corner_home = 0
            corner_away = 0
            corner_home_target = 0
            corner_away_target = 0

            foul_home = 0
            foul_away = 0
            foul_home_target = 0
            foul_away_target = 0

            current_time = 0
            current_time_target = 0
            prev_event_type = -1
            prev_event_type_target = -1
            for event_idx in range(generated_events.size(1)):
                event_type = generated_events[batch_idx, event_idx, 0]
                time_type = generated_events[batch_idx, event_idx, 1]
                event_type_target = target[batch_idx, event_idx, 0]
                time_type_target = target[batch_idx, event_idx, 1]

                if event_type_target == GAME_OVER:
                    break

                if event_type == GOAL_HOME:
                    goal_home += 1
                elif event_type == GOAL_AWAY:
                    goal_away += 1

                if event_type_target == GOAL_HOME:
                    goal_home_target += 1
                elif event_type_target == GOAL_AWAY:
                    goal_away_target += 1

                if event_type == SHOT_HOME:
                    shot_home += 1
                elif event_type == SHOT_AWAY:
                    shot_away += 1

                if event_type_target == SHOT_HOME:
                    shot_home_target += 1
                elif event_type_target == SHOT_AWAY:
                    shot_away_target += 1

                if event_type == CORNER_HOME:
                    corner_home += 1
                elif event_type == CORNER_AWAY:
                    corner_away += 1

                if event_type_target == CORNER_HOME:
                    corner_home_target += 1
                elif event_type_target == CORNER_AWAY:
                    corner_away_target += 1

                if event_type == FOUL_HOME:
                    foul_home += 1
                elif event_type == FOUL_AWAY:
                    foul_away += 1

                if event_type_target == FOUL_HOME:
                    foul_home_target += 1
                elif event_type_target == FOUL_AWAY:
                    foul_away_target += 1

                current_time = get_next_time(current_time, time_type, event_type, prev_event_type)
                current_time_target = get_next_time(current_time_target, time_type_target, event_type_target, prev_event_type_target)
                
                # To remove?
                goal_home_proba = event_scores[batch_idx, event_idx, GOAL_HOME].data.item()
                goal_away_proba = event_scores[batch_idx, event_idx, GOAL_AWAY].data.item()

                sentence = event_type_to_sentence[event_type.item()] + (" (%.4f - %.4f)" % (goal_home_proba, goal_away_proba))
                sentence_target = event_type_to_sentence[event_type_target.item()]

                f.write("[%d'] %s \t\t[%d'] %s\n" % (current_time, sentence, current_time_target, sentence_target))

                prev_event_type = event_type
                prev_event_type_target = event_type_target

            game_over_indices = (target[batch_idx, :] == GAME_OVER).nonzero()
            if len(game_over_indices) == 0:
                idx = target.size(1)
            else:
                idx = game_over_indices[0, 0]

            goal_home_proba = torch.sum(event_scores[batch_idx, :idx, GOAL_HOME]).data.item()
            goal_away_proba = torch.sum(event_scores[batch_idx, :idx, GOAL_AWAY]).data.item()

            f.write("\nPredicted score: %d - %d (%.2f - %.2f), real one was %d - %d\n\n" % (goal_home, goal_away, goal_home_proba, goal_away_proba, goal_home_target, goal_away_target))
            f.write("Predicted: shot home = %d, shot away = %d\n" % (shot_home, shot_away))
            f.write("Real: shot home = %d, shot away = %d\n\n" % (shot_home_target, shot_away_target))
            f.write("Predicted: corner home = %d, corner away = %d\n" % (corner_home, corner_away))
            f.write("Real: corner home = %d, corner away = %d\n\n" % (corner_home_target, corner_away_target))
            f.write("Predicted: foul by home = %d, foul by away = %d\n" % (foul_home, foul_away))
            f.write("Real: foul by home = %d, foul by away = %d\n" % (foul_home_target, foul_away_target))


def output_already_sampled_events_file(sampled_events, sampled_times, target, all_goal_home_proba, all_goal_away_proba, teams, filename):
    #print(event_scores)
    #print(time_scores)
    #print("target:", target)

    target = target.data

    with open('%s/%s' % (EVENTS_DIR, filename), 'w+') as f:
        for batch_idx in range(len(sampled_events)):
            f.write('\n\nNew game: %s -VS- %s\n' % (teams[batch_idx][0], teams[batch_idx][1]))
            f.write('--------------------\n\n')

            goal_home = 0
            goal_away = 0
            goal_home_target = 0
            goal_away_target = 0

            shot_home = 0
            shot_away = 0
            shot_home_target = 0
            shot_away_target = 0

            corner_home = 0
            corner_away = 0
            corner_home_target = 0
            corner_away_target = 0

            foul_home = 0
            foul_away = 0
            foul_home_target = 0
            foul_away_target = 0

            current_time = 0
            current_time_target = 0
            prev_event_type = -1
            prev_event_type_target = -1
            for event_idx in range(len(sampled_events[batch_idx])):
                event_type = sampled_events[batch_idx][event_idx]
                time_type = sampled_times[batch_idx][event_idx]
                event_type_target = target[batch_idx][event_idx, 0]
                time_type_target = target[batch_idx][event_idx, 1]

                if event_type_target == GAME_OVER:
                    break

                if event_type == GOAL_HOME:
                    goal_home += 1
                elif event_type == GOAL_AWAY:
                    goal_away += 1

                if event_type_target == GOAL_HOME:
                    goal_home_target += 1
                elif event_type_target == GOAL_AWAY:
                    goal_away_target += 1

                if event_type == SHOT_HOME:
                    shot_home += 1
                elif event_type == SHOT_AWAY:
                    shot_away += 1

                if event_type_target == SHOT_HOME:
                    shot_home_target += 1
                elif event_type_target == SHOT_AWAY:
                    shot_away_target += 1

                if event_type == CORNER_HOME:
                    corner_home += 1
                elif event_type == CORNER_AWAY:
                    corner_away += 1

                if event_type_target == CORNER_HOME:
                    corner_home_target += 1
                elif event_type_target == CORNER_AWAY:
                    corner_away_target += 1

                if event_type == FOUL_HOME:
                    foul_home += 1
                elif event_type == FOUL_AWAY:
                    foul_away += 1

                if event_type_target == FOUL_HOME:
                    foul_home_target += 1
                elif event_type_target == FOUL_AWAY:
                    foul_away_target += 1

                current_time = get_next_time(current_time, time_type, event_type, prev_event_type)
                current_time_target = get_next_time(current_time_target, time_type_target, event_type_target, prev_event_type_target)
                
                # To remove?
                goal_home_proba = all_goal_home_proba[batch_idx][event_idx]
                goal_away_proba = all_goal_away_proba[batch_idx][event_idx]

                sentence = event_type_to_sentence[event_type] + (" (%.4f - %.4f)" % (goal_home_proba, goal_away_proba))
                sentence_target = event_type_to_sentence[event_type_target.item()]

                f.write("[%d'] %s \t\t[%d'] %s\n" % (current_time, sentence, current_time_target, sentence_target))

                prev_event_type = event_type
                prev_event_type_target = event_type_target

            '''
            game_over_indices = (target[batch_idx, :] == GAME_OVER).nonzero()
            if len(game_over_indices) == 0:
                idx = target.size(1)
            else:
                idx = game_over_indices[0, 0]
            '''

            goal_home_proba = sum(all_goal_home_proba[batch_idx])
            goal_away_proba = sum(all_goal_away_proba[batch_idx])

            f.write("\nPredicted score: %d - %d (%.4f - %.4f), real one was %d - %d\n\n" % (goal_home, goal_away, goal_home_proba, goal_away_proba, goal_home_target, goal_away_target))
            f.write("Predicted: shot home = %d, shot away = %d\n" % (shot_home, shot_away))
            f.write("Real: shot home = %d, shot away = %d\n\n" % (shot_home_target, shot_away_target))
            f.write("Predicted: corner home = %d, corner away = %d\n" % (corner_home, corner_away))
            f.write("Real: corner home = %d, corner away = %d\n\n" % (corner_home_target, corner_away_target))
            f.write("Predicted: foul by home = %d, foul by away = %d\n" % (foul_home, foul_away))
            f.write("Real: foul by home = %d, foul by away = %d\n" % (foul_home_target, foul_away_target))


def output_already_sampled_events_file_no_target(sampled_events, sampled_times, all_goal_home_proba, all_goal_away_proba, teams, filename, aggr=False):
    #print(event_scores)
    #print(time_scores)

    nb_games = len(sampled_events)

    with open('%s/%s' % (EVENTS_DIR, filename), 'w+') as f:
        goals_home = 0
        goals_away = 0
        expected_goals_home = 0
        expected_goals_away = 0
        for batch_idx in range(nb_games):
            f.write('\n\nNew game: %s -VS- %s\n' % (teams[0], teams[1]))
            f.write('--------------------\n\n')

            goal_home = 0
            goal_away = 0
            goal_home_target = 0
            goal_away_target = 0

            shot_home = 0
            shot_away = 0
            shot_home_target = 0
            shot_away_target = 0

            corner_home = 0
            corner_away = 0
            corner_home_target = 0
            corner_away_target = 0

            foul_home = 0
            foul_away = 0
            foul_home_target = 0
            foul_away_target = 0

            current_time = 0
            current_time_target = 0
            prev_event_type = -1
            prev_event_type_target = -1
            for event_idx in range(len(sampled_events[batch_idx])):
                event_type = sampled_events[batch_idx][event_idx]
                time_type = sampled_times[batch_idx][event_idx]

                if event_type == GOAL_HOME:
                    goal_home += 1
                elif event_type == GOAL_AWAY:
                    goal_away += 1

                if event_type == SHOT_HOME:
                    shot_home += 1
                elif event_type == SHOT_AWAY:
                    shot_away += 1

                if event_type == CORNER_HOME:
                    corner_home += 1
                elif event_type == CORNER_AWAY:
                    corner_away += 1

                if event_type == FOUL_HOME:
                    foul_home += 1
                elif event_type == FOUL_AWAY:
                    foul_away += 1

                current_time = get_next_time(current_time, time_type, event_type, prev_event_type)
                
                # To remove?
                goal_home_proba = all_goal_home_proba[batch_idx][event_idx]
                goal_away_proba = all_goal_away_proba[batch_idx][event_idx]

                sentence = event_type_to_sentence[event_type] + (" (%.4f - %.4f)" % (goal_home_proba, goal_away_proba))

                f.write("[%d'] %s\n" % (current_time, sentence))

                prev_event_type = event_type

            '''
            game_over_indices = (target[batch_idx, :] == GAME_OVER).nonzero()
            if len(game_over_indices) == 0:
                idx = target.size(1)
            else:
                idx = game_over_indices[0, 0]
            '''

            goal_home_proba = sum(all_goal_home_proba[batch_idx])
            goal_away_proba = sum(all_goal_away_proba[batch_idx])

            goals_home += goal_home
            goals_away += goal_away
            expected_goals_home += goal_home_proba
            expected_goals_away += goal_away_proba

            f.write("\nPredicted score: %d - %d (%.4f - %.4f)\n\n" % (goal_home, goal_away, goal_home_proba, goal_away_proba))
            f.write("Predicted: shot home = %d, shot away = %d\n" % (shot_home, shot_away))
            f.write("Predicted: corner home = %d, corner away = %d\n" % (corner_home, corner_away))
            f.write("Predicted: foul by home = %d, foul by away = %d\n" % (foul_home, foul_away))

        f.write("\n-----------------------------------------\n\n")
        f.write("Average goals: %.2f - %.2f\n" % (goals_home / nb_games, goals_away / nb_games))
        f.write("Average expected goals: %.2f - %.2f\n" % (expected_goals_home / nb_games, expected_goals_away / nb_games))

def generate_events(event_scores, time_scores):
    event_scores_np = event_scores.data.numpy()
    time_scores_np = time_scores.data.numpy()

    event_tensors = []
    for batch_event_scores_np in event_scores_np:
        batch_event_indices = []
        for elem in batch_event_scores_np:
            event_idx = np.random.choice(np.arange(0, elem.shape[0]), p=elem)
            batch_event_indices.append(int(event_idx))

        batch_event_tensor = torch.LongTensor(batch_event_indices)
        event_tensors.append(batch_event_tensor)

    event_tensor = torch.stack(event_tensors, 0)
    #print(event_tensor.size())

    time_tensors = []
    for batch_time_scores_np in time_scores_np:
        batch_time_indices = []
        for elem in batch_time_scores_np:
            time_idx = np.random.choice(np.arange(0, elem.shape[0]), p=elem)
            batch_time_indices.append(int(time_idx))

        batch_time_tensor = torch.LongTensor(batch_time_indices)
        time_tensors.append(batch_time_tensor)

    time_tensor = torch.stack(time_tensors, 0)
    #print(time_tensor.size())

    #print('-------')
    tensor = torch.stack([event_tensor, time_tensor], 2)
    #print(tensor)

    return tensor

def create_new_events_file(from_filename, new_filename):
    print("*** Loading events file. ***")

    events_df = pd.read_csv(from_filename)
    events_df.loc[(events_df['event_type'] == SHOOT) & ((events_df['event_type2'] == OWN_GOAL) | (events_df['text'].str.startswith('Goal'))), 'event_type'] = GOAL
    events_df = events_df[['id_odsp', 'time', 'event_type', 'side', 'event_team', 'opponent']]

    ids_to_df = {key: events_df.loc[value] for key, value in events_df.groupby("id_odsp").groups.items()}
    games_to_events = {}
    for idd, df in tqdm(ids_to_df.items()):
        side = int(df['side'].iloc[0])
        event_team = df['event_team'].iloc[0]
        opponent = df['opponent'].iloc[0]
        if side == 1:
            home_team = event_team
            away_team = opponent
        elif side == 2:
            home_team = opponent
            away_team = event_team
        else:
            raise Exception('side was not valid')
            
        games_to_events[(idd, home_team, away_team)] = df[['time', 'side', 'event_type']].sort_values(by='time')

    nb_games_training = len(games_to_events)

    print("Nb games training:", nb_games_training)


    ########## Count max number of events in a game ##########

    print("*** Counting max events in a game. ***")

    max_events = 0
    for (idd, home_team, away_team), df in tqdm(games_to_events.items()):
        time_iter = 0
        row_idx = 0
        #df = df.sort_values(by='time')
        rows = [row for idx, row in df.iterrows()]
        nb_events = 0
        while row_idx < df.shape[0] or time_iter <= 90:
            #print(time_iter)
            if row_idx < df.shape[0]:
                row = rows[row_idx]
                side = int(row['side'])
                time = min(int(row['time']), 90) # if 91', 92', etc... -> put 90'
                event_type = int(row['event_type'])
            else:
                time = -1000

            if time == time_iter:
                # Event at this minute
                nb_events += 1
                row_idx += 1
            else:
                if time > 1 + time_iter:
                    # NO_EVENT at this minute
                    nb_events += 1

                time_iter += 1

        if nb_events > max_events:
            max_events = nb_events

    print("Max events in a game:", max_events)


    ########## Create one-hot vectors as X and Y ##########

    print("*** Creating training set. ***")

    all_rows = []
    same = diff = 0

    # Contains all event types (for home and for away team) + no event + match over +
    # 2 others telling if the event happens in the same minute than the last one or not + match over (for time)
    array_size = NB_ALL_EVENTS + NB_ALL_TIMES
    for (idd, home_team, away_team), df in tqdm(games_to_events.items()):
        previous_time = 0
        time_iter = 0
        row_idx = 0
        rows = [row for idx, row in df.iterrows()]
        game_tensors = []
        e = 0
        while e < max_events:
            # Padding for dummy events at the end of the game, so that we have
            # a constant number of events for every game (computed before: max_events).
            if row_idx >= df.shape[0] and time_iter >= 90:
                arr = [idd, home_team, away_team]
                for _ in range(array_size):
                    arr.append(0)

                arr[3 + GAME_OVER] = 1 # match over
                arr[3 + NB_ALL_EVENTS + GAME_NOT_RUNNING_TIME] = 1 # match over

                all_rows.append(arr)
                e += 1
                continue

            if row_idx < df.shape[0]:
                row = rows[row_idx]
                side = int(row['side'])
                time = int(row['time'])
                event_type = int(row['event_type'])
            else:
                time = 1000

            # TODO: Do something about final event?
            if time == time_iter:
                # Event at this minute
                arr = [idd, home_team, away_team]
                for _ in range(array_size):
                    arr.append(0)

                arr[3 + event_type + (side - 1) * NB_EVENT_TYPES] = 1
                if previous_time == time:
                    # Same time than previous event
                    arr[3 + NB_ALL_EVENTS + SAME_TIME_THAN_PREV] = 1
                    same += 1
                else:
                    # Different time
                    arr[3 + NB_ALL_EVENTS + DIFF_TIME_THAN_PREV] = 1
                    diff += 1

                previous_time = time
                row_idx += 1

                all_rows.append(arr)
            else:
                if time > 1 + time_iter:
                    # No event at this minute
                    arr = [idd, home_team, away_team]
                    for _ in range(array_size):
                        arr.append(0)

                    arr[3 + NO_EVENT] = 1 # no event
                    arr[3 + NB_ALL_EVENTS + DIFF_TIME_THAN_PREV] = 1 # thus different time
                    diff += 1
                    all_rows.append(arr)
                else:
                    e -= 1

                time_iter += 1

            e += 1

    print("Same: %.4f" % (same / (same + diff)))
    print("Diff: %.4f" % (diff / (same + diff)))

    labels = ['id_odsp', 'home_team', 'away_team']
    for i in range(array_size):
        labels.append(i)

    new_df = pd.DataFrame.from_records(all_rows, columns=labels)
    new_df.to_csv(new_filename)

def build_k_indices(data_and_targets, k_fold, seed=42):
    """Builds k indices for k-fold."""

    num_row = len(data_and_targets)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)

def train_valid_split_k_fold(train_loader, k_fold, seed=42):
    """
    Splits the data into training and validation sets based on
    the ratio passed in argument.
    """

    data = []
    targets = []
    teams = []

    # Wrap tensors
    for (d, targ, t) in train_loader:
        data.append(Variable(d))
        targets.append(Variable(targ))
        teams.append(t)

    # Create list of k indices
    k_indices = build_k_indices(data, k_fold, seed)

    all_train_data = []
    all_train_targets = []
    all_train_teams = []
    all_valid_data = []
    all_valid_targets = []
    all_valid_teams = []
    for k in range(k_fold):

        # Create the validation fold
        valid_data = [data[i] for i in k_indices[k]]
        valid_targets = [targets[i] for i in k_indices[k]]
        valid_teams = [teams[i] for i in k_indices[k]]

        # Create the training folds
        k_indices_train = np.delete(k_indices, k, 0)
        k_indices_train = k_indices_train.flatten()

        train_data = [data[i] for i in k_indices_train]
        train_targets = [targets[i] for i in k_indices_train]
        train_teams = [teams[i] for i in k_indices_train]

        all_train_data.append(train_data)
        all_train_targets.append(train_targets)
        all_train_teams.append(train_teams)
        all_valid_data.append(valid_data)
        all_valid_targets.append(valid_targets)
        all_valid_teams.append(valid_teams)

    return all_train_data, all_train_targets, all_train_teams, all_valid_data, all_valid_targets, all_valid_teams


def get_end_of_game_idx(target_events):
    game_over_indices = (target_events == GAME_OVER).nonzero()
    if len(game_over_indices) == 0:
        idx = target_events.size(0)
    else:
        idx = game_over_indices[0].data[0].item()

    return idx


def get_during_game_tensors(event_scores, time_scores, target, proba=True, return_end_game_idx=False):
    target_events = target[:, :, 0]
    target_time = target[:, :, 1]

    during_game_target_time_tensors = []
    during_game_time_tensors = []
    during_game_target_events_tensors = []
    during_game_events_tensors = []

    end_game_idx = []
    for batch in range(target_events.size(0)):
        idx = get_end_of_game_idx(target_events[batch])
        end_game_idx.append(idx)
        if proba:
            during_game_events_tensors.append(event_scores[batch, :idx, :])
            during_game_time_tensors.append(time_scores[batch, :idx, :])
        else:
            during_game_events_tensors.append(event_scores[batch, :idx])
            during_game_time_tensors.append(time_scores[batch, :idx])

        during_game_target_time_tensors.append(target_time[batch, :idx])
        during_game_target_events_tensors.append(target_events[batch, :idx])

    target_time_during_game = torch.cat(during_game_target_time_tensors)
    time_during_game = torch.cat(during_game_time_tensors)

    target_events_during_game = torch.cat(during_game_target_events_tensors)
    events_during_game = torch.cat(during_game_events_tensors)

    if return_end_game_idx:
        return events_during_game, target_events_during_game, time_during_game, target_time_during_game, end_game_idx
    else:
        return events_during_game, target_events_during_game, time_during_game, target_time_during_game


def get_during_game_goals(event_proba, target):
    target_events = target[:, :, 0]
    target_time = target[:, :, 1]

    goals_home = []
    goals_away = []
    goals_home_target = []
    goals_away_target = []
    for batch in range(target_events.size(0)):
        idx = get_end_of_game_idx(target_events[batch])

        goal_home = torch.sum(event_proba[batch, :idx, GOAL_HOME]).unsqueeze(0)
        goal_away = torch.sum(event_proba[batch, :idx, GOAL_AWAY]).unsqueeze(0)

        target_events_np = target_events[batch, :idx].data.numpy()
        goal_home_target = np.count_nonzero(target_events_np == GOAL_HOME)
        goal_away_target = np.count_nonzero(target_events_np == GOAL_AWAY)

        goals_home.append(goal_home)
        goals_away.append(goal_away)
        goals_home_target.append(goal_home_target)
        goals_away_target.append(goal_away_target)

    goals_home_tensor = torch.cat(goals_home)
    goals_away_tensor = torch.cat(goals_away)
    goals_home_target_tensor = Variable(torch.FloatTensor(goals_home_target))
    goals_away_target_tensor = Variable(torch.FloatTensor(goals_away_target))

    return goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor


def get_games_proba_from_goals_proba(goals_proba_tensor):
    def poisson_probability(actual, mean):
        # naive:   math.exp(-mean) * mean**actual / factorial(actual)

        # iterative, to keep the components from getting too large or small:
        p = torch.exp(-mean)
        for i in range(actual):
            p = p * mean
            p = p / (i+1)

        return p

    #final_tensor = torch.zeros(goals_proba_tensor.size(0), 3)
    all_home_win_proba = []
    all_home_loss_proba = []
    all_draw_proba = []
    for batch_idx in range(goals_proba_tensor.size(0)):
        expected_home_goals = goals_proba_tensor[batch_idx, 0]
        expected_away_goals = goals_proba_tensor[batch_idx, 1]

        home_win_proba = 0
        draw_proba = 0
        home_loss_proba = 0
        for home_goals in range(MAX_GOAL_FOR_TEAM + 1):
            for away_goals in range(MAX_GOAL_FOR_TEAM + 1):
                proba = poisson_probability(home_goals, expected_home_goals) * poisson_probability(away_goals, expected_away_goals)
                if home_goals > away_goals:
                    home_win_proba += proba
                elif home_goals == away_goals:
                    draw_proba += proba
                else:
                    home_loss_proba += proba

        all_home_win_proba.append(home_win_proba.unsqueeze(0))
        all_home_loss_proba.append(home_loss_proba.unsqueeze(0))
        all_draw_proba.append(draw_proba.unsqueeze(0))

    home_win_tensor = torch.cat(all_home_win_proba)
    home_loss_tensor = torch.cat(all_home_loss_proba)
    draw_tensor = torch.cat(all_draw_proba)
    to_return = torch.stack([home_win_tensor, home_loss_tensor, draw_tensor], 1)

    return to_return


def get_games_results_from_goals(goals_tensor):
    final_tensor = torch.zeros(goals_tensor.size(0)).type(torch.LongTensor)
    for batch_idx in range(goals_tensor.size(0)):
        home_goals = goals_tensor[batch_idx, 0].item()
        away_goals = goals_tensor[batch_idx, 1].item()
        if home_goals > away_goals:
            res = 0
        elif home_goals < away_goals:
            res = 1
        else:
            res = 2

        final_tensor[batch_idx] = res

    return Variable(final_tensor)


def output_event_proba(proba, sampled_events, sampled_times, home_team, away_team):
    new_dir = '%s_VS_%s' % (home_team, away_team)
    pathlib.Path('%s/%s' % (EVENTS_PROBA_DIR, new_dir)).mkdir(parents=True, exist_ok=True) 

    def proba_event_array_to_dict(proba_arr):
        to_return = {}
        for idx in range(len(proba_arr)):
            sentence = event_type_to_sentence[idx]
            to_return[sentence] = proba_arr[idx]

        return to_return

    current_time = 0
    last_events = [GAME_STARTING] * 3
    for event_idx in range(len(sampled_events)):
        current_time = get_next_time(current_time, sampled_times[event_idx], sampled_events[event_idx], last_events[-1])
        proba_dict = proba_event_array_to_dict(proba[event_idx])
        plot_events_proba(proba_dict, time=current_time, last_events=[event_type_to_sentence[e] for e in last_events], filename="%s/test_%d.pdf" % (new_dir, event_idx))
        last_events[0] = last_events[1]
        last_events[1] = last_events[2]
        last_events[2] = sampled_events[event_idx]

def output_time_proba(proba, sampled_events, sampled_times, home_team, away_team):
    new_dir = 'times_%s_VS_%s' % (home_team, away_team)
    pathlib.Path('%s/%s' % (EVENTS_PROBA_DIR, new_dir)).mkdir(parents=True, exist_ok=True) 

    def proba_time_array_to_dict(proba_arr):
        to_return = {}
        for idx in range(len(proba_arr)):
            sentence = time_type_to_sentence[idx]
            to_return[sentence] = proba_arr[idx]

        return to_return

    current_time = 0
    last_events = [GAME_STARTING] * 3
    for event_idx in range(len(sampled_events)):
        current_time = get_next_time(current_time, sampled_times[event_idx], sampled_events[event_idx], last_events[-1])
        proba_dict = proba_time_array_to_dict(proba[event_idx])
        plot_events_proba(proba_dict, time=current_time, last_events=[event_type_to_sentence[e] for e in last_events], filename="%s/test_%d.pdf" % (new_dir, event_idx))
        last_events[0] = last_events[1]
        last_events[1] = last_events[2]
        last_events[2] = sampled_events[event_idx]


def get_hyperparams_filename(filename, batch_size=None, learning_rate=None, hidden_layer_size1=None, hidden_layer_size2=None, dropout_rate=None):
    if batch_size is None:
        batch_size = CHOSEN_BATCH_SIZE

    if learning_rate is None:
        learning_rate = CHOSEN_LEARNING_RATE

    if hidden_layer_size1 is None:
        hidden_layer_size1 = CHOSEN_HIDDEN_LAYER_SIZES[0]

    if hidden_layer_size2 is None:
        hidden_layer_size2 = CHOSEN_HIDDEN_LAYER_SIZES[1]

    if dropout_rate is None:
        dropout_rate = CHOSEN_DROPOUT_RATE

    tab = filename.split('.')
    name = '.'.join(tab[:-1])
    extension = tab[-1]
    return "%s_%d_%.6f_(%d_%d)_%.1f.%s" % (name, batch_size, learning_rate, hidden_layer_size1, hidden_layer_size2, dropout_rate, extension)


def get_dated_filename(filename):
    tab = filename.split('.')
    name = '.'.join(tab[:-1])
    extension = tab[-1]
    return "%s_%s.%s" % (name, time.strftime("%Y%m%d-%H%M"), extension)

if __name__ == "__main__":
    create_new_events_file('../data/football-events/events.csv', '../data/football-events/new_events.csv')
