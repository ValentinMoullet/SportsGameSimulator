import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.autograd import Variable

from parameters import *
from game_prediction.parameters import CHOSEN_BATCH_SIZE, CHOSEN_LEARNING_RATE, CHOSEN_HIDDEN_LAYER_SIZES, CHOSEN_DROPOUT_RATE


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
                goal_home_proba = event_scores[batch_idx, event_idx, GOAL_HOME].data[0]
                goal_away_proba = event_scores[batch_idx, event_idx, GOAL_AWAY].data[0]

                sentence = event_type_to_sentence[event_type] + (" (%.4f - %.4f)" % (goal_home_proba, goal_away_proba))
                sentence_target = event_type_to_sentence[event_type_target]

                f.write("[%d'] %s \t\t[%d'] %s\n" % (current_time, sentence, current_time_target, sentence_target))

                prev_event_type = event_type
                prev_event_type_target = event_type_target

            game_over_indices = (target[batch_idx, :] == GAME_OVER).nonzero()
            if len(game_over_indices) == 0:
                idx = target.size(1)
            else:
                idx = game_over_indices[0, 0]

            goal_home_proba = torch.sum(event_scores[batch_idx, :idx, GOAL_HOME]).data[0]
            goal_away_proba = torch.sum(event_scores[batch_idx, :idx, GOAL_AWAY]).data[0]

            f.write("\nPredicted score: %d - %d (%.2f - %.2f), real one was %d - %d\n\n" % (goal_home, goal_away, goal_home_proba, goal_away_proba, goal_home_target, goal_away_target))
            f.write("Predicted: shot home = %d, shot away = %d\n" % (shot_home, shot_away))
            f.write("Real: shot home = %d, shot away = %d\n\n" % (shot_home_target, shot_away_target))
            f.write("Predicted: corner home = %d, corner away = %d\n" % (corner_home, corner_away))
            f.write("Real: corner home = %d, corner away = %d\n\n" % (corner_home_target, corner_away_target))
            f.write("Predicted: foul by home = %d, foul by away = %d\n" % (foul_home, foul_away))
            f.write("Real: foul by home = %d, foul by away = %d\n" % (foul_home_target, foul_away_target))

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

    # Contains all event types (for home and for away team) + no event + match over +
    # 2 others telling if the event happens in the same minute than the last one or not + match over (for time)
    array_size = NB_EVENT_TYPES * 2 + 2 + 3
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

                arr[3 + 2 * NB_EVENT_TYPES + 1] = 1 # match over
                arr[-1] = 1 # match over

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
                    arr[-3] = 1
                else:
                    # Different time
                    arr[-2] = 1

                previous_time = time
                row_idx += 1

                all_rows.append(arr)
            else:
                if time > 1 + time_iter:
                    # No event at this minute
                    arr = [idd, home_team, away_team]
                    for _ in range(array_size):
                        arr.append(0)

                    arr[3 + 2 * NB_EVENT_TYPES] = 1 # no event
                    arr[-2] = 1 # thus different time
                    all_rows.append(arr)
                else:
                    e -= 1

                time_iter += 1

            e += 1

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

    return all_train_data, all_train_targets, all_train_teams, all_valid_data, all_valid_targets, all_train_teams


def get_during_game_tensors(event_scores, time_scores, target, proba=True):
    target_events = target[:, :, 0]
    target_time = target[:, :, 1]

    during_game_target_time_tensors = []
    during_game_time_tensors = []
    during_game_target_events_tensors = []
    during_game_events_tensors = []
    for batch in range(target_events.size(0)):
        game_over_indices = (target_events[batch, :] == GAME_OVER).nonzero()
        if len(game_over_indices) == 0:
            idx = target_events.size(1)
        else:
            idx = game_over_indices[0, 0].data[0]

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

    return events_during_game, target_events_during_game, time_during_game, target_time_during_game


def get_during_game_goals(event_proba, time_proba, target):
    target_events = target[:, :, 0]
    target_time = target[:, :, 1]

    goals_home = []
    goals_away = []
    goals_home_target = []
    goals_away_target = []
    for batch in range(target_events.size(0)):
        game_over_indices = (target_events[batch, :] == GAME_OVER).nonzero()
        if len(game_over_indices) == 0:
            idx = target_events.size(1)
        else:
            idx = game_over_indices[0, 0].data[0]

        goal_home = torch.sum(event_proba[batch, :idx, GOAL_HOME])
        goal_away = torch.sum(event_proba[batch, :idx, GOAL_AWAY])

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

#create_new_events_file('../data/football-events/events.csv', '../data/football-events/new_events.csv')