import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.autograd import Variable

from parameters import *


########## Create events mapping ##########

event_type_to_sentence = {}
sides = ['home team', 'away team']
for i, side in enumerate(sides):
    for event_type in range(NB_EVENT_TYPES):
        if event_type == 0:
            sentence = "Goal for %s." % side
        elif event_type == 1:
            sentence = "Shot from %s." % side
        elif event_type == 2:
            sentence = "Corner for %s." % side
        elif event_type == 3:
            sentence = "Foul from %s." % side
        elif event_type == 4:
            sentence = "Yellow card for %s." % side
        elif event_type == 5:
            sentence = "Second yellow card for %s." % side
        elif event_type == 6:
            sentence = "Red card for %s." % side
        elif event_type == 7:
            sentence = "Substitution by %s." % side
        elif event_type == 8:
            sentence = "Free kick won by %s." % side
        elif event_type == 9:
            sentence = "Offside by %s." % side
        elif event_type == 10:
            sentence = "Hand ball by %s." % side
        elif event_type == 11:
            sentence = "Penalty conceded by %s." % side

        event_type_to_sentence[event_type + i * NB_EVENT_TYPES] = sentence

event_type_to_sentence[NB_EVENT_TYPES * 2] = "No event this minute."
event_type_to_sentence[NB_EVENT_TYPES * 2 + 1] = "Game is over."


def get_next_time_from_type(current_time, time_type, event_type):
    '''
    if event_type == NB_EVENT_TYPES * 2:
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

def output_events_file(event_scores, time_scores, target, filename):
    #print(event_scores)
    #print(time_scores)
    #print("target:", target)

    target = target.data

    generated_events = generate_events(event_scores, time_scores)

    with open('%s/%s' % (EVENTS_DIR, filename), 'w+') as f:
        for batch_idx in range(generated_events.size(0)):
            f.write('\nNew game:\n')
            f.write('--------------------\n\n')
            current_time = 0
            current_time_target = 0
            for event_idx in range(generated_events.size(1)):
                event_type = generated_events[batch_idx, event_idx, 0]
                time_type = generated_events[batch_idx, event_idx, 1]
                event_type_target = target[batch_idx, event_idx, 0]
                time_type_target = target[batch_idx, event_idx, 1]

                current_time = get_next_time_from_type(current_time, time_type, event_type)
                current_time_target = get_next_time_from_type(current_time_target, time_type_target, event_type_target)
                
                sentence = event_type_to_sentence[event_type]
                sentence_target = event_type_to_sentence[event_type_target]

                f.write("[%d'] %s \t\t[%d'] %s\n" % (current_time, sentence, current_time_target, sentence_target))


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
                time = int(row['time'])
                event_type = int(row['event_type'])
            else:
                time = 1000

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
            if row_idx >= df.shape[0] and time_iter > 90:
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


#create_new_events_file('../data/football-events/events.csv', '../data/football-events/new_events.csv')
