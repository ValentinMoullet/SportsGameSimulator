import sys, os, argparse
from tqdm import tqdm
tqdm.monitor_interval = 0

from model import *
from utils import *
from parameters import *
from plot import *


def sample_n_times_events(teams, n):
    model = load_latest_model()

    all_sampled_events = []
    all_sampled_times = []
    for s in range(n):
        sampled_events, sampled_times = model.sample([teams])

        all_sampled_events.append(sampled_events)
        all_sampled_times.append(sampled_times)

    return all_sampled_events, all_sampled_times

def sample_n_times(teams, n, output=False):
    #teams_tensor = get_teams_caracteristics([teams])

    model = load_latest_model()

    all_sampled_events = []
    all_sampled_times = []
    all_goal_home_proba = []
    all_goal_away_proba = []
    home_wins = 0
    away_wins = 0
    draws = 0
    for s in range(n):
        sampled_events, sampled_times, event_proba, time_proba = model.sample([teams], return_proba=True)

        goal_home = sampled_events.count(GOAL_HOME)
        goal_away = sampled_events.count(GOAL_AWAY)
        if goal_home > goal_away:
            home_wins += 1
        elif goal_home < goal_away:
            away_wins += 1
        else:
            draws += 1

        goal_home_proba = [e[GOAL_HOME] for e in event_proba]
        goal_away_proba = [e[GOAL_AWAY] for e in event_proba]

        all_sampled_events.append(sampled_events)
        all_sampled_times.append(sampled_times)
        all_goal_home_proba.append(goal_home_proba)
        all_goal_away_proba.append(goal_away_proba)

    if output:
        output_event_proba(event_proba, sampled_events, sampled_times, teams[0], teams[1])
        output_time_proba(time_proba, sampled_events, sampled_times, teams[0], teams[1])

        output_already_sampled_events_file_no_target(all_sampled_events, all_sampled_times, all_goal_home_proba, all_goal_away_proba, teams, get_dated_filename('sample.txt'))

        events_count_dict = count_events([e for sublist in all_sampled_events for e in sublist])
        plot_events_count(events_count_dict, {}, get_dated_filename('sample.pdf'))

        times_count_dict = count_times([e for sublist in all_sampled_times for e in sublist])
        plot_events_count(times_count_dict, {}, get_dated_filename('times_sample.pdf'))

    return home_wins / n, away_wins / n, draws / n


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise ValueError("Wrong number of arguments, should give home_team and away_team, and optionally the number of sample to do.")

    home_team = sys.argv[1]
    away_team = sys.argv[2]
    n_samples = 1
    if len(sys.argv) == 4:
        n_samples = int(sys.argv[3])

    teams = [home_team, away_team]

    #home_win_proba, away_win_proba, draws_proba = sample_n_times(teams, n_samples)
    print(sample_n_times(teams, n_samples, output=True))
