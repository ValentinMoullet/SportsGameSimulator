import sys, os, glob

from model import *
from utils import *
from parameters import *
from plot import *

if len(sys.argv) < 3 or len(sys.argv) > 4:
    raise ValueError("Wrong number of arguments, should give home_team and away_team, and optionally the number of sample to do.")

home_team = sys.argv[1]
away_team = sys.argv[2]
n_samples = 1
if len(sys.argv) == 4:
    n_samples = int(sys.argv[3])

teams = [home_team, away_team]
teams_tensor = get_teams_caracteristics([teams])

# Load latest model
model = LSTMEvents(hidden_dim=40, event_types_size=NB_ALL_EVENTS, time_types_size=NB_ALL_TIMES, num_layers=1, batch_size=BATCH_SIZE, learning_rate=0.01)
all_saved_models = glob.glob("%s/*.pt" % MODELS_DIR)
latest_model_file = max(all_saved_models, key=os.path.getctime)
model.load_state_dict(torch.load(latest_model_file))

all_sampled_events = []
all_sampled_times = []
all_goal_home_proba = []
all_goal_away_proba = []
for s in range(n_samples):
    sampled_events, sampled_times, event_proba, time_proba = model.sample([teams], return_proba=True)

    goal_home_proba = [e[GOAL_HOME] for e in event_proba]
    goal_away_proba = [e[GOAL_AWAY] for e in event_proba]

    all_sampled_events.append(sampled_events)
    all_sampled_times.append(sampled_times)
    all_goal_home_proba.append(goal_home_proba)
    all_goal_away_proba.append(goal_away_proba)

output_event_proba(event_proba, sampled_events, sampled_times, teams[0], teams[1])
output_time_proba(time_proba, sampled_events, sampled_times, teams[0], teams[1])

output_already_sampled_events_file_no_target(all_sampled_events, all_sampled_times, all_goal_home_proba, all_goal_away_proba, teams, get_dated_filename('sample.txt'))

events_count_dict = count_events([e for sublist in all_sampled_events for e in sublist])
plot_events_count(events_count_dict, {}, get_dated_filename('sample.pdf'))

times_count_dict = count_times([e for sublist in all_sampled_times for e in sublist])
plot_events_count(times_count_dict, {}, get_dated_filename('times_sample.pdf'))
