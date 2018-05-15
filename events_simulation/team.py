import torch
from torch.autograd import Variable
import pandas as pd

import presaved_models
from game_prediction.team import *


GAME_INFO_FILENAME = '../data/football-events/ginf.csv'
GAME_INFO_DF = pd.read_csv(GAME_INFO_FILENAME)


def get_teams(
    df,
    home_team_col_name='ht',
    away_team_col_name='at'):
    """
    Gets the home and away teams and ignore the ones that do not appear at
    least once as home and away team.

    Args:
        df: The DataFrame containing all the games.

    Returns:
        A numpy arrays containing the names of teams.
    """

    home_teams = df[home_team_col_name].unique()
    away_teams = df[away_team_col_name].unique()

    # Remove teams that do not appear in both home and away situations
    to_remove = []
    for i,team in enumerate(away_teams):
        if not team in home_teams:
            to_remove.append(i)

    teams = np.delete(away_teams, to_remove)

    return teams
    

def get_teams_caracteristics(teams):
    """
    Returns a tensor that describes the game between 'home_team' and
    'away_team' for each entry in 'teams'. This basically returns the
    last hidden layer of the network used for predicting the outcome of
    games, when fed with those teams.
    """

    layers = []
    for home_team, away_team in teams:
        # Get the league those teams are from
        league = get_teams_league(home_team, away_team)

        # Fetch the pre-saved model
        model = presaved_models.get_model(league)
        model.eval()

        # Filter and keep only the corresponding league
        game_info_df = GAME_INFO_DF[GAME_INFO_DF['league'] == league]

        teams_to_idx = get_teams_to_idx(game_info_df)

        X_home = torch.zeros(len(teams_to_idx))
        X_away = torch.zeros(len(teams_to_idx))
        X_home[teams_to_idx[home_team]] = 1
        X_away[teams_to_idx[away_team]] = 1

        input_tensor = Variable(torch.cat([X_home, X_away]).unsqueeze(0))

        #print("Input tensor:", input_tensor)

        last_layer = model.get_last_layer(input_tensor)

        #print("Last layer:", last_layer)

        layers.append(last_layer)

    to_return = torch.stack(layers, 1)

    #print("To return:", to_return)

    return to_return

def get_teams_league(*teams):
    if len(teams) == 0:
        raise ValueError("Cannot get the league of 0 teams.")

    leagues = []
    for team in teams:
        if is_team_from('D1', team):
            leagues.append('D1')
        elif is_team_from('E0', team):
            leagues.append('E0')
        elif is_team_from('F1', team):
            leagues.append('F1')
        elif is_team_from('I1', team):
            leagues.append('I1')
        elif is_team_from('SP1', team):
            leagues.append('SP1')
        else:
            raise ValueError("'%s' is not from any known league." % team)

    if not leagues.count(leagues[0]) == len(leagues):
        raise ValueError("Teams are not from the same league.")

    return leagues[0]

def is_team_from(league, team):
    game_info_df = GAME_INFO_DF[GAME_INFO_DF['league'] == league]
    return team in list(game_info_df['ht'].unique()) or team in list(game_info_df['at'].unique())

