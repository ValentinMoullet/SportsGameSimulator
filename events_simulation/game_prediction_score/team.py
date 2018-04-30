import numpy as np
import pandas as pd


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


def get_teams_to_idx(df):
    teams_to_idx = {}
    for i, team in enumerate(get_teams(df)):
        teams_to_idx[team] = i

    return teams_to_idx