import numpy as np
import pandas as pd
from utils import *
from parameters import *


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


def compute_confrontation_matrix(
    df,
    home_teams,
    away_teams,
    home_team_col_name='ht',
    away_team_col_name='at',
    home_score_col_name='fthg',
    away_score_col_name='ftag'):
    """
    Compute the matrix NxN (where N is the number of teams) where the values
    are numbers between 0 and 1, higher meaning the home team won more in avg,
    lower meaning the away team won easily in avg, 0.5 meaning they draw in avg.
    Values will ba NaN if the teams never played against in each other.

    Args:
        df: DataFrame containing the games.
        home_teams: Numpy array containing home teams.
        away_teams: Numpy array containing away teams.
        home_team_col_name: Name of the column containing home team.
        away_team_col_name: Name of the column containing away team.
        home_score_col_name: Name of the column containing home team's score.
        away_score_col_name: Name of the column containing away team's score.

    Returns:
        The confrontation matrix.
    """

    # Ensure that both home teams and away teams are the same
    assert(np.array_equal(home_teams, away_teams))

    # Get a mapping from team to index
    teams_to_idx = {}
    for i, team in enumerate(home_teams):
        teams_to_idx[team] = i

    X = np.identity(home_teams.size)
    counts = np.zeros((home_teams.size, away_teams.size))

    for index, row in df.iterrows():
        home_team = row['ht']
        away_team = row['at']
        home_score = row['fthg']
        away_score = row['ftag']
        if not home_team in teams_to_idx or not away_team in teams_to_idx:
            continue
            
        prob_home_team_win = sigmoid(home_score - away_score)
        home_idx = teams_to_idx[home_team]
        away_idx = teams_to_idx[away_team]
        counts[home_idx, away_idx] += 1
        X[home_idx, away_idx] += prob_home_team_win

    np.seterr(divide='ignore')
    X = np.divide(X, counts)
    np.seterr(divide='raise')
    X[X == np.inf] = np.nan

    return X


def get_sorted_confrontation_matrix(conf_matrix):
    """
    Sort the rows and columns of the confrontation matrix.

    Args:
        conf_matrix: The confrontation matrix.

    Returns:
        The sorted confrontation matrix; first sort row (by sum), then sort
        columns (by sum). NaN values are evaluated as if they were the worst:
        0.0 for rows, 1.0 for columns.
    """

    sorted_conf_matrix = conf_matrix.copy()

    # Sort confrontation matrix by rows' sum, then by columns' sum
    temp_conf_matrix = conf_matrix.copy()
    temp_conf_matrix[np.isnan(temp_conf_matrix)] = 0.0
    sorted_conf_matrix = sorted_conf_matrix[np.argsort(-temp_conf_matrix.sum(axis=1))]

    temp_conf_matrix = conf_matrix.copy()
    temp_conf_matrix[np.isnan(temp_conf_matrix)] = 1.0
    sorted_conf_matrix = sorted_conf_matrix[:, np.argsort(-temp_conf_matrix.sum(axis=0))]

    return sorted_conf_matrix


def get_sorted_teams(teams, conf_matrix, home):
    """
    Sort the 'teams' received in arguments from worst to best performing as
    home or away (see 'home' parameter) based on the 'conf_matrix'.

    Args:
        teams: Numpy array of teams.
        conf_matrix: NxN confrontation matrix (N = nb teams).
        home: Boolean telling if those are the home teams or not.

    Returns:
        The teams sorted from the worst to the best, based on the
        confrontation matrix.
    """

    assert(conf_matrix.shape[0] == conf_matrix.shape[1] and conf_matrix.shape[0] == teams.size)

    # Sort by sum of col -> for away_teams
    teams_with_sum = []
    temp_conf_matrix = conf_matrix.copy()

    # Replace NaN by worst value (1.0 or 0.0, depending if home teams or not)
    if home:
        temp_conf_matrix[np.isnan(temp_conf_matrix)] = 0.0
    else:
        temp_conf_matrix[np.isnan(temp_conf_matrix)] = 1.0

    for i in range(teams.size):
        if home:
            ssum = np.sum(temp_conf_matrix[i, :])
        else:
            ssum = np.sum(temp_conf_matrix[:, i])

        teams_with_sum.append((teams[i], ssum))

    teams_with_sum.sort(key=lambda pair: -pair[1])
    sorted_teams = list(map(lambda pair: pair[0], teams_with_sum))

    return sorted_teams


def replace_nan_by_mean(X):
    """
    Fill the NaNs with the mean of the matrix.

    Args:
        X: A matrix.

    Returns:
        The same matrix with all NaNs replaced by the mean of entire matrix.
    """

    temp_X = X.copy()
    temp_X[np.isnan(temp_X)] = np.nanmean(temp_X)
    return temp_X


def get_bookies_pred(
    df,
    odd_home_col_name='odd_h',
    odd_draw_col_name='odd_d',
    odd_away_col_name='odd_a',
    classes_proba=False):
    """
    
    """

    to_return = []
    for idx, row in df.iterrows():
        odd_h = row[odd_home_col_name]
        odd_d = row[odd_draw_col_name]
        odd_a = row[odd_away_col_name]

        if classes_proba:
            #to_return.append([0, 0, 0])
            to_return.append([1/odd_a, 1/odd_d, 1/odd_h])
        else:
            if odd_h < odd_d and odd_h < odd_a:
                to_return.append(1)
            elif odd_a < odd_h and odd_a < odd_d:
                to_return.append(-1)
            else:
                to_return.append(0)
            
    return to_return 


def get_bookies_accuracy(
    df,
    score_home_col_name='fthg',
    score_away_col_name='ftag',
    odd_home_col_name='odd_h',
    odd_draw_col_name='odd_d',
    odd_away_col_name='odd_a'):
    """
    Get the bookmakers classification accuracy (home win, draw, away win) over
    all the games in 'df'.

    Args:
        df: DataFrame containing the games.
        score_home_col_name: Column name for score of home team.
        score_away_col_name: Column name for score of away team.
        odd_home_col_name: Column name for odd of home team.
        odd_draw_col_name: Column name for odd of draw team.
        odd_away_col_name: Column name for odd of away team.

    Returns:
        The ratio of correctly predicted games.
    """

    total_correct = 0
    for idx, row in df.iterrows():
        fthg = row[score_home_col_name]
        ftag = row[score_away_col_name]
        odd_h = row[odd_home_col_name]
        odd_d = row[odd_draw_col_name]
        odd_a = row[odd_away_col_name]
        if odd_h < odd_d and odd_h < odd_a and fthg > ftag:
            total_correct+=1
            
        if odd_d < odd_h and odd_d < odd_a and fthg == ftag:
            total_correct+=1
            
        if odd_a < odd_d and odd_a < odd_d and fthg < ftag:
            total_correct+=1
            
    return total_correct / df.shape[0]


def create_latent_df(df, W, H, home_teams, away_teams, mix_all_features=False):
    """
    Create the DataFrame containing the games with the latent features of the
    corresponding teams. This DataFrame can be used for predicting outcome
    of games.

    Args:
        df: The DataFrame containing the games.
        W: The matrix containing latent features of home teams.
        H: The matrix containing latent features of away teams.
        home_teams: Numpy array containing home teams.
        away_teams: Numpy array containing away teams.
        mix_all_features: Boolean to put home AND away features for both teams.

    Returns:
        DataFrame containing every game with latent features of teams involved.
    """

    assert(W.shape[1] == H.shape[0])

    # Computing number of latent features
    k = W.shape[1]

    # Mapping (home and away) teams to their latent features
    home_teams_to_latent = {}
    for i, team_name in enumerate(home_teams):
        home_teams_to_latent[team_name] = W[i]

    away_teams_to_latent = {}
    for i, team_name in enumerate(away_teams):
        away_teams_to_latent[team_name] = H[:,i]

    # Creating rows of new DataFrame
    games = []
    for idx, row in df.iterrows():
        home_team = row['ht']
        away_team = row['at']
        home_goals = row['fthg']
        away_goals = row['ftag']
        season = row['season']
        if not home_team in home_teams or not away_team in away_teams:
            continue

        prob_home_team_win = sigmoid(home_goals - away_goals)

        to_append = []
        to_append.append([prob_home_team_win])
        to_append.append(home_teams_to_latent[home_team])
        if mix_all_features:
            to_append.append(away_teams_to_latent[home_team])
            to_append.append(home_teams_to_latent[away_team])
        to_append.append(away_teams_to_latent[away_team])

        # Flatten list
        to_append = [item for sublist in to_append for item in sublist]
        games.append(to_append)

    # Create column names
    labels = ['y']
    for i in range(1, k+1):
        labels.append('home_team_home_latent_%d' % i)
    if mix_all_features:
        for i in range(1, k+1):
            labels.append('home_team_away_latent_%d' % i)
        for i in range(1, k+1):
            labels.append('away_team_home_latent_%d' % i)  
    for i in range(1, k+1):
        labels.append('away_team_away_latent_%d' % i)
        
    games_latent_df = pd.DataFrame.from_records(games, columns=labels)
    return games_latent_df


def get_target(df, target_is_classes=True):
    to_return = []
    for idx, row in df.iterrows():
        home_goals = row['fthg']
        away_goals = row['ftag']
        if home_goals > away_goals:
            to_return.append(1)
        elif home_goals < away_goals:
            to_return.append(-1)
        else:
            to_return.append(0)

    return to_return


def get_data_and_target(df, data_labels, target_label):
    """
    
    """

    data = df[data_labels].values
    target = df[[target_label]].values.ravel()

    return data, target


def split_in_training_and_test_data(df, shuffle=True):
    """
    Split the data into training and test data.

    Args:
        df: DataFrame containing all the games with their features.
        shuffle: Boolean to shuffle or not the data before splitting.

    Returns:
        The training and test data.
    """

    test_set_size = int(df.shape[0] * TEST_SET_RATIO)

    if shuffle:
        msk = np.random.rand(df.shape[0]) < 1 - TEST_SET_RATIO
        training_data = df[msk]
        test_data = df[~msk]
    else:
        training_data = df[:-test_set_size]
        test_data = df[-test_set_size:]

    return training_data, test_data


def continuous_to_win_draw_loss_df(df, y_label='y'):
    """
    Take a DataFrame with the labels being values between 0 and 1 explaining
    the likelihood of the home team winning, and returns the same DataFrame
    with the labels transformed into 1, 0 or -1 (home win, draw, away win);
    0.5 in the likelihood of winning means draw.

    Args:
        df: The DataFrame containing data, with one column being the labels.
        y_label: The name of the column containing the labels.

    Returns:
        DataFrame with the labels column replaced to be 1, 0, -1:
        home win, draw, away win respectively.
    """

    to_return_df = df.copy()
    to_return_df.loc[to_return_df.y < 0.5, y_label] = -1
    to_return_df.loc[to_return_df.y > 0.5, y_label] = 1
    to_return_df.loc[to_return_df.y == 0.5, y_label] = 0
    to_return_df.y = to_return_df.y.astype(int)

    return to_return_df


def balance_df(df, target_label='y'):
    """
    
    """

    min_size = df[target_label].value_counts().min()

    balanced_df = pd.DataFrame()
    for cls in df[target_label].unique():
        cls_games = df[df[target_label] == cls]
        balanced_df = pd.concat([balanced_df, cls_games[:min_size]])

    return balanced_df


