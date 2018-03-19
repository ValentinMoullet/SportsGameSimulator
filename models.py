from sklearn.decomposition import PCA, NMF, KernelPCA
from sklearn.neural_network import MLPRegressor
from sklearn import svm

from parameters import *
from preprocessing import *
from utils import *


def NMF_grid_search(
    X,
    k, 
    inits=['random'],
    solvers=['cd'],
    beta_losses=['frobenius'],
    max_iters=[200],
    alphas=[0],
    l1_ratios=[0],
    shuffles=[False]):
    """
    
    Args:

    Returns:

    """

    best_W = None
    best_H = None
    best_model = None

    for init in inits:
        for solver in solvers:
            for beta_loss in beta_losses:
                for max_iter in max_iters:
                    for alpha in alphas:
                        for l1_ratio in l1_ratios:
                            for shuffle in shuffles:
                                model = NMF(
                                    n_components=k,
                                    init=init,
                                    solver=solver,
                                    beta_loss=beta_loss,
                                    max_iter=max_iter,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    shuffle=shuffle)
                                W = model.fit_transform(X)
                                H = model.components_

                                if best_model is None or best_model.reconstruction_err_ > model.reconstruction_err_:
                                    best_W = W
                                    best_H = H
                                    best_model = model

    return best_W, best_H, best_model


def get_NMF_scores(
    X,
    max_k,
    game_info_train_df,
    game_info_test_df,
    home_teams,
    away_teams,
    estimator,
    parameters,
    classifying=True,
    scoring=None):

    k_train_test_scores = []

    for k in range(2, max_k+1, 2):
        W, H, model = NMF_grid_search(
            X,
            k,
            inits=NMF_INITS,
            solvers=NMF_SOLVERS,
            beta_losses=NMF_BETA_LOSSES,
            max_iters=NMF_MAX_ITERS,
            alphas=NMF_ALPHAS,
            l1_ratios=NMF_L1_RATIOS,
            shuffles=NMF_SHUFFLES)

        # For training
        games_latent_train_df = create_latent_df(game_info_train_df, W, H, home_teams, away_teams)
        data_labels = games_latent_train_df.columns[1:]
        target_label = games_latent_train_df.columns[0]
        data, target = get_data_and_target(games_latent_train_df, data_labels, target_label)

        # For test
        games_latent_test_df = create_latent_df(game_info_test_df, W, H, home_teams, away_teams)
        data_test, target_test = get_data_and_target(games_latent_test_df, data_labels, target_label)

        # Setup
        if classifying:
            games_latent_train_df = continuous_to_win_draw_loss_df(games_latent_train_df, target_label)
            games_latent_test_df = continuous_to_win_draw_loss_df(games_latent_test_df, target_label)
            if BALANCE:
                games_latent_train_df = balance_df(games_latent_train_df, target_label)
            
            data, target = get_data_and_target(games_latent_train_df, data_labels, target_label)
            data_test, target_test = get_data_and_target(games_latent_test_df, data_labels, target_label)

        # Create model
        model = grid_search_CV_report(estimator, data, target, parameters, cv=3, verbose=False, scoring=scoring)

        #y_pred = model.predict(data_test)
        #y_true = target_test
        #plot_confusion_matrices(y_true, y_pred)

        #print("Training score:", model.best_score_)
        #print("Test score:", model.score(data_test, target_test))
        #print('')

        if classifying:
            training_score = model.best_score_
            test_score = model.score(data_test, target_test)
        else:
            training_score = reg_score(target, model.predict(data))
            test_score = reg_score(target_test, model.predict(data_test))

        print("For k = %d, best test score: %.5f, best params: %s" % (k, test_score, model.best_params_))
        k_train_test_scores.append((k, model.best_score_, test_score))

    return k_train_test_scores

