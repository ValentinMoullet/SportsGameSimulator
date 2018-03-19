import math
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix, log_loss

from parameters import *

def grid_search_CV_report(estimator, data, target, parameters, cv=3, scoring=None, verbose=True):
    """
    
    """

    if verbose:
        print("Tuning hyper-parameters...")

    if scoring is None:
        model = GridSearchCV(estimator, parameters, cv=cv)
    else:
        model = GridSearchCV(estimator, parameters, cv=cv, scoring=scoring)

    model.fit(data, target)

    if verbose:
        print("Done.")
        print("Best parameters set (out of %d):" % len(model.cv_results_['mean_test_score']))
        print()
        print(model.best_params_)
        print("Score:", model.best_score_)
        print()
        print("Grid scores on development set (top 10):")
        print()

        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        params = model.cv_results_['params']
        means_stds_and_params = list(zip(means, stds, params))
        means_stds_and_params.sort(key=lambda x: -x[0])
        means, stds, params = list(zip(*means_stds_and_params[:min(len(means), 10)]))

        for mean, std, params in zip(means, stds, params):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    '''
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    '''

    return model


def reg_score(ground_truth, predictions):
    """

    """

    predictions = reg_to_classes(predictions)
    score = 0
    for i, value in enumerate(ground_truth):
        if value == 0.5 and predictions[i] == 0:
            score += 1
        elif value > 0.5 and predictions[i] == 1:
            score += 1
        elif value < 0.5 and predictions[i] == -1:
            score += 1

    return score / len(ground_truth)


def reg_score_function():
    """

    """

    return make_scorer(reg_score, greater_is_better=True)


def reg_to_classes(arr):
    """

    """

    to_return = []
    for value in arr:
        if value < REG_DRAW_MIN:
            to_return.append(-1)
        elif value <= REG_DRAW_MAX:
            to_return.append(0)
        else:
            to_return.append(1)

    return to_return


def sigmoid_values_to_classes(arr):
    """

    """

    to_return = []
    for value in arr:
        if value == 0.5:
            to_return.append(0)
        elif value < 0.5:
            to_return.append(-1)
        else:
            to_return.append(1)

    return to_return


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:


    Returns:

    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrices(y_true, y_pred):
    """

    """

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=CLASSES_NAMES,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=CLASSES_NAMES, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def sigmoid(x):
    """
    Compute the sigmoid of the given number.

    Args:
        The number we want the sigmoid of.

    Returns:
        The sigmoid of the given number.
    """

    return 1 / (1 + math.exp(-x))


def log_loss_proba(y_true, y_pred, labels=None):
    return math.exp(-log_loss(y_true, y_pred, labels=labels))

