#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Plot functions to create figures.
"""

import matplotlib
matplotlib.use('agg') # Workaround for using it without monitors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

from parameters import *


def plot_history(histories, filename, title="", verbose=False, thick=None):
    epochs = [i for i in range(len(histories[0][1]))]

    for i in range(len(histories)):
        if not thick is None and i in thick:
            plt.plot(epochs, histories[i][1], label=histories[i][0], linewidth=5.0)
        else:
            plt.plot(epochs, histories[i][1], label=histories[i][0])

    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.title(title)
    leg = plt.legend(loc='upper right', shadow=True)

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    plt.savefig(path_to_save)
    plt.gcf().clear()

    if verbose:
        print("Plot saved at %s." % path_to_save)

def plot_events_count(events_count_dict, events_target_count_dict, filename, title="", verbose=False):
    plt.subplot(2, 1, 1)
    plt.bar(range(len(events_count_dict)), list(events_count_dict.values()), align='center')
    plt.xticks(range(len(events_count_dict)), list(events_count_dict.keys()), fontsize=4, rotation=50)
    plt.xlabel('Events')
    plt.ylabel('# in predicted game')

    plt.subplot(2, 1, 2)
    plt.bar(range(len(events_target_count_dict)), list(events_target_count_dict.values()), align='center')
    plt.xticks(range(len(events_target_count_dict)), list(events_target_count_dict.keys()), fontsize=4, rotation=50)
    plt.xlabel('Events')
    plt.ylabel('# in real game')

    plt.title(title)

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    plt.savefig(path_to_save)
    plt.gcf().clear()

    if verbose:
        print("Plot saved at %s." % path_to_save)


def plot_events_proba(proba_dict, time, last_events, filename, verbose=False):
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.subplot(gs[0])
    plt.bar(range(len(proba_dict)), list(proba_dict.values()), align='center')
    plt.xticks(range(len(proba_dict)), list(proba_dict.keys()), fontsize=5, rotation=50, horizontalalignment='right')
    plt.xlabel('Events')
    plt.ylabel('Prob. of event happening')

    plt.title(("[%d'] " % time) + " -> ".join(last_events))

    path_to_save = '%s/%s' % (EVENTS_PROBA_DIR, filename)
    plt.savefig(path_to_save)
    plt.gcf().clear()

    if verbose:
        print("Plot saved at %s." % path_to_save)


def plot_3_bars(arr1, arr2, arr3, actual_number, filename, title="", verbose=False):
    plt.title(title)

    plt.subplot(3, 1, 1)
    bar1 = plt.bar(range(len(arr1)), arr1, align='center')
    bar1[actual_number].set_color('r')
    plt.ylabel('Global')

    plt.subplot(3, 1, 2)
    bar2 = plt.bar(range(len(arr2)), arr2, align='center')
    bar2[actual_number].set_color('r')
    plt.ylabel('Team')

    plt.subplot(3, 1, 3)
    bar3 = plt.bar(range(len(arr3)), arr3, align='center')
    bar3[actual_number].set_color('r')
    plt.ylabel('Sampled')

    path_to_save = '%s/%s' % (DISTR_DIR, filename)
    plt.savefig(path_to_save)
    plt.gcf().clear()

    if verbose:
        print("Plot saved at %s." % path_to_save)

