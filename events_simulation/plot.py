#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Plot functions to create figures.
"""

import numpy as np
import matplotlib.pyplot as plt
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
