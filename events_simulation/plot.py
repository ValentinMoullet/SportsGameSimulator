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


def plot_history(histories, filename, title="", verbose=False):
    epochs = [i for i in range(len(histories[0][1]))]

    for i in range(len(histories)):
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