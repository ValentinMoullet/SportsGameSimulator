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

def plot_all_leagues_accuracy(filename, title="", verbose=False):

    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    axes = plt.gca()
    axes.set_ylim([0.3,0.5])

    plt.title(title)

    bookies_vals = [0.44006, 0.41873, 0.41472, 0.41095, 0.39635]
    rects1 = ax.bar(ind, bookies_vals, width, color='r')
    my_vals = [0.44700, 0.41203, 0.41502, 0.4075, 0.39810]
    rects2 = ax.bar(ind+width, my_vals, width, color='b')

    ax.set_ylabel('Accuracy')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels( ('Spain', 'England', 'Italy', 'Germany', 'France') )
    ax.legend( (rects1[0], rects2[0]), ('Bookmaker', 'Us') )

    def autolabel(rects, left=True):
        for rect in rects:
            h = rect.get_height()
            if left:
                x_pos = rect.get_x()
                y_pos = h
            else:
                x_pos = rect.get_x()+rect.get_width()
                y_pos = h

            ax.text(x_pos, y_pos, '%.4f'%h,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2, left=False)

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    plt.savefig(path_to_save)

    if verbose:
        print("Plot saved at %s." % path_to_save)

def plot_weights_teams_tsne(teams, model, filename, title="", verbose=False):
    # Latent variables for home_teams
    W_embedded = TSNE(n_components=2).fit_transform(model.input_layer[0].weight.data.numpy().T)
    x = W_embedded[:,0]
    y = W_embedded[:,1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, team_name in enumerate(teams):
        ax.annotate(team_name, (x[i],y[i]))

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    fig.savefig(path_to_save)

    if verbose:
        print("Plot saved at %s." % path_to_save)

def plot_weights_teams_pca(teams, model, filename, title="", verbose=False):
    # Latent variables for home_teams
    W_embedded = PCA(n_components=2).fit_transform(model.input_layer[0].weight.data.numpy().T)
    x = W_embedded[:,0]
    y = W_embedded[:,1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, team_name in enumerate(teams):
        ax.annotate(team_name, (x[i],y[i]))

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    fig.savefig(path_to_save)

    if verbose:
        print("Plot saved at %s." % path_to_save)

def plot_weights_teams_kernel_pca(teams, model, filename, title="", verbose=False, kernel='rbf'):
    # Latent variables for home_teams
    W_embedded = KernelPCA(n_components=2, kernel=kernel).fit_transform(model.input_layer[0].weight.data.numpy().T)
    x = W_embedded[:,0]
    y = W_embedded[:,1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, team_name in enumerate(teams):
        ax.annotate(team_name, (x[i],y[i]))

    path_to_save = '%s/%s' % (IMAGES_DIR, filename)
    fig.savefig(path_to_save)

    if verbose:
        print("Plot saved at %s." % path_to_save)

plot_all_leagues_accuracy('all_leagues_accuracy.pdf', title="Accuracy across all leagues")
