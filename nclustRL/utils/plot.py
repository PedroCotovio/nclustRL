
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np


def get_tunning_results(df):

    pass


def get_training_results(*args):

    pass


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", **kwargs):

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    plt.legend(loc='best').set_draggable(True)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)
