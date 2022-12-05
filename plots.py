import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


warnings.filterwarnings("ignore")
sns.set_style('whitegrid')


def _check_arrays(y, y_pred):
    y = check_array(y, ensure_2d=False, force_all_finite=True)
    y_pred = check_array(y_pred, ensure_2d=False, force_all_finite=True)

    check_consistent_length(y, y_pred)

    return y, y_pred
    
    
def format_plot(ax=None):
    ax.tick_params(axis='x', labelrotation=0, labelsize=14, colors="#2639E9", grid_color="#FFF")
    ax.tick_params(axis='y', labelrotation=0, labelsize=14, colors="#2639E9", grid_color="#FFF")
    ax.spines['top'].set_color("#2639E9")
    ax.spines['bottom'].set_color("#2639E9")
    ax.spines['right'].set_color("#2639E9")
    ax.spines['left'].set_color("#2639E9")


def plot_auc_roc(y, y_pred, title=None, xlabel=None, ylabel=None, save=None, **kwargs):
    y, y_pred = _check_arrays(y, y_pred)

    # Define the arrays for plotting
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred)

    # Define the plot settings
    if title is None:
        title = "ROC curve"
    if xlabel is None:
        xlabel = "False Positive Rate"
    if ylabel is None:
        ylabel = "True Positive Rate"

    plt.plot(fpr, fpr, linestyle=":", color="#F76E6C", label="Random Model")
    plt.plot(fpr, tpr, color="#2639E9", label="Model (AUC: {:.5f})".format(auc_roc))
    plt.stackplot(fpr, tpr, color="#2639E9", alpha=0.2)
    plt.title(title, fontdict={"fontsize": 14})
    plt.xlabel(xlabel, fontdict={"fontsize": 14})
    plt.ylabel(ylabel, fontdict={"fontsize": 14})
    plt.legend(frameon=False, loc='upper left')
    
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    # Save figure if requested. Pass kwargs.
    if save:
        plt.savefig(fname=save, **kwargs)
        plt.close()


def plot_ks(y, y_pred, title=None, xlabel=None, ylabel=None, save=None, **kwargs):
    y, y_pred = _check_arrays(y, y_pred)

    n_samples = y.shape[0]
    n_event = np.sum(y)
    n_nonevent = n_samples - n_event

    idx = y_pred.argsort()
    yy = y[idx]
    pp = y_pred[idx]

    cum_event = np.cumsum(yy)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event
    ks_score = np.max(p_diff)
    ks_max_idx = np.argmax(p_diff)

    # Define the plot settings
    if title is None:
        title = "Kolmogorov-Smirnov"
    if xlabel is None:
        xlabel = "Threshold"
    if ylabel is None:
        ylabel = "Cumulative probability"

    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlabel, fontdict={'fontsize': 14})
    plt.ylabel(ylabel, fontdict={'fontsize': 14})

    plt.plot(pp, p_event, color="#FE7715", label="Cumulative events")
    plt.plot(pp, p_nonevent, color="#F76E6C", label="Cumulative non-events")
    plt.plot(pp, np.abs(p_nonevent - p_event), color="#2639E9", label="Kolmogorov-Smirnov")
    plt.fill_between(pp, p_event, p_nonevent, color="#2639E9", alpha=0.2)

    plt.vlines(pp[ks_max_idx], ymin=p_event[ks_max_idx], ymax=p_nonevent[ks_max_idx], color="#F76E6C", linestyles=":")

    # Set KS value inside plot
    pos_x = pp[ks_max_idx] + 0.02
    pos_y = 0.5 * (p_nonevent[ks_max_idx] + p_event[ks_max_idx])
    text = "KS: {:.2%} at {:.2f}".format(ks_score, pp[ks_max_idx])
    plt.text(pos_x, pos_y, text, fontsize=14, rotation_mode="anchor")

    plt.legend(frameon=False, loc='upper left')
    
    expend = (max(y_pred) - min(y_pred)) / 25
    print(max(y_pred), min(y_pred), expend)
    plt.xlim(min(y_pred) - expend, max(y_pred) + expend)
    plt.ylim(-0.05, 1.05)

    # Save figure if requested. Pass kwargs.
    if save:
        plt.savefig(fname=save, **kwargs)
        plt.close()


def eval_plot(score, target, title="", figsize=(18, 7), save=None):
        fig = plt.figure(figsize=figsize)
        
        ax1 = plt.subplot(1, 2, 1)
        plot_ks(target, score)
        format_plot(ax1)
        
        ax2 = plt.subplot(1, 2, 2)
        plot_auc_roc(target, score)
        format_plot(ax2)

        title = "" if not title else f"{title} "
        plt.suptitle(f"{title}K-S & ROC CURVE", fontsize=14, fontweight="bold")
        
        if save:
            if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))
            
            plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

        return fig
    

def score_hist(score, y_true, figsize=(18, 7), bins=30, save=None):
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    palette = sns.diverging_palette(340, 267, n=2, s=100, l=40)

    sns.histplot(
                x=score, hue=y_true.replace({0: "good", 1: "bad"}), element="step", stat="density", bins=bins, common_bins=True, common_norm=True, palette=palette
            )
    
    format_plot(ax)

    ax.set_xlabel("score", fontdict={'fontsize': 16})
    ax.set_ylabel("density", fontdict={'fontsize': 16})
    
    plt.legend(["bad", "good"], frameon=False, loc='upper left')

    if save:
        if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))
            
        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")
    
    return fig