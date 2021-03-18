# Find the xvalues and members of the best examples
# plot graph output
# plot mean + std method output
# Save

import numpy as np
from os import makedirs
from shutil import copy2, copyfile
import matplotlib.pyplot as plt
import sys

from os.path import isfile
from matplotlib.animation import FuncAnimation, PillowWriter

from ..PersistentGraph import PersistentGraph
from ..PersistentGraph.plots import *

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

PG_TYPE = 'KMeans'

SCORE_TYPES = [
    'max_variance'
    ]

if PG_TYPE == 'Naive':
    SCORE_TYPES = ['max_diameter']


ZERO_TYPE = 'uniform'


PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/toyexamples/"
SUB_DIRS = ["spaghettis/", "true_distrib/", "spaghettis_true_distrib/", "data/"]
PATH_BEST = PATH_FIG_PARENT+ "best/"
PATH_OUTPUTS = "PG/" +PG_TYPE+"-mean_std/"
PATH_GIF = "GIF/"
PATH_DISTRIB = ["2/gaussian/", "3/gaussian/", "N/gaussian/", "2/uniform/"]
FIG_SIZE = (5,5)
FIG_SIZE2 = (21,8)


best_2_gaussian = [
    "std_1-1_peak_0_const_0",
    "std_1-1_peak_0_const_2",
    "std_1-1_peak_0_const_4",
    "std_1-1_peak_2_const_0",
    "std_1-1_peak_2_const_2",
    "std_1-1_peak_2_const_4",
    "std_1-1_peak_4_const_4",
    "std_1-3_peak_0_const_0bis",
    "std_1-3_peak_0_const_6bis",
    "std_1-3_peak_2_const_6bis",
    "std_1-3_peak_4_const_2bis",
    "std_3-1_peak_6_const_6bis",
    "std_3-3_peak_4_const_0",
    "std_1-9_peak_6_const_2",
    ]
best_3_gaussian = [
    "std_1-1_peak_0_const_4",
    "std_1-1_peak_2_const_2",
    "std_1-1_peak_2_const_6",
    "std_1-3_peak_6_const_6bis",
    "std_1-5_peak_6_const_0",
    "std_1-5_peak_6_const_4",
]
best_N_gaussian = [
    'std_1-1_slope_1.0',
    'std_3-7_slope_1.0',
]
best_2_uniform = [
    'high_1-4_const_0_slope_0.0_ratios_[0.5, 0.5]',
    'high_1-4_const_2_slope_0.0_ratios_[0.8, 0.2]',
    'high_1-4_const_0_slope_0.33_ratios_[0.8, 0.2]',
]

best = [best_2_gaussian, best_3_gaussian, best_N_gaussian, best_2_uniform]


def plot_pg_mean_std(
    g = None,
    xvalues=None,
    members=None,
    score_type=None,
    weights=None
):

    fig = plt.figure(figsize = FIG_SIZE2, tight_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=5)

    # ------------------------------------
    # Construct graph
    # ------------------------------------
    if g is None:

        g = PersistentGraph(
            time_axis = xvalues,
            members = members,
            score_is_improving = False,
            score_type = score_type,
            zero_type = 'uniform',
            model_type = PG_TYPE,
            weights=weights,
            k_max = 8,
        )
        g.construct_graph(
            verbose=True,
        )

    # Plot Graph
    ax0 = fig.add_subplot(gs[:, 0:2])
    _, ax0 = plot_as_graph(
        g, show_vertices=True, show_edges=True,ax=ax0,
        show_std=True)
    ax0.set_title("Graph method")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Values")
    ax0 = annot_ax(g, ax=ax0)

    # k_plot
    ax1 = fig.add_subplot(gs[0, 2], sharex=ax0)
    _, ax1, _ = k_plot(g, k_max = 5, ax=ax1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Relevance")
    ax1.set_title('Number of clusters: relevance')

    # ------------------------------------
    # Current 'mean and std' method
    # ------------------------------------
    mean = np.mean(members, axis = 0)
    std = np.std(members, axis = 0)
    ax2 = fig.add_subplot(gs[:, 3:], sharey=ax0, sharex=ax0)
    ax2.plot(mean, label = 'Mean', color='r', lw=1)
    ax2.plot(mean+std, label = 'std', color='r', lw=0.5, ls='--')
    ax2.plot(mean-std, color='r', lw=0.5, ls='--')
    ax2.set_title("Mean and standard deviation method")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Values")
    ax2.legend()

    return g, fig, fig.axes

def rename_best():
    for i_type_best in range(len(best)):
        count = 0
        for name_fig in best[i_type_best]:
            for score in SCORE_TYPES:
                for subfold in ['plots/', 'graphs/']:
                    #for i_plot, plot_dir in enumerate(SUB_DIRS):
                    source_name = (
                        PATH_BEST
                        + PATH_DISTRIB[i_type_best] # 2/gaussian etc.
                        + PATH_OUTPUTS              # PG_TYPE-mean_std
                        + score + "/"               # max_inertia etc or ''
                        + subfold                   # 'plots/','graphs/' or ['']
                        + name_fig                  # std_1-1_slope_1.0 etc
                    )

                    dest_path = (
                        PATH_BEST
                        + "renamed/"
                        + PATH_OUTPUTS
                        + score + '/'
                        + subfold
                    )

                    makedirs(dest_path, exist_ok = True)

                    # Copy file
                    if subfold == 'plots/':
                        extension = ".png"
                    else:
                        extension = ""
                    file_name = source_name + extension
                    copy2(file_name, dest_path+str(count)+extension)

                    # Copy files in case of weights version available
                    file_name = source_name + "_with_weights" + extension
                    if isfile(file_name):
                        copy2(
                            file_name,
                            dest_path+str(count)+"_with_weights"+extension
                            )
            count += 1

def re_plot_saved_graph():
    for i_type_best in range(len(best)):
        for name_fig in best[i_type_best]:
            for score_type in SCORE_TYPES:

                # ------------------------------------
                # Find and copy data of best examples
                # ------------------------------------
                source_name_parent = (
                    PATH_BEST
                    + PATH_DISTRIB[i_type_best]
                    + PATH_OUTPUTS
                    + score_type+ "/"
                )
                source_name_graph = source_name_parent + "graphs/"

                # Load data
                file_name = source_name_parent + "_members.npy"
                members = np.load(file_name)
                file_name = source_name_parent + "_xvalues.npy"
                xvalues = np.load(file_name)
                file_name = source_name_parent + "_weights.npy"
                if isfile(file_name):
                    weights = np.load(file_name)
                else:
                    weights = None

                # Generate and save output plots
                g, fig, axs = plot_pg_mean_std(xvalues, members, score_type)
                dest_name_parent = None
                dest_name_fig = dest_name_parent + "plots/"
                makedirs(dest_name_fig, exist_ok = True)
                fig.savefig(dest_name_fig + name_fig +".png")

                dest_name_graph = dest_name_parent + "graphs/"
                makedirs(dest_name_graph, exist_ok = True)
                g.save(dest_name_graph + name_fig)

                # Same thing with weights
                if weights is not None:
                    g, fig, axs = plot_pg_mean_std(xvalues, members, weights=weights)
                    fig.savefig(dest_name_fig + name_fig +"_with_weights.png")
                    g.save(dest_name_graph + name_fig +"_with_weights")

def main():

    for i_type_best in range(len(best)):
        for name_fig in best[i_type_best]:
            for score_type in SCORE_TYPES:


                # ------------------------------------
                # Find and copy data of best examples
                # ------------------------------------
                source_name = (
                    PATH_BEST
                    + PATH_DISTRIB[i_type_best]
                    + SUB_DIRS[-1]
                    + name_fig
                )

                # Load data
                file_name = source_name + "_members.npy"
                members = np.load(file_name)
                file_name = source_name + "_xvalues.npy"
                xvalues = np.load(file_name)
                file_name = source_name + "_weights.npy"
                if isfile(file_name):
                    weights = np.load(file_name)
                else:
                    weights = None

                # Generate and save output plots
                g, fig, axs = plot_pg_mean_std(
                    xvalues = xvalues,
                    members = members,
                    score_type = score_type
                )
                dest_name_parent = (
                    PATH_BEST
                    + PATH_DISTRIB[i_type_best]
                    + PATH_OUTPUTS
                    + score_type+ "/"
                )
                dest_name_fig = dest_name_parent + "plots/"
                makedirs(dest_name_fig, exist_ok = True)
                fig.savefig(dest_name_fig + name_fig +".png")

                dest_name_graph = dest_name_parent + "graphs/"
                makedirs(dest_name_graph, exist_ok = True)
                g.save(dest_name_graph + name_fig)

                # Same thing with weights
                if weights is not None and PG_TYPE == 'Naive':
                    g, fig, axs = plot_pg_mean_std(
                        xvalues = xvalues,
                        members = members,
                        score_type = score_type,
                        weights = weights
                    )
                    fig.savefig(dest_name_fig + name_fig +"_with_weights.png")
                    g.save(dest_name_graph + name_fig +"_with_weights")




def make_gif_best():
    for i_type_best in range(len(best)):
        for name_fig in best[i_type_best]:

            # ------------------------------------
            # Find and copy data of best examples
            # ------------------------------------
            source_name = PATH_BEST + PATH_DISTRIB[i_type_best] + SUB_DIRS[-1] + name_fig

            # Load data
            file_name = source_name + "_members.npy"
            members = np.load(file_name)
            file_name = source_name + "_xvalues.npy"
            xvalues = np.load(file_name)
            file_name = source_name + "_weights.npy"
            if isfile(file_name):
                weights = np.load(file_name)
            else:
                weights = None

            # Construct graph
            g = PersistentGraph(members, time_axis=xvalues)
            g.construct_graph()

            # Get animation
            ani = make_gif(
                g,
                cumulative=False,
                max_iter=int(g.nb_steps/10),
                verbose=True,
            )
            writer = PillowWriter(fps=2)

            # Save
            dest_name = PATH_BEST + PATH_DISTRIB[i_type_best] + PATH_GIF
            makedirs(dest_name, exist_ok = True)
            ani.save(dest_name + name_fig +".gif", writer=writer)

            # Same thing with weights
            if weights is not None:

                # Construct graph
                g = PersistentGraph(members, time_axis=xvalues, weights=weights)
                g.construct_graph()

                # Get animation
                ani = make_gif(
                    g,
                    cumulative=False,
                    max_iter=int(max(g.nb_steps/10, 100)),
                    verbose=True,
                )
                # Save
                ani.save(dest_name + name_fig +"_with_weights.gif", writer=writer)

main()
rename_best()
#make_gif_best()