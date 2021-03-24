#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from ...PersistentGraph import PersistentGraph
from ...PersistentGraph.plots import *
from ...utils.nc import print_nc_dict



# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

SCORE_TYPE = [
    'MedDevMean',
    'max_MedDevMean',
    'max_inertia',
    'weighted_inertia',
    'inertia',
    'variance',
    'max_variance',
    'weighted_variance',
    ]

ZERO_TYPE = 'bounds'

# Use
# - 'overview' if you want the overview plot (entire graph + k_plot +
# most relevant components)
# - 'inside' if you want graph and k_plots on the same fig
# - 'outside' if you want 2 figs
# - anything else if you just want the entire graph
show_k_plot = 'overview'

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/MLVis2021/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = (
    "/home/natacha/Documents/tmp/figs/PG/MLVis/"
)


def preprocess_MLVis_data(verbose = True):
    # Find files
    files = [
        fname for fname in listdir(PATH_DATA)
        if fname.startswith("ec.ens.") and fname.endswith(".nc")
    ]

    # Root dictionary
    data = {}
    # subdictionary names
    dic_names = ['Lothar', 'Sandy', 'heatwave']
    f_startswith = ['ec.ens.1999', 'ec.ens.2012', 'ec.ens.2019']
    vars = [['u10', 'v10'], ['tcwv'], ['t2m']]
    #vars = [['u10'], ['tcwv'], ['t2m']]
    var_name = ['ff10', 'tcwv', 't2m']
    long_name = []

    for i, name in enumerate(dic_names):
        # New dic for each weather event
        d = {}
        # Meteogram names associated with this weather event
        d['names'] = [f for f in files if f.startswith(f_startswith[i])]
        # nc files associated with this weather event
        d['nc'] = [Dataset(PATH_DATA + f,'r') for f in d['names']]
        if verbose:
            # Show what is inside these meteograms
            print(" ----------------------- %s ----------------------- " %name)
            print_nc_dict(d['nc'][0])

        # short name for each variable of interest
        d['var_name'] = var_name[i]
        # long name (as defined by the nc file)
        d['long_name'] = d['nc'][0].variables[vars[i][0]].long_name
        # units (as defined by the nc file)
        d['units'] = d['nc'][0].variables[vars[i][0]].units

        # For each nc, create a list of np arrays containing the variable
        # of interest corresponding to the weather event
        var = [
            [ np.array(nc.variables[v]).squeeze() for v in vars[i] ]
            for nc in d['nc']
        ]

        # Compute wind speed from u10 and v10
        if name == 'Lothar':
            # keep non missing values
            u10 = var[0][0]
            print([
                (t, np.all(u10[t] == -32767))
                for t in range(len(u10)) if np.any(u10[t] == -32767)]
            )
            print(var[0][0])
            var = [ [np.sqrt(v_nc[0]**2 + v_nc[1]**2)] for v_nc in var]
            d['long_name'] = 'wind speed'

        # Now var is simply a list of np arrays(N, T)
        var = [np.swapaxes(v_nc[0], 0,1) for v_nc in var]
        d['var'] = var

        # time axis
        d['time'] = [
            nc.variables["time"] - nc.variables["time"][0]  for nc in d['nc']
        ]
        # add this weather event to our root dictionary
        data[name] = d

    return data


def plot_MLVisData():
    data = preprocess_MLVis_data()
    for name, d in data.items():
        for i in range(len(d['nc'])):
            print(d['var'][i].shape)
            plt.figure()
            ax = plt.gca()
            for m in d['var'][i]:
                ax.plot(d['time'][i], m)
            title = name + "\n" + d['names'][i]
            ax.set_title(title)
            ax.set_xlabel('Time (h)')
            ax.set_ylabel(d['long_name'] + ' ('+d['units']+')')
            plt.savefig(
                PATH_FIG_PARENT + name +'_'+ d['names'][i][:-3]+'.png'
            )

def main():

    # ---------------------------
    # Load and preprocess data
    # ---------------------------

    data = preprocess_MLVis_data()
    weights_range = [False]
    for weights in weights_range:
        for score in SCORE_TYPE:
            for pg_type in ['Naive', 'KMeans']:
                path_root = (
                    PATH_FIG_PARENT
                    + pg_type + '/'
                    + score + '/'
                )
                for name, d in data.items():
                    if name == 'Lothar':
                        continue
                    for i in range(len(d['nc'])):

                        filename = d['names'][i]

                        # --------------------------------------------
                        # ----- Prepare folders and paths ------------
                        # --------------------------------------------

                        path_fig = path_root + "plots/"
                        name_fig = path_fig + filename[:-3]
                        makedirs(path_fig, exist_ok = True)

                        path_graph = path_root + "graphs/"
                        makedirs(path_graph, exist_ok = True)
                        name_graph = path_graph + filename[:-3]

                        # ---------------------------
                        # Construct graph
                        # ---------------------------

                        g = PersistentGraph(
                                time_axis = d['time'][i],
                                members = d['var'][i],
                                score_type = score,
                                zero_type = 'bounds',
                                model_type = pg_type,
                                k_max = None,
                        )
                        g.construct_graph(
                            verbose=True,
                        )

                        # ---------------------------------
                        # Plot entire graph (with k_plot)
                        # ---------------------------------

                        ax0 = None
                        fig0 = None
                        if show_k_plot == 'inside' or show_k_plot == 'outside':
                            if show_k_plot == 'inside':
                                fig0 = plt.figure(figsize = (25,15), tight_layout=True)
                                gs = fig0.add_gridspec(nrows=2, ncols=3)
                                ax0 = fig0.add_subplot(gs[:, 0:2])
                                ax1 = fig0.add_subplot(gs[0, 2], sharex=ax0)
                            else:
                                ax1 = None
                            fig1, ax1, _ = k_plot(g, k_max = 5, ax=ax1)
                            ax1_title = 'Number of clusters: relevance'
                            ax1.set_title(ax1_title)
                            ax1.set_xlabel("Time")
                            ax1.set_ylabel("Relevance")

                            if show_k_plot == 'outside':
                                fig1.savefig(name_fig + "_k_plots")

                        ax_kw = {
                            'xlabel' : "Time (h)",
                            'ylabel' : d['long_name'] + ' ('+d['units']+')'
                            }

                        fig_suptitle = filename + "\n" + d['var_name']

                        # If overview:
                        if show_k_plot == 'overview':
                            fig0, ax0 = plot_overview(
                                g, k_max=8, show_vertices=True, show_edges=True,
                                show_std = True, ax_kw=ax_kw, ax = ax0, fig=fig0,
                            )
                            name_fig += '_overview'

                        else:
                            fig0, ax0 = plot_as_graph(
                                g, show_vertices=True, show_edges=True, show_std = True,
                                ax_kw=ax_kw, ax = ax0, fig=fig0,
                            )

                            ax0_title = 'Entire graph'
                            ax0.set_title(ax0_title)


                        # if weights:
                        #     fig_suptitle += ", with weights"
                        #     name_fig += '_weights'
                        fig0.suptitle(fig_suptitle)


                        # ---------------------------
                        # Save plot and graph
                        # ---------------------------.
                        name_fig += '.png'
                        fig0.savefig(name_fig)
                        plt.close()
                        g.save(name_graph)

if __name__ == "__main__":
    #preprocess_MLVis_data()
    main()
