#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from netCDF4 import Dataset

from ...PersistentGraph import PersistentGraph
from ...PersistentGraph.plots import *
from ...utils.nc import print_nc_dict
from ...utils.plt import from_list_to_subplots



# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

SCORE_TYPES = [
    'inertia',
    'mean_inertia',
    #'weighted_inertia',
    'max_inertia',
    #'min_inertia',       # Shouldn't be used: taking min makes no sense
    # ----------
    'variance',
    'mean_variance',
    #'weighted_variance', # Shouldn't be used: favors very high k values
    #'min_variance',      # Shouldn't be used: taking min makes no sense
    'max_variance',
    # ----------
    #'diameter',      # WARNING: diameter should be used with weights
    #'max_diameter',  # WARNING: Max diameter should be used with weights
    # ----------
    'MedDevMean',
    'mean_MedDevMean',
    'max_MedDevMean',
    # ----------
    #'max_MedDevMed', # Shouldn't be used: see details below
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


def preprocess_MLVis_data(verbose = False):
    # Find files
    files = [
        fname for fname in listdir(PATH_DATA)
    ]
    start_date = datetime.datetime(1900,1,1,0)

    # Root dictionary
    data = {}
    # subdictionary names
    dic_names = ['Lothar', 'Sandy', 'heatwave', 'coldwave']
    years = ['1999' , '2012', '2019', '2021']
    f_startswith = ['ec.ens.' + y for y in years]
    ctrl_startswith = ['ec.ensctrl.' + y for y in years]
    obs_startswith = ['e5.ans.' + y for y in years[:-1]] + ['od.ans.2021']
    vars = [['u10', 'v10'], ['msl'], ['t2m'], ['t2m']]
    var_name = ['ff10', 'msl', 't2m', 't2m']
    long_name = []

    for i, name in enumerate(dic_names):
        # New dic for each weather event
        d = {}
        # Meteogram names associated with this weather event
        d['names'] = sorted([f for f in files if f.startswith(f_startswith[i])])
        d['obs_name'] = sorted([
            f for f in files if f.startswith(obs_startswith[i])
        ])[0] #There's just one file
        d['ctrl_names'] = sorted([
            f for f in files if f.startswith(ctrl_startswith[i])
        ])

        # nc files associated with this weather event
        d['nc'] = [Dataset(PATH_DATA + f,'r') for f in d['names']]
        d['obs_nc'] = Dataset(PATH_DATA + d['obs_name'],'r')
        d['ctrl_nc'] = [Dataset(PATH_DATA + f,'r') for f in d['ctrl_names']]

        if verbose:
            # Show what is inside these meteograms
            print(" ----------------------- %s ----------------------- " %name)
            print(" ==== FORECAST ==== ")
            print_nc_dict(d['nc'][0])
            print(" ==== CONTROL ==== ")
            print_nc_dict(d['ctrl_nc'][0])
            print(" ==== OBSERVATION ==== ")
            print_nc_dict(d['obs_nc'])

        # short name for each variable of interest
        d['var_name'] = var_name[i]
        # long name (as defined by the nc file)
        d['long_name'] = d['nc'][0].variables[vars[i][0]].long_name
        # units (as defined by the nc file)
        d['units'] = d['nc'][0].variables[vars[i][0]].units
        # time axis
        d['time'] = [np.array(nc.variables["time"])  for nc in d['nc'] ]
        d['dates'] = [np.array(
            [
                datetime.timedelta(hours=int(t)) + start_date
                for t in nc.variables["time"]
            ]) for nc in d['nc']
        ]
        d['ctrl_time'] = [
            np.array(nc.variables["time"]) for nc in d['ctrl_nc']
        ]
        d['ctrl_dates'] = [np.array(
            [
                datetime.timedelta(hours=int(t)) + start_date
                for t in nc.variables["time"]
            ]) for nc in d['ctrl_nc']
        ]
        d['obs_time'] = np.array(d['obs_nc'].variables["time"])
        d['obs_dates'] = np.array(
            [
                datetime.timedelta(hours=int(t)) + start_date
                for t in d['obs_nc'].variables["time"]
            ])



        # For each nc, create a list of np arrays containing the variable
        # of interest corresponding to the weather event
        var = [
            [ np.array(nc.variables[v]).squeeze() for v in vars[i] ]
            for nc in d['nc']
        ]
        obs_var = [
            np.array(d['obs_nc'].variables[v]).squeeze() for v in vars[i]
        ]
        ctrl_var = [
            [np.array(nc.variables[v]).squeeze() for v in vars[i]]
            for nc in d['ctrl_nc']
        ]

        # Remove missing values
        if name == 'Lothar':
            # Remove missing values
            idx = np.array(
                [bool(i % 2 == 0) for i in range(len(d['time'][0])) ]
            )
            var = [ [ v[idx] for v in v_nc ] for v_nc in var ]
            ctrl_var = [[ v[idx] for v in v_nc ] for v_nc in ctrl_var ]
            d['time'] = [time_nc[idx] for time_nc in d['time'] ]
            d['dates'] = [dates_nc[idx] for dates_nc in d['dates'] ]
            d['ctrl_time'] = [time_nc[idx] for time_nc in d['ctrl_time'] ]
            d['ctrl_dates'] = [dates_nc[idx] for dates_nc in d['ctrl_dates'] ]

        if var_name[i] == 'ff10':
            var = [ [np.sqrt(v_nc[0]**2 + v_nc[1]**2)] for v_nc in var]
            obs_var = [np.sqrt(obs_var[0]**2 + obs_var[1]**2)]
            ctrl_var = [
                [np.sqrt(v_nc[0]**2 + v_nc[1]**2)]
                for v_nc in ctrl_var
            ]
            d['long_name'] = 'wind speed'

        # Now var is simply a list of np arrays(N, T)
        var = [np.swapaxes(v_nc[0], 0,1) for v_nc in var]
        d['var'] = var
        d['obs_var'] = obs_var[0]
        d['ctrl_var'] = [v_nc[0] for v_nc in ctrl_var]


        # add this weather event to our root dictionary
        data[name] = d

    return data

def find_common_dates(t, t_obs):
    # Assume that they are sorted
    start_i_obs = 0
    start_i = 0
    n = len(t)
    n_obs = len(t_obs)
    if t[0] < t_obs[0]:
        for i in range(n):
            if t[i] == t_obs[0]:
                start_i = i
                break
    if t[start_i] > t_obs[0]:
        start_i = 0
        for i in range(n_obs):
            if t[0] <= t_obs[i]:
                start_i_obs = i
                break
    i_obs = []
    i = 0
    while len(i_obs) < n:
        if n == 51:
            if t[i+start_i] == t_obs[i+start_i_obs]:
                i_obs.append(i+start_i_obs)
        else:
            # Lothar case
            if t[i//2+start_i] == t_obs[i+start_i_obs]:
                i_obs.append(i+start_i_obs)
        i += 1
    return i_obs

def plot_MLVisData(show_obs=True):
    data = preprocess_MLVis_data()
    for name, d in data.items():
        for i in range(len(d['nc'])):
            print(d['var'][i].shape)
            fig, ax = plt.subplots(figsize=(15,10))
            common_t = find_common_dates(d['time'][i], d['obs_time'])

            for m in d['var'][i]:
                ax.plot(d['time'][i]-d['time'][i][0], m)
                ax.scatter(
                    d['obs_time'][common_t] - d['time'][i][0],
                    d['obs_var'][common_t], marker="*", edgecolor='black',
                    c='r', s=200, zorder=100, lw=0.5
                )
            title = name + "\n" + d['names'][i]
            ax.set_title(title)
            ax.set_xlabel('Time (h)')
            ax.set_ylabel(d['long_name'] + ' ('+d['units']+')')
            plt.savefig(
                PATH_FIG_PARENT + name +'_'
                + d['names'][i][:-3] + "_" + d['var_name']
                +'.png'
            )

def add_obs(obs_var, obs_time, ax):
    obs_line, = ax.plot(
        obs_time,
        obs_var,
        c='orange', zorder=100, lw=1.5,
        label='obs'
    )
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [obs_line]
    # labels += ['obs']
    # ax.legend(handles, labels)
    return ax

def add_ctrl(ctrl_var, dates, ax):
    ctrl_line, = ax.plot(
        dates,
        ctrl_var,
        c='green', zorder=100, lw=1.2,
        label='ctrl'
    )
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [ctrl_line]
    # labels += ['ctrl']
    # ax.legend(handles, labels)
    return ax

def plot_spaghetti(
    show_obs=True,
    show_ctrl=True,
):
    data = preprocess_MLVis_data()
    for name, d in data.items():
        for i in range(len(d['nc'])):

            common_t = find_common_dates(d['time'][i], d['obs_time'])

            kwargs = {
                "figsize" : (15,10),
                "plot_show_mean" : True,
                "plot_show_std" : True,
                "plot_mean_zorder" : 3,
                "plot_mean_color" : 'grey',
                "plot_mean_lw" : 2,
                "plot_std_lw" : 1.5,
                "plot_std_color" : 'grey',
                "plot_std_zorder" : 3,
                "plot_std_alpha" : 0,
                "lw" : 0.6,
                "c" : "lightgrey",
                "alpha" : 1,
            }


            fig, axs = from_list_to_subplots(
                list_yvalues=d['var'][i],
                list_xvalues=d['dates'][i],
                plt_type = "plot_mean_std",
                show=False,
                dict_kwargs=kwargs
                )
            ax = axs[0,0]

            # add obs
            if show_obs:
                ax = add_obs(
                    obs_var=d['obs_var'][common_t],
                    obs_time=d['obs_dates'][common_t],
                    ax=ax
                )
            if show_ctrl:
            # add control member
                ax = add_ctrl(
                    ctrl_var=d['ctrl_var'][i],
                    dates=d['ctrl_dates'][i],
                    ax=ax
                )
            ax.legend()


            title = name + "\n" + d['names'][i]
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel(d['long_name'] + ' ('+d['units']+')')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            fig.autofmt_xdate()
            plt.savefig(
                PATH_FIG_PARENT + name +'_'
                + d['names'][i][:-3] + "_" + d['var_name']
                +'.png'
            )


def plot_obs():
    data = preprocess_MLVis_data()
    for name, d in data.items():
        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(d['obs_time'], d['obs_var'])
        title = name + "\n" + d['obs_name']
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(d['long_name'] + ' ('+d['units']+')')
        plt.savefig(
            PATH_FIG_PARENT + name +'_'+ d['obs_name'][:-3]+'.png'
        )

def select_best_examples():
    data = preprocess_MLVis_data()
    best = [
        'ec.ens.1999122512.sfc.meteogram.nc', # Lothar
        'ec.ens.2012102500.sfc.meteogram.nc', # Sandy
        'ec.ens.2019072100.sfc.meteogram.nc', # Heatwave
        'ec.ens.2021020800.sfc.meteogram.nc', # Coldwave
        ]
    max_dates = [
        datetime.datetime(1999, 12, 29, 12), # Lothar
        datetime.datetime(2012, 11,  3, 12), # Sandy
        datetime.datetime(2019,  7, 30, 12), # Heatwave
        datetime.datetime(2021,  2, 17,  0), # Coldwave
    ]
    names = ['Lothar', 'Sandy', 'heatwave', 'coldwave']
    idx = [
        data[names[i]]['nc'].index(best[i]) for i in range(len(names))
    ]

    # ================================
    # Lothar
    # ================================

    # ================================
    # Sandy
    # ================================



def use_dates_as_xticks(
    ax,
    time_axis,
    start_date = datetime.datetime(1900,1,1,0),
):
    # If you want to have dates as xticks  indpt of what you used to plot your curve
    # uniformly spaced hours between the min and max
    total_hours = np.arange(
        start=time_axis[0], stop=time_axis[-1], step=time_axis[1]-time_axis[0]
    )
    # Now choose how often you want it displayed
    idx = [i for i in range(len(total_hours)) if i%4==0]
    kept_hours = total_hours[idx]

    # Create the string of dates at the kept locations
    # Creating string here directly is the safest way of
    # controling the format in the plot......
    # Dealing with ax.xaxis.set_major_formatter
    # And ax.xaxis.set_minor_formatter is way too painful...
    labels = [
        (datetime.timedelta(
            hours=int(h)
        ) + start_date).strftime('%m-%d') for h in kept_hours]
    # Specify where you want the xticks
    ax.set_xticks(kept_hours)
    # Specify what you want as xtick labels
    ax.set_xticklabels(labels)
    # The only solution working if you want to change just one ax
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    return ax

def main(show_obs=True):

    # ---------------------------
    # Load and preprocess data
    # ---------------------------
    plt.rcParams.update({'font.size': 30})

    data = preprocess_MLVis_data()
    weights_range = [False]
    for weights in weights_range:
        for score in SCORE_TYPES:
            for pg_type in ['Naive', 'KMeans']:
                path_root = (
                    PATH_FIG_PARENT
                    + pg_type + '/'
                    + score + '/'
                )
                for name, d in data.items():
                    for i in range(len(d['nc'])):

                        filename = d['names'][i]

                        # --------------------------------------------
                        # ----- Prepare folders and paths ------------
                        # --------------------------------------------

                        path_fig = path_root + "plots/"
                        name_fig = (
                            path_fig + name +'_'
                            + filename[:-3] + "_" + d['var_name']
                        )
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
                                k_max = 10,
                        )
                        g.construct_graph(
                            verbose = False,
                            quiet = False,
                        )

                        # ---------------------------------
                        # Plot entire graph (with k_plot)
                        # ---------------------------------

                        ax0 = None
                        fig0 = None

                        ax_kw = { }
                        fig_suptitle = filename + "\n" + d['var_name']


                        fig0, ax0 = plot_overview(
                            g, k_max=8, show_vertices=True, show_edges=True,
                            show_std = True, ax_kw=ax_kw, ax = ax0, fig=fig0,
                        )
                        name_fig += '_overview'

                        ax0 = use_dates_as_xticks(ax0[0],  d['time'][i])
                        ax0.set_xlabel('Date')
                        ax0.set_ylabel(d['long_name'] + ' ('+d['units']+')')

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
