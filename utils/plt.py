from math import ceil, floor, sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

from .npy import unpack_2d_dimensions

# FIXME: remove xlabels and ylabels in the middle when xaxis and xaxis are
# not shared
# TODO: remove the prefix "list_" from "list_ax_titles/labels.etc"
# TODO: hist type plot
# TODO: change this:
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
# TODO: In set_list_list_legends: add the list to the existing legend


def show_kwargs(verbose=True):
    keys = {}
    fig_keys = [
        "nrows", "ncols", "figsize", "dpi", "sharex", "sharey", "squeeze",
    ]
    font_keys = [
        "small_size", "medium_size", "bigger_size", "font", "axes_title",
        "axes_label", "xtick", "ytick", "legend", "figure"
    ]
    text_keys = [
        "fig_suptitle", "list_ax_titles", "list_xlabels", "list_ylabels",
        "axes_title", "list_list_legends"
    ]
    line_keys = [
        "ls", "linestyle", "linewidth", "lw", "color",
        "c", "marker", "alpha"
    ]
    plot_keys = [
        # Booleans
        "plot_show_mean", "plot_show_std", "plot_show_std_label",
        "plot_show_mean_label",
        # Style
        "plot_color", "plot_alpha", "plot_lw", "plot_ls",
        "plot_mean_color", "plot_std_color", "plot_std_alpha",
        "plot_mean_zorder", "plot_std_zorder",
        "plot_mean_lw", "plot_mean_linewidth",
        "plot_mean_ls", "plot_mean_linestyle",
        "plot_std_lw", "plot_std_linewidth",
        "plot_std_ls", "plot_std_linestyle",
        # Labels texts
        "plot_mean_label", "plot_std_label",

    ]
    kde_keys = [
        "kde_label", "kde_color"
    ]
    fit_keys = [
        "fit_type", "fit_label", "fit_color", "fit_show_mean", "fit_show_std",
        "fit_ls",
    ]
    hist_keys = [
        "hist_label", "hist_color"
    ]
    dict_keys = {
        "font" : font_keys,
        "fig" : fig_keys,
        "text" : text_keys,
        "kde" : kde_keys,
        "fit" : fit_keys,
        "hist" : hist_keys,
        "plot" : plot_keys,
        "line" : line_keys,
    }
    if verbose:
        print("================================== ")
        print("============= GENERAL ============ ")
        print("================================== ")
        print("\n===== plt.figure() ===== ")
        print(fig_keys)
        print("\n=== Text properties ==== ")
        print(font_keys)
        print("\n==== Text content ====== ")
        print(text_keys)
        print("================================== ")
        print("=========== plt.plot() =========== ")
        print("================================== ")
        print("\n======== Line2D  ======= ")
        print(line_keys)
        print("\n====== customize ======= ")
        print(plot_keys)
        print("================================== ")
        print("======== sns.distplot() ========== ")
        print("================================== ")
        print("\n========= kde ========== ")
        print(kde_keys)
        print("\n========= fit ========== ")
        print(fit_keys)
        print("\n=========  hist ======== ")
        print(hist_keys)
    return(dict_keys)


def sort_kwargs(
    kwargs
):
    dict_keys = show_kwargs(verbose=False)
    dict_kwargs = {
        "font" : {},
        "fig" : {},
        "text" : {},
        "dflt" : {},
        "kde" : {},
        "fit" : {},
        "hist" : {},
        "plot" : {},
        "line" : {},
    }

    for key in kwargs:
        if key in dict_keys['fig']:
            dict_kwargs["fig"][key] = kwargs[key]
        elif key in dict_keys['font']:
            dict_kwargs['font'][key] = kwargs[key]
        elif key in dict_keys['text']:
            dict_kwargs['text'][key] = kwargs[key]
        elif key in dict_keys['kde']:
            dict_kwargs['kde'][key[4:]] = kwargs[key]
        elif key in dict_keys['fit']:
            dict_kwargs['fit'][key[4:]] = kwargs[key]
        elif key in dict_keys['hist']:
            dict_kwargs['hist'][key[5:]] = kwargs[key]
        elif key in dict_keys['plot']:
            dict_kwargs['plot'][key[5:]] = kwargs[key]
        elif key in dict_keys['line']:
            dict_kwargs['line'][key] = kwargs[key]
        else:
            dict_kwargs['dflt'][key] = kwargs[key]
    return(dict_kwargs)

def set_kwargs_default_values(dic, plt_type):
    if plt_type == "distplot":
        # Default axis labels
        dic['text']["list_xlabels"] = dic['text'].pop("list_xlabels" ,"Values")
        dic['text']["list_ylabels"] = dic['text'].pop("list_ylabels" ,"Density")

        # keywords default values for the gaussian
        dic['fit']["color"] = dic['fit'].pop("color", "red")
        dic['fit']["type"] = dic['fit'].pop("type", "norm")
        dic['fit']["show_mean"] = dic['fit'].pop("show_mean", True)
        dic['fit']["show_std"] = dic['fit'].pop("show_std", True)
        dic['fit']["ls"] = dic['fit'].pop("ls", "--")

        # keywords default values for the kde
        dic['kde']["label"] = dic['kde'].pop("label", "kde fit")
        dic['kde']["color"] = dic['kde'].pop("color", "blue")

        # keywords default values for the histogram
        dic['hist']["label"] = dic['hist'].pop("label", "histogram")
        dic['hist']["color"] = dic['hist'].pop("color", "blue")

    elif plt_type in ["plot",  "plot_mean_std"]:
        dic['plot']['mean_label'] = dic['plot'].pop("mean_label", "Mean")
        dic['plot']['std_alpha'] = dic['plot'].pop("std_alpha", 0.1)
        dic['plot']['mean_zorder'] = dic['plot'].pop("mean_zorder", None)
        dic['plot']['std_zorder'] = dic['plot'].pop("std_zorder", None)
        dic['plot']['std_zorder'] = dic['plot'].pop("std_zorder", None)

        dic['plot']["mean_lw"] = dic['plot'].pop("mean_lw", 1)
        dic['plot']['std_lw'] = dic['plot'].pop("std_lw", 0.5)
        dic['plot']["mean_ls"] = dic['plot'].pop("mean_ls", "-")
        dic['plot']['std_ls'] = dic['plot'].pop("std_ls", "--")

        if plt_type == "plot":
            # For custom multi-line plots,potentially with mean and std
            dic['plot']['show_mean'] = dic['plot'].pop("show_mean", False)
            dic['plot']['show_std'] = dic['plot'].pop("show_std", False)
            dic['plot']['show_mean_label'] = dic['plot'].pop("show_mean_label", True)
            dic['plot']['show_std_label'] = dic['plot'].pop("show_std_label", False)

        if plt_type == "plot_mean_std":
            # For plot with mean and std
            dic['plot']['show_mean'] = dic['plot'].pop("show_mean", True)
            dic['plot']['show_std'] = dic['plot'].pop("show_std", True)
            dic['plot']['show_mean_label'] = dic['plot'].pop("show_mean_label", True)
            dic['plot']['show_std_label'] = dic['plot'].pop("show_std_label", True)
            dic['plot']['mean_label'] = dic['plot'].pop("mean_label", 'Mean')
            dic['plot']['std_label'] = dic['plot'].pop("std_label", 'std')
            # If color is None, matplotlib will pick a value
            dic['plot']['mean_color'] = dic['plot'].pop("mean_color", None)
            dic['plot']['std_color'] = dic['plot'].pop("std_color", None)
    return dic

def get_nrows_ncols_from_nplots(
    nplots: int,
    nrows: int = None,
    ncols: int = None,
):
    if nplots < 2:
        ncols = 1
        nrows = 1
    if (nrows is None) and (ncols is None):
        ncols = max(ceil(sqrt(nplots)), 1)
        nrows = max(ceil(nplots/ncols), 1)
    elif nrows is None:
        nrows = max(ceil(nplots/ncols), 1)
    else:
        ncols = max(ceil(nplots/nrows), 1)
    return(nrows,ncols)

def get_subplot_indices(
    i_subplot: int,
    ncols: int,
    squeeze:bool = False,
):
# Return the ax index of the subplot number i_subplot
    if ncols == 1 and squeeze:
        # len(axs.shape)=1 if ncols=1
        idx = i_subplot
    else:
        idx = (int(i_subplot/ncols), i_subplot % ncols)
    return idx


def pretty_subplots(
    nplots: int = 1,
    dict_kwargs={},
    sorted_kwargs = False,
):

    fig_kw = dict_kwargs.pop("fig", {})
    # Set default values:
    nrows = fig_kw.pop("nrows", None)
    ncols = fig_kw.pop("ncols", None)
    sharex = fig_kw.pop("sharex", True)
    sharey = fig_kw.pop("sharey", True)
    squeeze = fig_kw.pop("squeeze", False)
    dpi = fig_kw.pop("dpi", 100)
    figsize = fig_kw.pop("figsize", (32,15))

    nrows, ncols = get_nrows_ncols_from_nplots(
        nplots = nplots,
        nrows = nrows,
        ncols = ncols,
    )
    set_fonts(**dict_kwargs['font'])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        dpi=dpi,
        figsize=figsize,
        **fig_kw,
        )
    #fig.tight_layout()
    fig, axs = set_text(
        axs,
        fig=fig,
        **dict_kwargs['text']
    )
    return fig, axs

def set_fonts(
    axs=None,
    small_size: int = None,
    medium_size: int = None,
    bigger_size: int = None,
    font: int = None,
    axes_title: int = None,
    axes_label: int = None,
    xtick: int = None,
    ytick: int = None,
    legend: int = None,
    figure: int = None,
):
    if small_size is None:
        small_size = 15
    if medium_size is None:
        medium_size = 20
    if bigger_size is None:
        bigger_size = 25
    # Then set global setting
    if axs is None:
        # controls default text sizes
        if font is None:
            matplotlib.rc('font', size=small_size)
        else:
            matplotlib.rc('font', size=font)
        # fontsize of the axes title
        if axes_title is None:
            matplotlib.rc('axes', titlesize=medium_size)
        else:
            matplotlib.rc('axes', titlesize=axes_title)
        # fontsize of the x and y labels
        if axes_label is None:
            matplotlib.rc('axes', labelsize=medium_size)
        else:
            matplotlib.rc('axes', labelsize=axes_label)
        # fontsize of the tick labels
        if xtick is None:
            matplotlib.rc('xtick', labelsize=small_size)
        else:
            matplotlib.rc('xtick', labelsize=xtick)
        # fontsize of the tick labels
        if ytick is None:
            matplotlib.rc('ytick', labelsize=small_size)
        else:
            matplotlib.rc('ytick', labelsize=ytick)
        # legend fontsize
        if legend is None:
            matplotlib.rc('legend', fontsize=medium_size)
        else:
            matplotlib.rc('legend', fontsize=legend)
        # fontsize of the figure title
        if figure is None:
            matplotlib.rc('figure', titlesize=bigger_size)
        else:
            matplotlib.rc('figure', titlesize=figure)
    # Set only fonts on the given ax
    else:
        is_axs_squeezed = False
        if not isinstance(axs,  np.ndarray):
            is_axs_squeezed = True
            axs = np.array(axs, ndmin=2)
        for ax in axs.flat:
            items = [ax.title, ax.xaxis.label, ax.yaxis.label]
            fonts = [axes_title, axes_label, axes_label, xtick, ytick]
            dflt_fonts = [small_size, small_size, medium_size, small_size,
                        small_size]

            lgd = ax.get_legend()
            if lgd:
                list_lgd_txt =  lgd.get_texts()
                for i, lgd_txt in enumerate(list_lgd_txt):
                    items += lgd_txt
                    fonts += legend
                    dflt_fonts += small_size

            for i, item in enumerate(items):
                if fonts[i] is None:
                    font_size = fonts[i]
                else:
                    font_size = dflt_fonts[i]
                item.set_fontsize(font_size)
        if is_axs_squeezed:
            axs = axs[0,0]
    return axs


def set_list_fig_suptitle(
    list_figs,
    list_fig_suptitles,
    dict_kwargs = {},
):
    """For each fig, set the corresponding title

    If list_fig_suptitles is None, does nothing
    If list_fig_suptitles has only one element (or if it is a string)
    but not list_figs, Then set the same title for each fig in figs

    :param list_figs: [description]
    :type list_figs: List[fig],
    :param list_fig_suptitles: [description]
    :type list_fig_suptitles:

        List[str] or str
        len=len(list_fig_suptitles) or 1

    :return: [description]
    :rtype: [type]
    """
    if not isinstance(list_figs, list):
        list_figs = [list_figs]
    if list_fig_suptitles is not None:
        nfigs = len(list_figs)
        if isinstance(list_fig_suptitles, str) :
            list_fig_suptitles = [list_fig_suptitles]*nfigs
        elif len(list_fig_suptitles)==1:
            list_fig_suptitles = list_fig_suptitles*nfigs
        for i, fig in enumerate(list_figs):
            fig.suptitle(list_fig_suptitles[i], **dict_kwargs)
    return list_figs

def set_list_ax_titles(
    axs,
    list_ax_titles,
    dict_kwargs = {},
):
    """For each subplot of axs, set the corresponding title

    If list_ax_titles is None, does nothing
    If list_ax_titles has only one element (or if it is a string)
    but not axs, Then set the same title for each ax in axs

    :param axs: [description]
    :type axs: ndarray(nrows, ncols)
    :param list_ax_titles:
    :type list_ax_titles:

        List[str] or str
        len = nrow*ncols or 1

    :return: axs
    :rtype:
    """
    if list_ax_titles is not None:
        is_axs_squeezed = False
        if not isinstance(axs,  np.ndarray):
            is_axs_squeezed = True
            axs = np.array(axs, ndmin=2)
        nplots = len(axs.flat)
        if isinstance(list_ax_titles, str):
            list_ax_titles = [list_ax_titles]*nplots
        elif len(list_ax_titles)==1:
            list_ax_titles = list_ax_titles*nplots
        for i, ax in enumerate(axs.flat):
            ax.set_title(list_ax_titles[i], dict_kwargs)
        if is_axs_squeezed:
            axs = axs[0,0]
    return axs

def set_list_xlabels(
    axs,
    list_labels,
    dict_kwargs = {},
):
    """For each subplot of axs, set the corresponding xlabel

    If list_labels is None, does nothing
    If list_labels has only one element (or if it is a string)
    axs, Then set the same label for each ax in axs

    :param axs: [description]
    :type axs: ndarray(nrows, ncols)
    :param list_labels: [description]
    :type list_labels:

        List[str] or str,
        len = nrow*ncols or 1
    :return: [description]
    :rtype: [type]
    """
    if list_labels is not None:
        is_axs_squeezed = False
        if not isinstance(axs,  np.ndarray):
            is_axs_squeezed = True
            axs = np.array(axs, ndmin=2)
        nplots = len(axs.flat)
        if isinstance(list_labels, str):
            list_labels = [list_labels]*nplots
        elif len(list_labels)==1:
            list_labels = list_labels*nplots
        for i, ax in enumerate(axs.flat):
            ax.set_xlabel(list_labels[i], **dict_kwargs)
        if is_axs_squeezed:
            axs = axs[0,0]
    return axs

def set_list_ylabels(
    axs,
    list_labels,
    dict_kwargs = {},
):
    """For each subplot of axs, set the corresponding ylabel

    If list_labels is None, does nothing
    If list_labels has only one element (or if it is a string)
    axs, Then set the same label for each ax in axs

    :param axs: [description]
    :type axs: ndarray(nrows, ncols)
    :param list_labels: [description]
    :type list_labels:

        List[str] or str
        len = 1 or len = nrow*ncols

    :return: [description]
    :rtype: [type]
    """
    if list_labels is not None:
        is_axs_squeezed = False
        if not isinstance(axs,  np.ndarray):
            is_axs_squeezed = True
            axs = np.array(axs, ndmin=2)
        nplots = len(axs.flat)
        if isinstance(list_labels, str):
            list_labels = [list_labels]*nplots
        elif len(list_labels)==1:
            list_labels = list_labels*nplots
        for i, ax in enumerate(axs.flat):
            ax.set_ylabel(list_labels[i], dict_kwargs)
        if is_axs_squeezed:
            axs = axs[0,0]
    return axs

def set_list_list_legends(
    axs,
    list_list_legends,  # List[List[str]]
    list_list_lines=None,
    dict_kwargs = {},
):
    """For each lin of each subplot of axs, set the corresponding list_legend

    If list_list_legends is None, does nothing
    If list_list_legends has only one element (or if it is a string)
    axs, Then set the same label for each ax in axs

    :param axs: [description]
    :type axs: ndarray(nrows, ncols)
    :param list_list_legends: [description]
    :type list_list_legends:

        List[List[str]] or List[str] str
        len = 1 or len = nrow*ncols

    :return: [description]
    :rtype: [type]
    """
    is_axs_squeezed = False
    if not isinstance(axs,  np.ndarray):
        is_axs_squeezed = True
        axs = np.array(axs, ndmin=2)
    if list_list_legends is not None:

        nplots = len(axs.flat)
        # one legend per subplot and same one for each subplot
        if isinstance(list_list_legends, str):
            list_list_legends = [[list_list_legends]]*nplots
        else:
            # make sure list_legend is a list
            if isinstance(list_list_legends[0], str):
                list_list_legends = [list_list_legends]*nplots
        # same legend for each subplot (NO)
        if len(list_list_legends)==1:
            list_list_legends = list_list_legends*nplots
        for i, ax in enumerate(axs.flat):
            # Check if some legends have already been added during the plot phase
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                list_legends = labels + list_list_legends[i]
            else:
                list_legends = list_list_legends[i]
            ax.legend(labels=list_legends, **dict_kwargs)
            # ax = add_labels_to_legend(
            #     ax = ax,
            #     text_labels=list_list_legends[i],
            #     lines = None,
            #     **kwargs)
    else:
        for i, ax in enumerate(axs.flat):
            # Check if some legends have already been added during the plot phase
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(labels=labels, loc='best')
    if is_axs_squeezed:
        axs = axs[0,0]
    return axs

def add_labels_to_legend(
    ax,
    text_labels,
    style_labels = [{}],   # List of style kw (color, lw, ls, etc)
    lines = [],
    link_labels_to_existing_lines: bool = True,
    clear_prev_legend: bool = False,
    dict_kwargs = {},
):
    prev_handles, prev_labels = ax.get_legend_handles_labels()
    if prev_labels and clear_prev_legend:
        ax.get_legend().remove()
    if isinstance(text_labels, str):
        text_labels = [text_labels]
    if isinstance(text_labels, list):
        nlines = len(text_labels)
        if not lines:
            lines = []
            if isinstance(style_labels, dict):
                style_labels = [style_labels]*nlines
            if len(style_labels) == 1:
                style_labels = style_labels*nlines
            for i in range(nlines):
                lines.append(Line2D([0], [0], **style_labels[i]))
        if prev_handles:
            lines = prev_handles + lines
            text_labels = prev_labels + text_labels
        ax.legend(lines, text_labels, **dict_kwargs)
    if prev_handles and prev_labels:
        ax.legend(prev_handles, prev_labels, **dict_kwargs)
    return ax

def move_legend(
    ax1,
    ax2,
    legend_kwargs = {},
):
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, **legend_kwargs)

def plot_mean_and_std(
    yvalues = None,
    mean_values = None,
    std_values = None,
    xvalues = None,
    fig = None,
    ax = None,
    dict_kwargs={},
    sorted_kwargs = False,
):
    """
    Plot mean and std

    You can either give directly the mean line in 'mean_line' or
    give raw data in 'yvalues' (n_lines, n_values)-array

    :param yvalues: [description], defaults to None
    :type yvalues: [type], optional
    :param mean_line: [description], defaults to None
    :type mean_line: [type], optional
    :param xvalues: [description], defaults to None
    :type xvalues: [type], optional
    :param fig: [description], defaults to None
    :type fig: [type], optional
    :param ax: [description], defaults to None
    :type ax: [type], optional
    :param dict_kwargs: [description], defaults to {}
    :type dict_kwargs: dict, optional
    :param sorted_kwargs: [description], defaults to False
    :type sorted_kwargs: bool, optional
    :return: [description]
    :rtype: [type]
    """

    if ax is None:
        fig, ax = pretty_subplots(
            dict_kwargs = dict_kwargs,
            sorted_kwargs = sorted_kwargs
            )
    if not sorted_kwargs:
        dict_kwargs = sort_kwargs(dict_kwargs)
    if xvalues is None:
        xvalues = np.arange(len(yvalues))

    dict_kwargs = set_kwargs_default_values(dict_kwargs, plt_type = "plot_mean_std")
    if yvalues is not None:
        mean = np.mean(yvalues, axis=0)
        std = np.std(yvalues, axis=0)
    else :
        mean = mean_values
        std = std_values

    # ------------------ Draw the mean line-----------------------------
    mean_line, = ax.plot(
        xvalues, mean,
        color = dict_kwargs['plot']['mean_color'],
        lw = dict_kwargs['plot']['mean_lw'],
        ls = dict_kwargs['plot']['mean_ls'],
        label = dict_kwargs['plot']['mean_label'],
        )

    # --------------------- Draw std lines------------------------------
    std_color = dict_kwargs['plot']['std_color']
    # If None then give the same color as the mean line
    if std_color is None:
        std_color = line[-1].get_color()
    # --------- If alpha = 0 just show border lines ------------
    if dict_kwargs['plot']['std_alpha'] == 0:
        std_line, = ax.plot(
            xvalues, mean + std,
            color = std_color,
            lw = dict_kwargs['plot']['std_lw'],
            ls = dict_kwargs['plot']['std_ls'],
            )
        ax.plot(
            xvalues, mean - std,
            color = std_color,
            lw = dict_kwargs['plot']['std_lw'],
            ls = dict_kwargs['plot']['std_ls'],
            label = dict_kwargs['plot']['std_label']
        )
    # -- Else don't show border lines, just fill between them --
    else:
        ax.fill_between(
            xvalues, mean + std, mean - std,
            alpha = dict_kwargs['plot']['std_alpha'],
            color = std_color,
            zorder = dict_kwargs['plot']['std_zorder'],)

    return fig, ax


# def custom_plot(
#     xvalues,
#     yvalues,
#     dict_kwargs = {},
#     ax=None,
#     fig=None,
# ):
#     """
#     Add mean and std to multi-line plots

#     :param xvalues: [description]
#     :type xvalues: [type]
#     :param yvalues: [description]
#     :type yvalues: [type]
#     :param dict_kwargs: [description]
#     :type dict_kwargs: [type]
#     :param ax: [description], defaults to None
#     :type ax: [type], optional
#     :param fig: [description], defaults to None
#     :type fig: [type], optional
#     :return: [description]
#     :rtype: [type]
#     """
#     if ax is None:
#         fig, ax = plt.subplots(**dict_kwargs['fig'])
#     text_labels = []
#     lines_labels = []
#     if dict_kwargs['plot']['show_mean']:
#         mean_line = np.mean(yvalues, axis=0)
#         line = ax.plot(
#             xvalues,
#             mean_line,
#             ls="--",
#             c=dict_kwargs['plot']['mean_color'],
#             lw=5,
#             zorder=dict_kwargs['plot']['mean_zorder'],
#         )
#         if dict_kwargs['plot']['show_mean_label']:
#             text_labels.append(dict_kwargs['plot']['show_mean_label'])
#             lines_labels.append(line)
#     if dict_kwargs['plot']['show_std']:
#         mean_line = np.mean(yvalues, axis=0)
#         std_values = np.std(yvalues, axis=0)
#         std_line_sup = mean_line + std_values
#         std_line_inf = mean_line - std_values
#         # If alpha = 0 just show border lines
#         if dict_kwargs['plot']['std_alpha'] == 0:
#             ax.plot(
#                 xvalues,std_line_inf,
#                 ls="--", lw=2,
#                 color=dict_kwargs['plot']['mean_color'],
#                 zorder=dict_kwargs['plot']['std_zorder'],
#             )
#             line = ax.plot(
#                 xvalues,std_line_sup,
#                 ls="--", lw=2,
#                 color=dict_kwargs['plot']['mean_color'],
#                 zorder=dict_kwargs['plot']['std_zorder'],
#             )
#             if dict_kwargs['plot']['show_mean_label']:
#                 text_labels.append("std")
#                 lines_labels.append(line)

#         # Else don't show border lines, just fill between them
#         else:
#             ax.fill_between(
#                 xvalues,
#                 std_line_inf, std_line_sup,
#                 alpha=dict_kwargs['plot']['std_alpha'],
#                 color=dict_kwargs['plot']['mean_color'],
#                 zorder=dict_kwargs['plot']['std_zorder'],)
#     if dict_kwargs['plot']['show_mean_label']:
#         add_labels_to_legend(
#             ax,
#             text_labels=text_labels,
#             lines=lines_labels,
#         )

#     return fig, ax



def distrib_vs_gaussian(
    ax,
    distrib,
    bins: int = 50,
    dict_kwargs = {},
):
    fit_kw = dict_kwargs['fit']
    if fit_kw["type"] == "norm":
        (mu, sigma) = norm.fit(distrib)
        fit = norm
        label_fit = "Gaussian fit"
    else:
        fit = None
        label_fit = ""
        show_mean = False,
        show_std = False
    if fit_kw["type"] == "norm" and fit_kw.pop("show_mean"):
        mu = f"{mu:.3}"
        label_fit += " \n$\mu$=" + mu
    if fit_kw["type"] == "norm" and fit_kw.pop("show_std"):
        sigma = f"{sigma:.3}"
        label_fit += "  \n$\sigma$=" + sigma
    fit_kw.pop('type')  # we should remove this kw now
    fit_kw["label"] = fit_kw.pop("label", label_fit)

    distrib = distrib.flatten()
    ax = sns.distplot(
        distrib,
        ax=ax,
        fit=fit,
        color="r",
        bins=bins,
        fit_kws=fit_kw,
        hist_kws=dict_kwargs['hist'],
        kde_kws=dict_kwargs['kde'],
        #**dict_kwargs['dflt']
    )
    return ax

def density_pair_plots(
    fig,
    ax,
    xvalues,
    yvalues,
):
    ax.hist2d(
        xvalues, yvalues,
        bins=(100, 100), density=True, cmap=plt.cm.Reds
    )
    return ax

def set_text(
    axs,
    fig = None,
    fig_suptitle: str = None,
    list_ax_titles = None,
    list_xlabels = None,
    list_ylabels = None,
    list_list_legends = None,
    dict_kwargs = {},
):
    if axs is not None:
        axs = set_list_ax_titles(axs, list_ax_titles=list_ax_titles, dict_kwargs=dict_kwargs)
        axs = set_list_xlabels(axs, list_labels=list_xlabels, dict_kwargs=dict_kwargs)
        axs = set_list_ylabels(axs, list_labels=list_ylabels, dict_kwargs=dict_kwargs)
        axs = set_list_list_legends(axs, list_list_legends=list_list_legends, dict_kwargs=dict_kwargs)
    if fig is not None:
        set_list_fig_suptitle(fig, list_fig_suptitles=fig_suptitle, dict_kwargs=dict_kwargs)
    return fig, axs




def from_list_to_subplots(
    list_yvalues,  # List[ndarray([n_lines, ] n_values )]
    list_xvalues=None,  # List[ndarray([1, ] n_values )]
    plt_type: str = "plot",
    fig = None,
    axs = None,
    show: bool = True,
    dict_kwargs = {},
    **kwargs,
):

    # Sort key_words because matplotlib doesn't support unexpected kw
    dict_kwargs = sort_kwargs(dict_kwargs)

    # Default kw values
    dict_kwargs = set_kwargs_default_values(dict_kwargs, plt_type)


    if not isinstance(list_yvalues, list):
        list_yvalues = [list_yvalues]

    # Initalize the figure
    nplots = len(list_yvalues)
    if fig is None or axs is None:
        fig, axs = pretty_subplots(
            nplots=nplots, dict_kwargs=dict_kwargs, sorted_kwargs=True
        )

    is_xvalues_known = True
    is_xvalues_unique = False
    # If no xvalues was given at all
    if list_xvalues is None:
        is_xvalues_known = False
    # A vector was given as xvalues
    elif not isinstance(list_xvalues, list):
        xvalues = list_xvalues
        is_xvalues_unique = True
    # A list of one element (or less than nplots) was given as xvalues
    if isinstance(list_xvalues, list) and len(list_xvalues) < nplots:
        xvalues = list_xvalues
        is_xvalues_unique = True

    # Plot every subplots
    is_axs_squeezed = False
    if not isinstance(axs,  np.ndarray):
        is_axs_squeezed = True
        axs = np.array(axs, ndmin=2)
    for i_subplot, ax in enumerate(axs.flat):

        yvalues = list_yvalues[i_subplot]  # ndarray([n_lines, ] n_values )

        # Unpack yvalues dimensions
        ((n_lines, n_values), yvalues) = unpack_2d_dimensions(yvalues)

        # if there is a different xvalues for each subplot
        if not is_xvalues_unique and is_xvalues_known:
            xvalues = list_xvalues[i_subplot]
        elif not is_xvalues_known:
            xvalues = np.arange(n_values)

        # draw each line in this subplot
        for i_line in range(n_lines):
            if plt_type == "scatter":
                ax.scatter(
                    xvalues,
                    yvalues[i_line],
                    **dict_kwargs['dflt'])
            elif plt_type == "distplot":
                ax = distrib_vs_gaussian(
                    ax=ax,
                    distrib=yvalues,
                    dict_kwargs = dict_kwargs,
                )
            elif plt_type == "density pairplot":
                ax = density_pair_plots(
                    fig=fig,
                    ax=ax,
                    xvalues=xvalues.squeeze(),
                    yvalues=yvalues.squeeze(),
                )
            else:
                ax.plot(
                    xvalues,
                    yvalues[i_line],
                    **dict_kwargs['line'])

        if plt_type == "plot_mean_std":
            fig, ax = plot_mean_and_std(
                xvalues = xvalues,
                yvalues = yvalues,
                dict_kwargs=dict_kwargs,
                sorted_kwargs=True,
                fig = fig,
                ax = ax,
            )
        # handles, labels = ax.get_legend_handles_labels()
        # if handles and labels:
        #     ax.legend(handles, labels)
    # axs = set_list_list_legends(
    #     axs,
    #     list_list_legends=dict_kwargs['text'].pop("list_list_legends" , None),
    #     list_list_lines=None)
    fig, axs = set_text(
        axs,
        fig=fig,
        **dict_kwargs['text']
    )
    if show:
        plt.show()
    plt.rcdefaults()
    if is_axs_squeezed:
        axs = axs[0,0]
    return fig, axs

def from_list_to_pairplots(
    list_values,  # List[ndarray([1, ] n_values )]
    list_labels = None,
    show: bool = True,
    dict_kwargs={},
):
    # First create a PandaDataFrame (makes things easier...)
    nrows = len(list_values)
    nplots = nrows*nrows
    values = np.transpose(np.array(list_values))
    if list_labels is None:
        list_labels = ["Data " + str(i) for i in range(nrows)]
    data = pd.DataFrame(data=values, columns=list_labels)

    # Sort key_words because matplotlib doesn't support unexpected kw
    axs = pd.plotting.scatter_matrix(
        data,
        alpha=0.3,
        figsize=(32,18),
        diagonal="kde"),
    plt.rcdefaults()
    return axs