import numpy as np
from os import makedirs
from os.path import isfile
from shutil import copy2, copyfile


PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/toyexamples/"
SUB_DIRS = ["spaghettis/", "true_distrib/", "spaghettis_true_distrib/", "data/"]
PATH_BEST = PATH_FIG_PARENT + "best/"
PATH_DISTRIB = ["2/gaussian/", "3/gaussian/", "N/gaussian/", "2/uniform/"]

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

def select_best():
    for i_type_best in range(len(best)):
        for name_fig in best[i_type_best]:
            for i_plot, plot_dir in enumerate(SUB_DIRS):
                source_name = PATH_FIG_PARENT + PATH_DISTRIB[i_type_best] + plot_dir + name_fig
                dest_name = PATH_BEST + PATH_DISTRIB[i_type_best] + plot_dir
                makedirs(dest_name, exist_ok = True)
                # Copy data
                if i_plot == 3:
                    file_name = source_name + "_members.npy"
                    copy2(file_name, dest_name)
                    file_name = source_name + "_xvalues.npy"
                    copy2(file_name, dest_name)
                    file_name = source_name + "_weights.npy"
                    if isfile(file_name):
                        copy2(file_name, dest_name)
                # Copy figs
                else:
                    file_name = source_name + ".png"
                    copy2(file_name, dest_name)

def rename_best():
    for i_type_best in range(len(best)):
        count = 0
        for name_fig in best[i_type_best]:
            for i_plot, plot_dir in enumerate(SUB_DIRS + ['GIF/']):
                source_name = PATH_BEST + PATH_DISTRIB[i_type_best] + plot_dir + name_fig
                dest_path = PATH_BEST + "renamed/" + plot_dir
                makedirs(dest_path, exist_ok = True)
                # Copy data
                if i_plot == 3 :
                    file_name = source_name + "_members.npy"
                    copy2(file_name, dest_path+str(count)+"_members.npy")
                    file_name = source_name + "_xvalues.npy"
                    copy2(file_name, dest_path+str(count)+"_xvalues.npy")
                    file_name = source_name + "_weights.npy"
                    if isfile(file_name):
                        copy2(file_name, dest_path+str(count)+"_weights.npy")
                # Copy figs
                elif i_plot < 3:
                    file_name = source_name + ".png"
                    copy2(file_name, dest_path+str(count)+".png")
                # Copy GIF
                elif i_plot == 4:
                    file_name = source_name + ".gif"
                    copy2(file_name, dest_path+str(count)+".gif")

            count += 1

#select_best()
rename_best()