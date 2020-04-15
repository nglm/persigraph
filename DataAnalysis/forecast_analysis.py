import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from math import ceil, floor, sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# =========================================================
# Forecast analysis
# =========================================================




def find_best_shift(
    member_ref,  # ndarray(n_time, [n_long, n_lat])
    member_add,
    time_window:(int,int) = None, # time window where the member are compared
    time_shift_max: float = 1, 
    value_shift_max: float = 0.2,
):
    n_time = member_ref.shape[0]
    if time_window is None:
        (t_start,t_end) = (0,n_time)
    # Compute the best shift in y_axis values
    v_shift = np.mean(member_ref) - np.mean(member_add)
    if v_shift > 0:
        v_shift = min(v_shift,value_shift_max)
    else:
        v_shift = max(v_shift,-value_shift_max)
    member_shift = member_add + v_shift
    # Initialize min values
    t_max = time_shift_max
    while (t_end + t_max >= n_time) and (t_max>0):
        t_max -= 1
    t_shift = t_max
    rmse_min = mean_squared_error(
            member_ref[t_start:t_end], 
            member_shift[t_start+t_shift:t_end+t_shift],
            squared=True
            )
    # Find the min values within the time_window 
    for t in range(-time_shift_max, t_max):
        if (t_start + t >= 0) and (t_end + t < n_time):
            rmse = mean_squared_error(
                member_ref[t_start:t_end], 
                member_shift[t_start+t:t_end+t],
                squared = True
                )
            if rmse < rmse_min:
                t_shift = t
                rmse_min = rmse
    return(t_shift,v_shift,rmse_min)

def mat_rmse_to_member_ref(
    list_members,
    member_ref,
    find_best_shift: bool = False,
):
    return None

def rmse_between_members(
    members,  #ndarray(n_members, n_time [n_long, n_lat])
):
    
    rmse = mean_squared_error(
        member_ref[t_start:t_end], 
        member_shift[t_start+t:t_end+t],
        squared = True
    )
    return(None) #ndarray(n_members, n_members [n_long, n_lat])
    
def diff_with_member_ref():
    return None

def list_members_agree_with_member_ref(
    list_members,  # List[ndarray, n_time, [n_long, n_lat]]
    is_best_shift=False
):
    return None
    

def analyse_forecast(
    list_members,   # List[ndarray, n_time, [n_long, n_lat]]
):
    return None