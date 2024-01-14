# Objective : Plot and save the locogram.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from tslearn import metrics
import scipy.stats
import os
import pandas as pd


def locogram(data_lf, data_rf, steps_lim, output):

    # one table for the right foot, one for the left foot
    events_right = steps_lim[steps_lim["Foot"]== 1]
    events_left = steps_lim[steps_lim["Foot"] == 0]

    n_tot = len(steps_lim)
    n_r = len(events_right)
    n_l = len(events_left)

    # we build concatenation to simply locogram matrix computation : first left, then right. 
    # for time series : gyration and jerk
    gyr_conc, jerk_conc, offset = concatenate_signals(data_lf, data_rf)

    # for gait events : heel-strikes (offset taken into account for right events)
    hs_conc = concatenate_events(steps_lim)
    
    pea = np.zeros((n_tot, n_tot))

    for z1 in range(n_tot):
        for z2 in range(z1+1):
            # parameters
            start1 = hs_conc[z1]
            end1 = hs_conc[z1+1]
            s_y1 = np.array([jerk_conc[start1:end1] / np.max(jerk_conc[start1:end1]),
                             gyr_conc[start1:end1] / np.max(abs(gyr_conc[start1:end1]))])
            s_y1 = s_y1.transpose()

            start2 = hs_conc[z2]
            end2 = hs_conc[z2+1]
            s_y2 = np.array([jerk_conc[start2:end2] / np.max(jerk_conc[start2:end2]),
                             gyr_conc[start2:end2] / np.max(abs(gyr_conc[start2:end2]))])
            s_y2 = s_y1.transpose()
        
            r = 2
            path_min, sim_min = metrics.dtw_path(s_y1, s_y2, global_constraint="itakura", itakura_max_slope=r)
            
            pea[z1][z2] = sim_min
            pea[z2][z1] = pea[z1][z2]

    # Plot
    x1_bis = np.arange(1, n_r + 1, 5)
    # x1 = np.sort(n_r + 1 - x1_bis)
    x2_bis = np.arange(1, n_l + 1, 5)
    # x2 = np.sort(n_l + 1 - x2_bis)

    fig_loco, axs = plt.subplots(2, 2, sharex='all', sharey="all", figsize=(12, 10))
    plt.ion()

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    h0 = axs[0, 0].imshow(pea, origin='lower', cmap=cmap, norm=norm)

    axs[0, 0].imshow(pea[n_r:n_tot, 0:n_r], origin='lower', cmap=cmap, norm=norm)
    axs[1, 0].imshow(pea[0:n_r, 0:n_r], origin='lower', cmap=cmap, norm=norm)
    axs[0, 1].imshow(pea[n_r:n_tot, n_r:n_tot], origin='lower', cmap=cmap, norm=norm)
    axs[1, 1].imshow(pea[0:n_r, n_r:n_tot], origin='lower', cmap=cmap, norm=norm )

    axs[0, 0].set_ylabel("Left steps", fontsize=20)
    axs[1, 1].set_xlabel("Left steps", fontsize=20)
    axs[1, 0].set_xlabel("Right steps", fontsize=20)
    axs[1, 0].set_ylabel("Right steps", fontsize=20)

    axlist = [axs[1, 0], axs[0, 0], axs[0, 1], axs[1, 1]]

    cbar = fig_loco.colorbar(h0, ax=axlist, ticks=np.arange(np.min(pea), np.max(pea), 5),
                             orientation='vertical', fraction=0.04, pad=0.04)
    cbar.ax.set_yticklabels(['', 'Non similaire', '', '', '', '', '', '', '', 'Très similaire', ''], rotation=90,
                            fontsize=20)

    plt.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.85,
                        top=0.92,
                        wspace=0.05,
                        hspace=-0.01)

    fig_loco.suptitle('Locogram for ' + ref, fontsize=30)
    fig_loco.set_size_inches(20, 20)

    # save fig
    os.chdir(output)
    plt.ioff()
    plt.savefig(fname=("_loco.svg"), bbox_inches='tight')

    return None


def concatenate_signals(data_lf, data_rf):

    gyr_lf = data_lf["Gyr_Y"]
    gyr_rf = data_rf["Gyr_Y"]
    gyr_conc = np.concatenate((np.transpose(gyr_lf), np.transpose(gyr_rf)), axis=0)

    jerk_lf = calculate_jerk(data_lf)
    jerk_rf = calculate_jerk(data_rf)
    jerk_conc = np.concatenate((np.transpose(jerk_lf), np.transpose(jerk_rf)), axis=0)

    offset = int(len(gyr_lf))

    return gyr_conc, jerk_conc, offset


def concatenate_events(steps_lim):
    hs_lf = steps_lim[steps_lim["Foot"]== 0]["HS"]
    hs_rf = steps_lim[steps_lim["Foot"]== 1]["HS"]

    hs_conc = np.concatenate((np.transpose(hs_lf), np.transpose(hs_rf)), axis=0)
    
    return hs_conc


def calculate_jerk(data, freq=100):
    """Calculate jerk from acceleration data. 

    Parameters
    ----------
        data {dataframe} -- pandas dataframe.
        freq {int} -- acquisition frequency in Herz.

    Returns
    -------
        z {array} -- jerk time series.
    """
    
    jerk_tot = np.sqrt(
        np.diff(data["FreeAcc_X"]) ** 2 + np.diff(data["FreeAcc_Z"]) ** 2 + np.diff(data["FreeAcc_Y"]) ** 2)
    jerk_tot = np.array(jerk_tot.tolist() + [0])
    y = pd.DataFrame(jerk_tot)
    # Rolling the jerk with a center window 
    y_mean = y.rolling(9, center=True, win_type='boxcar').sum()
    y_mean = y_mean.fillna(0)
    # Transpose to have a numpy array 
    z = y_mean.to_numpy().transpose()[0]
    
    return z
