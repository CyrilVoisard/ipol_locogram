# Objective : Plot and save the locogram.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from tslearn import metrics
import scipy.stats
import os


def locogram(steps_lim_bis, s_tot, n_tot, n_r, n_l, ref, output):

    # one table for the right foot, one for the left foot
    events_right = steps_lim_bis[steps_lim_bis["Foot"]== 1]
    events_left = steps_lim_bis[steps_lim_bis["Foot"] == 0]

    nb_tot = len(steps_lim_bis)
    n_r = len(events_right)
    n_l = len(events_left)

    # we build concatenation to simply locogram matrix computation : first left, then right. 
    # for time series : gyration and jerk
    gyr_conc, jerk_conc, offset = concatenate_signals()

    # for gait events : heel-strikes (offset taken into account for right events)
    hs_conc = concatenate_events()
    
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
            s_y2 = np.array([jerk / np.max(jerk),
                             gyr / np.max(abs(gyr))])
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


def rotagram(steps_lim_bis, seg_lim, data_lb, output):
    seg_lim = pd.DataFrame(seg_lim)
    os.chdir(output)
    
    # figure and color definition
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [20, 1, 1]}, figsize=(12, 10))
    ax[1].set_title('        Stance', fontsize=10, weight='bold')
    ax[2].set_title('phase       ', fontsize=10, weight='bold')
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[1].set_xticks([0])
    ax[2].set_xticks([0])
    ax[1].set_xticklabels(['Left\nfoot'], fontsize=8, horizontalalignment='center')
    ax[2].set_xticklabels(['Right\nfoot'], fontsize=8, horizontalalignment='center')

    linewidth=3
    color_r, line_r = 'blue', '-'
    color_l, line_l = 'red', '-'
    color_r_2, line_r = 'green', '-'
    color_l_2, line_l = 'orange', '-'

    # one table for the right foot, one for the left foot
    events_right = steps_lim_bis[steps_lim_bis["Foot"]== 1]
    events_left = steps_lim_bis[steps_lim_bis["Foot"] == 0]

    t = data_lb["PacketCounter"] - seg_lim.iloc[1, 0]/ 100
    sc = - data_lb["Gyr_X"]

    # Fine black cumulative curve for u-turn
    cumulative_curve = np.cumsum(sc[seg_lim.iloc[1, 0]-50: seg_lim.iloc[2, 0]+50])
    coef = np.sign(cumulative_curve.iloc[-1])* 180 / cumulative_curve.iloc[-1]
    cumulative_curve = np.sign(cumulative_curve.iloc[-1]) * cumulative_curve * 180 / cumulative_curve.iloc[-1]
    leg3 = ax[0].plot(cumulative_curve, t[seg_lim.iloc[1, 0]-50: seg_lim.iloc[2, 0]+50], 'k', linewidth=2)

    W = abs(min(cumulative_curve))
    W1 = abs(max(cumulative_curve))
    if W1 > W:
        W = W1
    W = max(abs(cumulative_curve))

    if max(cumulative_curve) > 100:
        ax[0].annotate('U-turn to the right', xy=(2, 1), xytext=(30, (-1)), fontsize=10, weight='bold')
    else:
        ax[0].annotate('U-turn to the left', xy=(2, 1), xytext=(-W + 30, (-1)), fontsize=10, weight='bold')


    # rotation plot for each stance phase
    for y in range(len(events_right)):
        if y != len(events_right)-1:
            if inside(events_right["HS"].tolist()[y], events_right["TO"].tolist()[y+1], seg_lim):
                # plot
                leg_rf = ([events_right["HS"].tolist()[y], events_right["TO"].tolist()[y+1]] - seg_lim.iloc[1, 0]) / 100
                leg1 = ax[2].plot([0, 0], leg_rf, line_r, linewidth=linewidth, color=color_r)
        if inside(events_right["TO"].tolist()[y], events_right["HS"].tolist()[y], seg_lim):
            """
            ax[0].plot(np.cumsum(sc[int(events_right["HS"].tolist()[y]):int(events_right["TO"].tolist()[y+1])]) * coef,
                       t[int(events_right["HS"].tolist()[y]):int(events_right["TO"].tolist()[y+1])],
                       line_r, linewidth=linewidth, color=color_r)
            """
            ax[0].plot(np.cumsum(sc[int(events_right["TO"].tolist()[y]):int(events_right["HS"].tolist()[y])]) * coef,
                       t[int(events_right["TO"].tolist()[y]):int(events_right["HS"].tolist()[y])],
                       line_r, linewidth=linewidth, color=color_r)

    for y in range(len(events_left)-1):
        if y != len(events_left)-1:
            if inside(events_left["HS"].tolist()[y], events_left["TO"].tolist()[y+1], seg_lim):
                # leg_lf = ([events_left["HS"].tolist()[y]  - len(data_lb), events_left["TO"].tolist()[y+1] - len(data_lb)] - seg_lim.iloc[1, 0]) / 100
                leg_lf = ([events_left["HS"].tolist()[y], events_left["TO"].tolist()[y+1]] - seg_lim.iloc[1, 0]) / 100
                leg2 = ax[1].plot([0, 0], leg_lf, line_r, linewidth=linewidth, color=color_l)
        if inside(events_left["TO"].tolist()[y], events_left["HS"].tolist()[y], seg_lim):
            """
            ax[0].plot(np.cumsum(sc[int(events_left["HS"].tolist()[y] - len(data_lb)):int(events_left["TO"].tolist()[y+1] - len(data_lb))]) * coef,
                       t[int(events_left["HS"].tolist()[y] - len(data_lb)):int(events_left["TO"].tolist()[y+1] - len(data_lb))],
                       line_l, linewidth=linewidth, color=color_l)
            """
            ax[0].plot(np.cumsum(sc[int(events_left["TO"].tolist()[y]):int(events_left["HS"].tolist()[y])]) * coef,
                       t[int(events_left["TO"].tolist()[y]):int(events_left["HS"].tolist()[y])],
                       line_l, linewidth=linewidth, color=color_l)
            # ax[0].plot(np.cumsum(sc[int(events_left["HS"].tolist()[y]):int(events_left["TO"].tolist()[y+1])]) * coef,
              #         t[int(events_left["HS"].tolist()[y]):int(events_left["TO"].tolist()[y+1])],
               #        line_l, linewidth=linewidth, color=color_l)

        # coloring the areas of the figure
        ax[0].add_patch(
            patches.Rectangle(
                (-W, (seg_lim.iloc[0, 0] - seg_lim.iloc[1, 0]) / 100),  # (x,y)
                W,  # width
                (seg_lim.iloc[3, 0] - seg_lim.iloc[0, 0]) / 100,  # height
                facecolor="red",
                alpha=0.01
            )
        )
        ax[0].add_patch(
            patches.Rectangle(
                (0, (seg_lim.iloc[0, 0] - seg_lim.iloc[1, 0]) / 100),  # (x,y)
                W,  # width
                (seg_lim.iloc[3, 0] - seg_lim.iloc[0, 0]) / 100,  # height
                alpha=0.01,
                color=color_r
            )
        )
        ax[0].add_patch(
            patches.Rectangle(
                (-W, 0),  # (x,y)
                W * 2,  # width
                (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100,  # height
                alpha=0.1,
                facecolor="yellow", linestyle='dotted'
            )
        )

        ax[1].add_patch(
            patches.Rectangle(
                (-W, (seg_lim.iloc[0, 0] - seg_lim.iloc[1, 0]) / 100),  # (x,y)
                W * 2,  # width
                (seg_lim.iloc[3, 0] - seg_lim.iloc[0, 0]) / 100,  # height
                alpha=0.01,
                facecolor="red"
            )
        )

        ax[1].add_patch(
            patches.Rectangle(
                (-W, 0),  # (x,y)
                W * 2,  # width
                (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100,  # height
                alpha=0.1,
                facecolor="yellow", linestyle='dotted'
            )
        )

        ax[2].add_patch(
            patches.Rectangle(
                (-W, (seg_lim.iloc[0, 0] - seg_lim.iloc[1, 0]) / 100),  # (x,y)
                W * 2,  # width
                (seg_lim.iloc[3, 0] - seg_lim.iloc[0, 0]) / 100,  # height
                alpha=0.01,
                facecolor="blue"
            )
        )
        ax[2].add_patch(
            patches.Rectangle(
                (-W, 0),  # (x,y)
                W * 2,  # width
                (seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100,  # height
                alpha=0.1,
                facecolor="yellow", linestyle='dotted'
            )
        )

        # Légendes
        fig.suptitle('Rotagram', fontsize=13, weight='bold')
        ax[0].set_xticks([-W, (-W / 2), 0, W / 2, W])
        ax[0].set_xticklabels(['180°', '90°', '0°', '90°', '180°'], fontsize=8)
        ax[0].set_title('Trunk rotation angle (axial plane)', weight='bold', size=10)

        ax[0].tick_params(axis=u'both', which=u'both', length=0)
        e = str(((seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100)) + 's.'
        ax[1].tick_params(axis=u'both', which=u'both', length=0)
        ax[1].spines['right'].set_visible(False)
        ax[2].set_yticks(
            [((seg_lim.iloc[0, 0] - seg_lim.iloc[1, 0]) / 100) + 1.1, -0.5, ((seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100) / 2,
             ((seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100) + 0.5,
             ((seg_lim.iloc[3, 0] - seg_lim.iloc[1, 0]) / 100) - 1.1])
        e = str(((seg_lim.iloc[2, 0] - seg_lim.iloc[1, 0]) / 100)) + 's.'
        ax[2].set_yticklabels(['Gait\nstart', 'U-Turn\nstart', e, 'U-Turn\nend', 'Gait\nend'],
                              fontsize=8)
        ax[2].tick_params(axis=u'both', which=u'both', length=0)
        ax[2].spines['left'].set_visible(False)
        ax[2].yaxis.tick_right()

        ax[0].legend([leg1[0], leg2[0], leg3[0]], ['right', 'left', 'cumulative_angle'])
    
    # save fig
    plt.savefig(fname=("rota.svg"))

    return None


def inside(ge_1, ge_2, seg_lim): 
    if ge_1 > ge_2:
        ge_1, ge_2 = ge_2, ge_1
    if ge_1 <= seg_lim.iloc[1, 0]:
        if ge_2 <= seg_lim.iloc[1, 0]:
            return True
        else:
            if ge_2 > seg_lim.iloc[2, 0]:
                return False
            else: 
                return True
    else: 
        return True
    
        
    
