# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:55:56 2018

@author: labuser
"""

# Delay Scans from DIL + 2 GHz as field is increased.
# For Static Field Recombination paper

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def model_func_fit(x, y0, a, phi):
    """Sinusoidal plus offset model for delay scan phase dependence.
    "x" is the delay in wavelengths
    "y" is the normalized Rydberg signal.
    Returns model dataframe and fit parameters.
    """
    return y0 + a*np.sin(2*np.pi*x + phi)


def running_mean(df, n):
    """Running square window mean of n points"""
    cumsum = np.array(df['nsignal'].cumsum(skipna=True))
    y = (cumsum[n:] - cumsum[:-n]) / n
    cumsum = np.array(df['wavelengths'].cumsum(skipna=True))
    x = (cumsum[n:] - cumsum[:-n]) / n
    return x, y


def phase_plot(data_tot, mask, nave, ax, ls):
    """Standardize the process of masking data, averaging, fitting, and then
    plotting the averaged signal and fit."""
    data = data_tot[mask].copy(deep=True)
    data.sort_values(by='wavelengths', inplace=True)
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='darkgrey')
    x = np.array(data['wavelengths'])
    popt = np.array(data[['y0', 'a', 'phi']].iloc[0])
    y = model_func_fit(x, *popt)
    ax.plot(x, y, ls, c='k', lw=3)
    return ax


def FieldPhase():
    """From DIL + 2GHz, plot select delay scans showing the behavior in
    increasing static field."""
    # load
    dname = os.path.join("..", "..", "Data", "StaPD-Analysis")
    fname = os.path.join(dname, "moddata.txt")
    data_tot = pd.read_csv(fname, sep="\t", index_col=0)
    # pick out DIL + 2 GHz
    mask_p2 = (data_tot["DL-Pro"] == 365872.6)
    mask_p2 = mask_p2 & (data_tot["Attn"] == 44)
    # excluded data
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    for fname in excluded:
        mask_p2 = mask_p2 & (data_tot["Filename"] != fname)
    statics = np.sort(data_tot['Static'].unique())
    # plot 0, 7.2, -7.2, 36, 108 mV/cm
    istatics = [21, 23, 19, 29, -7]
    fig, ax = plt.subplots(figsize=(3.375, 3.375))
    nave = 3
    for i in istatics:
        print("{0} \t {1} mV \t {2} mV/cm".format(
                i, statics[i], np.round(statics[i]*0.1*0.72, 1)))
        mask = mask_p2 & (data_tot['Static'] == statics[i])
        if i == 19:
            ls = '--'
        else:
            ls = '-'
        ax = phase_plot(data_tot, mask, nave, ax, ls)
    # tidy
    ax.set_xlabel(r"Delay $\omega t_0$ (rad.)", fontsize=9)
    ax.set_ylabel("Norm. Signal", fontsize=9)
    ax.set(xticks=np.arange(0, 3.5, 0.5),
           xticklabels=["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$",
                        r"$5\pi$", r"$6\pi$"],
           ylim=(-0.03, 0.37))
    ax.tick_params(labelsize=8, direction='in')
    # twin axes
    ax2 = ax.twiny()
    ax2.tick_params('x')
    tps = 1/(15.932*1e9)*1e12
    xlims = ax.get_xlim()
    tlims = tuple(np.array(xlims)*tps)
    # xticks = ax.get_xticks()
    # tticks = tuple(np.array(xticks)*tps)
    ax2.set_xlim(tlims)
    ax2.set_xticks(range(0, 190, 20))
    ax2.tick_params(labelsize=8, direction='in')
    ax2.set_xlabel(r"Delay $t_0$ (ps)", fontsize=9)
    for side in ['bottom', 'right', 'left']:
        ax2.spines[side].set_visible(False)
    # labels
    props = dict(boxstyle='round', facecolor='white', edgecolor='white',
                 alpha=1.0)
    align = {'verticalalignment': 'center',
             'horizontalalignment': 'right'}
    ax.text(xlims[1] - 0.1, 0.28, "0.0 mV/cm", **align, bbox=props,
            fontsize=8)
    ax.text(xlims[1] - 0.1, 0.16, "+/- 7.2 mV/cm", **align, bbox=props,
            fontsize=8)
    # ax.text(xlims[1], 0.25, "-7.2 mV/cm", **align, bbox=props)
    ax.text(xlims[1] - 0.1, 0.06, "36.0 mV/cm", **align, bbox=props,
            fontsize=8)
    ax.text(xlims[1] - 0.1, -0.01, "108.0 mV/cm", **align, bbox=props,
            fontsize=8)
    # save
    fig.tight_layout()
    fig.savefig("FieldPhase.pdf")
    fig.savefig(os.path.join("..", "FieldPhase.pdf"))
    return data_tot


# main script
FieldPhase()
