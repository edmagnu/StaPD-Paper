# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:50:40 2018

@author: labuser
"""

# Delay Scans from changing the Field Angle
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
    """Moving step function average for n points"""
    cumsum = np.array(df['nsignal'].cumsum(skipna=True))
    y = (cumsum[n:] - cumsum[:-n]) / n
    cumsum = np.array(df['wavelengths'].cumsum(skipna=True))
    x = (cumsum[n:] - cumsum[:-n]) / n
    return x, y


def circstat_dscan_plot(ax, data_tot, fits_tot, fname, nave):
    """Standard delay scan plot for CircleDelay figures"""
    mask = (data_tot['Filename'] == fname)
    data = data_tot[mask].copy()
    data.sort_values(by='wavelengths', inplace=True)
    mask = (fits_tot['Filename'] == fname)
    fits = fits_tot[mask].copy()
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='darkgrey')
    x = np.array(data['wavelengths'])
    popt = fits[['y0', 'a', 'phi']].values[0]
    y = model_func_fit(x, *popt)
    ax.plot(x, y, '-', c='k')
    return ax


def circstat_delays():
    """Plot Delay Scans for different field angles"""
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "Circle Static",
                         "rawdata.txt")
    data_tot = pd.read_csv(fname, index_col=0)
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "Circle Static",
                         "fits.txt")
    fits_tot = pd.read_csv(fname, index_col=0)
    mask = (data_tot['group'] == 2)
    # maskf = (fits_tot['group'] == 2)
    print("fx = ", data_tot.loc[mask, 'fx'].unique())
    print("fz = ", data_tot.loc[mask, 'fz'].unique())
    print("fa = ", data_tot.loc[mask, 'fa'].unique())
    print("names = ", data_tot.loc[mask, 'Filename'].unique())
    fnames = data_tot.loc[mask, 'Filename'].unique()
    fnames = fnames[[1, 5, 3, 0, 4, 2]]
    # print(fnames)
    # figure
    fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, sharey=True,
                             figsize=(6, 9))
    nave = 3
    for i in [0, 1, 2, 3, 4, 5]:
        ax = axes[i]
        fname = fnames[i]
        ax = circstat_dscan_plot(ax, data_tot, fits_tot, fname, nave)
    # text
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    angles = [r"$-17.5^\circ$", r"$-11.5^\circ$", r"$-5.7^\circ$",
              r"$-0.0^\circ$", r"$+5.7^\circ$", r"$+11.5^\circ$"]
    for i in [0, 1, 2, 3, 4, 5]:
        axes[i].text(0.98, 0.93, angles[i], transform=axes[i].transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 fontsize=14, bbox=props)
    # beautify
    xticks = np.array([0, 0.5, 1, 1.5, 2.0]) + 1/12
    xticklabels = [r"$\pi/6$", "", r"$\pi + \pi/6$", "", r"$2\pi + \pi/6$"]
    axes[5].set(xticks=xticks, xticklabels=xticklabels)
    # total axes labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                    right='off')
    plt.grid(False)
    plt.xlabel("Delay (rad.)", fontsize=14)
    plt.ylabel("Norm. Signal", fontsize=14, labelpad=20)
    # fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.tight_layout()
    plt.savefig("CircleDelay.pdf")
    return


def circstat_zero():
    """Plot just the smallest circlestat modulation compared to -14 mV/cm."""
    # Get the best Horizontal field plot, Set 1 at -1.44 mV/cm.
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "Circle Static",
                         "rawdata.txt")
    data_tot = pd.read_csv(fname, index_col=0)
    mask = (data_tot['group'] == 1) & (data_tot['fz'] == -1.44)
    data_h = data_tot[mask].copy()
    # Get the peak modulation from DIL - 14 GHz at ~ +/- 14.4 mV/cm
    dname = os.path.join("..", "..", "Data", "StaPD-Analysis")
    fname = os.path.join(dname, "moddata.txt")
    data_tot = pd.read_csv(fname, sep="\t", index_col=0)
    mask_p = data_tot['Filename'] == '2016-09-24\\10_delay.txt'
    data_p = data_tot[mask_p].copy()
    mask_m = data_tot['Filename'] == '2016-09-24\\9_delay.txt'
    data_m = data_tot[mask_m].copy()
    # set wavelengths to radians
    data_h['rad'] = data_h['wavelengths']*2*np.pi
    data_p['rad'] = data_p['wavelengths']*2*np.pi
    data_m['rad'] = data_m['wavelengths']*2*np.pi
    # limit to 0 to 2 wavelenghts
    mask = (data_p['rad'] < 4*np.pi) & (data_p['rad'] > 0)
    data_p = data_p[mask]
    mask = (data_m['rad'] < 4*np.pi) & (data_m['rad'] > 0)
    data_m = data_m[mask]
    mask = (data_h['rad'] < 4*np.pi) & (data_h['rad'] > 0)
    data_h = data_h[mask].copy()
    # sort
    data_h.sort_values(by='rad', inplace=True)
    data_p.sort_values(by='rad', inplace=True)
    data_m.sort_values(by='rad', inplace=True)
    # print(data_m['Filename'].unique())
    # rolling mean
    nave = 3
    data_h['nsignal_rm'] = \
        data_h['nsignal'].rolling(window=nave, center=True).mean()
    data_p['nsignal_rm'] = \
        data_p['nsignal'].rolling(window=nave, center=True).mean()
    data_m['nsignal_rm'] = \
        data_m['nsignal'].rolling(window=nave, center=True).mean()
    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    data_h.plot(x='rad', y='nsignal_rm', ax=axes[1], sharex=True)
    data_p.plot(x='rad', y='nsignal_rm', ax=axes[0], sharex=True)
    data_m.plot(x='rad', y='nsignal_rm', ax=axes[2], sharex=True)
    # text
    # text
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'right'}
    texts = ["(a)",
             "(b)",
             "(c)"]
    for i in [0, 1, 2]:
        axes[i].text(0.98, 0.9, texts[i], transform=axes[i].transAxes,
                 **align, fontsize=14, bbox=props)
    # pretty
    for i in [0, 1, 2]:
        axes[i].set(xticks=[])
    for i in [0, 1, 2]:
        axes[i].legend().remove()
    axes[2].minorticks_off()
    axes[2].set(
            xlim=(-np.pi/6, (4+2/6)*np.pi),
            xticks=np.array([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi]) + np.pi/6,
            xticklabels=[r"$\pi/6$", r"$\pi + \pi/6$", r"$2\pi + \pi/6$",
                         r"$3\pi + \pi/6$", r"$4\pi + \pi/6$"],
            xlabel="",
            ylabel="",
            ylim=(0.27, 0.43))
    # Total Labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                    right='off')
    plt.grid(False)
    plt.xlabel("Delay (rad.)", fontsize=14)
    plt.ylabel("Norm. Signal", fontsize=14)
    # final
    fig.tight_layout()
    fig.savefig("CircleDelay.pdf")
    fig.savefig(os.path.join("..", "CircleDelay.pdf"))
    return


# main
circstat_zero()
