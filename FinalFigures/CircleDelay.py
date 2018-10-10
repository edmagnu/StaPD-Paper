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
import scipy.optimize


def model_func(x, y0, a, phi):
    """Sinusoidal plus offset model for delay scan phase dependence.
    "x" is the delay in wavelengths
    "y" is the normalized Rydberg signal.
    Returns model dataframe and fit parameters.
    """
    return y0 + a*np.sin(2*np.pi*x + phi)


def model_p0(x, y):
    """Guesses reasonable starting parameters p0 to pass to model_func().
    x and y are pandas.Series
    Returns p0, array of [y0, a, phi]"""
    # y0 is the mean
    y0 = y.mean()
    # phi from averaged maximum and minimum
    yroll = y.rolling(window=9, center=True).mean()
    imax = yroll.idxmax()
    imin = yroll.idxmin()
    phi = ((x[imax] % 1) + ((x[imin]-0.5) % 1)) / 2
    phi = ((phi-0.25) % 1)*np.pi
    # a is max and min
    mx = yroll.max()
    mn = yroll.min()
    a = (mx-mn)/2
    return [y0, a, phi]


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
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True,
                             figsize=(3.375, 3.375))
    phase_plot(data_h, [True]*len(data_h), 3, axes[1], ls='-')
    # data_h.plot(x='rad', y='nsignal_rm', ax=axes[1], sharex=True)
    data_p.plot(x='rad', y='nsignal_rm', ax=axes[0], sharex=True)
    data_m.plot(x='rad', y='nsignal_rm', ax=axes[2], sharex=True)
    # text
    # text
    props = dict(boxstyle='round', facecolor='white', edgecolor='white',
                 alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'right'}
    texts = ["(a)", "(b)", "(c)"]
    for i in [0, 1, 2]:
        axes[i].text(0.98, 0.9, texts[i], transform=axes[i].transAxes,
                 **align, fontsize=8, bbox=props)
    # pretty
    for i in [0, 1, 2]:
        axes[i].set(xticks=[])
    for i in [0, 1, 2]:
        axes[i].legend().remove()
    axes[2].minorticks_off()
    axes[2].set(
            xlim=(-np.pi/6, (4+2/6)*np.pi),
            # xticks=np.array([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi]) + np.pi/6,
            # xticklabels=[r"$\pi/6$", r"$\pi + \pi/6$", r"$2\pi + \pi/6$",
            #              r"$3\pi + \pi/6$", r"$4\pi + \pi/6$"],
            xticks=np.array([0, 1, 2, 3, 4])*np.pi,
            xticklabels=[r"0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"],
            xlabel="",
            ylabel="",
            ylim=(0.27, 0.43))
    # style
    for i in [0, 1, 2]:
        axes[i].tick_params(labelsize=8, direction='in')
    # Total Labels
    axes2 = fig.add_subplot(111, frameon=False)
    axes2.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                      right='off')
    axes2.grid(False)
    axes2.set_xlabel("Delay (rad.)", fontsize=9)
    axes2.set_ylabel("Norm. Signal", fontsize=9)
    # final
    fig.tight_layout()
    fig.savefig("CircleDelay.pdf")
    fig.savefig(os.path.join("..", "CircleDelay.pdf"))
    return data_p, data_m, data_h


def Circle_Delay():
    # ==========
    # figure
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col', sharey='row',
                           figsize=(3.375, 3.375))
    nave = 3
    ls = '-'
    # ==========
    # Get the best Horizontal field plot, Set 1 at -1.44 mV/cm.
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "Circle Static",
                         "rawdata.txt")
    data_tot = pd.read_csv(fname, index_col=0)
    mask = (data_tot['group'] == 1) & (data_tot['fz'] == -1.44)
    data = data_tot[mask].copy()
    # get fit parameters for Horizontal
    p0 = model_p0(data['wavelengths'], data['nsignal'])
    popt, pcov = scipy.optimize.curve_fit(
            model_func, data['wavelengths'].astype(float),
            data["nsignal"].astype(float), p0)
    data['y0'] = popt[0]
    data['a'] = popt[1]
    data['phi'] = popt[2]
    # trim
    mask = (data['wavelengths'] < 2) & (data['wavelengths'] > 0)
    data = data[mask]
    # save and plot horizontal
    data_h = data.copy()
    phase_plot(data, [True]*len(data), nave, ax[1], ls)
    # Get the peak modulation from DIL - 14 GHz at ~ + 14.4 mV/cm
    dname = os.path.join("..", "..", "Data", "StaPD-Analysis")
    fname = os.path.join(dname, "moddata.txt")
    data_tot = pd.read_csv(fname, sep="\t", index_col=0)
    mask = data_tot['Filename'] == '2016-09-24\\10_delay.txt'
    data = data_tot[mask].copy()
    # trim
    mask = (data['wavelengths'] < 2) & (data['wavelengths'] > 0)
    data = data[mask]
    # save and plot
    data_p = data.copy()
    phase_plot(data, [True]*len(data), nave, ax[0], ls)
    # Get the peak modulation from DIL - 14 GHz at ~ + 14.4 mV/cm
    mask = data_tot['Filename'] == '2016-09-24\\9_delay.txt'
    data = data_tot[mask].copy()
    # trim
    mask = (data['wavelengths'] < 2) & (data['wavelengths'] > 0)
    data = data[mask]
    # save and plot
    data_m = data.copy()
    phase_plot(data, [True]*len(data), nave, ax[2], ls)
    # ==========
    # xlims
    xlims = ax[0].get_xlim()
    xlims = (xlims[0], xlims[1]*1.05)
    ax[2].set_xlim(xlims)
    # Time axes
    ax2 = ax[0].twiny()
    ax2.tick_params('x')
    tps = 1/(15.932*1e9)*1e12
    xlims = ax[0].get_xlim()
    tlims = tuple(np.array(xlims)*tps)
    ax2.set_xlim(tlims)
    ax2.set_xticks(range(0, 140, 20))
    ax2.tick_params(labelsize=8, direction='in')
    ax2.set_xlabel(r"Delay $t_0$ (ps)", fontsize=8)
    for side in ['bottom', 'right', 'left']:
        ax2.spines[side].set_visible(False)
    # Labels
    xticks = [0, 0.5, 1, 1.5, 2]
    ax[2].set_xticks(xticks)
    xticklabels = ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"]
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_xlabel(r"Delay $\omega t_0$ (rad.)", fontsize=8)
    for i in [0, 1, 2]:
        ax[i].set_ylabel("Norm. Signal", fontsize=8)
        ax[i].set_yticks([0.30, 0.35, 0.40])
        ax[i].tick_params(labelsize=8, direction='in')
    for i in [0, 1]:
        ax[i].set_xticklabels(xticklabels, visible=False)
    # Text
    props = dict(boxstyle='round', facecolor='white', edgecolor='white',
                 alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'right'}
    texts = ["(a)", "(b)", "(c)"]
    for i in [1, 2]:
        ax[i].text(0.98, 0.85, texts[i], bbox=props, **align,
                   transform=ax[i].transAxes, fontsize=8)
    for i in [0]:
        align['verticalalignment'] = 'bottom'
        ax[i].text(0.98, 0.15, texts[i], bbox=props, **align,
                   transform=ax[i].transAxes, fontsize=8)
    # final
    fig.tight_layout()
    fig.savefig("CircleDelay.pdf")
    fig.savefig(os.path.join("..", "CircleDelay.pdf"))
    return data_p, data_h, data_m
    

# main
data_h = Circle_Delay()
