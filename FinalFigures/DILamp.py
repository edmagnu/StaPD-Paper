# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:17:52 2018

@author: labuser
"""

# Mean, Modulation, Phase plots for fits from DIL + 2 GHz
# For Static Field Recombination paper

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def xticks_2p():
    """Return ticks and ticklabels starting at pi/6 separated by pi/2"""
    ticklabels = [r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"]
    ticks = [np.pi/6, 4*np.pi/6, 7*np.pi/6, 10*np.pi/6]
    return ticks, ticklabels


def ebar_pam_prep(fits, mask, excluded, ph_th):
    """Prepare fit parameters into a plottable form."""
    fsort = fits[mask].copy(deep=True)
    # if amp is negative, negate and shift phi by pi
    mask = (fsort['a'] < 0)
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + np.pi
    fsort['phi'] = fsort['phi'] % (2*np.pi)
    # amplitude -> pk-pk
    fsort['a'] = 2*fsort['a']
    # mV/cm
    fsort['Static'] = fsort['Static']*0.72*0.1
    # manually exclude bad data runs
    for fname in excluded:
        fsort = fsort[fsort['Filename'] != fname]
    fsort.sort_values(by=['Static'], inplace=True)
    # translate
    data = pd.DataFrame()
    data['Ep'] = fsort['Static']
    data['a'] = fsort['a']
    data['sig_a'] = fsort['sig_a']
    data['x0'] = fsort['phi']
    data['sig_x0'] = fsort['sig_phi']
    data['y0'] = fsort['y0']
    data['sig_y0'] = fsort['sig_y0']
    return data


def ebar_pam_plot(data, ph_th, ax0, ax1, ax2):
    """Standarize plot of amp, mean phase with error bars."""
    # phase threshold
    mask = (data['x0'] >= (ph_th - np.pi)) & (data['x0'] < ph_th)
    data.loc[mask, 'a'] = -data[mask]['a']
    mask = (data['x0'] >= (ph_th + np.pi))
    data.loc[mask, 'a'] = -data[mask]['a']
    # phase
    mask = (data['x0'] < (ph_th - np.pi))
    data.loc[mask, 'x0'] = data.loc[mask, 'x0'] + np.pi
    mask = (data['x0'] >= (ph_th - np.pi)) & (data['x0'] < ph_th)
    data.loc[mask, 'x0'] = data.loc[mask, 'x0'] + 0
    mask = (data['x0'] >= ph_th) & (data['x0'] < (ph_th + np.pi)) 
    data.loc[mask, 'x0'] = data.loc[mask, 'x0'] - np.pi
    mask = (data['x0'] >= (ph_th + np.pi))
    data.loc[mask, 'x0'] = data.loc[mask, 'x0'] - 2*np.pi
    # plot
    # pk-pk amp.
    ax = ax0
    ax.axhline(0, color='grey')
    # data.plot(x='Ep', y='a', yerr='sig_a', ax=ax)
    ax.plot(data['Ep'], data['a'], '-', c='C0')
    ax.plot(data['Ep'], data['a'], '.', c='k')
    ax.errorbar(data['Ep'], data['a'], yerr=data['sig_a'],
                fmt='none', ecolor='k',
                capsize=3)
    # mean
    ax = ax1
    ax.axhline(0, color='grey')
    ax.plot(data['Ep'], data['y0'], '-', c='C0')
    ax.plot(data['Ep'], data['y0'], '.', c='k')
    ax.errorbar(data['Ep'], data['y0'], yerr=data['sig_y0'],
                fmt='none', ecolor='k',
                capsize=3)
    # phase
    ax = ax2
    ax.axhline(np.pi/6, color='grey')
    ax.axhline(7*np.pi/6, color='grey')
    ax.plot(data['Ep'], data['x0'], '-', c='C0')
    ax.plot(data['Ep'], data['x0'], '.', c='k')
    ax.errorbar(data['Ep'], data['x0'], yerr=data['sig_x0'],
                fmt='none', ecolor='k',
                capsize=3)
    return ax0, ax1, ax2


def amp_mean_phase_plot(fits, mask, excluded, title):
    """Standardize figure for amp, mean, phase plotting."""
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3.375, 3.375*1.2))
    ph_th = 5.5/6*np.pi
    data = ebar_pam_prep(fits, mask, excluded, ph_th)
    ax[0], ax[1], ax[2] = ebar_pam_plot(data, ph_th, ax[0], ax[1], ax[2])
    # x
    # xlims = (-200, 200)
    # ax[2].set(xlim=xlims, xlabel=r"Field $E_S$ (mV/cm)")
    ax[2].set_xlabel(r"Field $E_S$ (mV/cm)", fontsize=9)
    # y
    ax[0].set_ylabel("Pk-Pk Modulation", fontsize=9)
    ax[1].set_ylabel("Mean Signal", fontsize=9)
    yticks, ylabels = xticks_2p()
    ax[2].set_ylabel("Phase (rad.)", fontsize=9)
    ax[2].set(ylim=(-2*np.pi/6, 13*np.pi/6), yticks=yticks,
              yticklabels=ylabels)
    for i in [0, 1, 2]:
        ax[i].tick_params(labelsize=8, direction='in')
    # title
    # ax[0].set(title=title)
    # text
    props = dict(boxstyle='round', facecolor='white', edgecolor='white',
                 alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'right'}
    texts = ["(a)", "(b)", "(c)"]
    for i in [0, 1, 2]:
        ax[i].text(0.98, 0.92, texts[i], bbox=props, **align,
                   transform=ax[i].transAxes, fontsize=9)
    # finish
    fig.tight_layout()
    return fig, ax


def amp_figs():
    """Plot of modulation, amplitude, and phase for DIL +2, -14, -30 GHz"""
    # read in fit data
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # DIL + 2 GHz
    mask = (fits['DL-Pro'] == 365872.6) & (fits['Attn'] == 44)
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    title = "DIL + 2 GHz"
    fig, ax = amp_mean_phase_plot(fits, mask, excluded, title)
    fig.tight_layout()
    fig.savefig("DILP2.pdf")
    fig.savefig(os.path.join("..", "DILP2.pdf"))
    # DIL - 14 GHz
    mask = (fits['DL-Pro'] == 365856.7)
    excluded = ["2016-09-23\\5_delay.txt", "2016-09-23\\11_delay.txt",
                "2016-09-23\\12_delay.txt", "2016-09-23\\16_delay.txt",
                "2016-09-23\\17_delay.txt", "2016-09-26\\8_delay.txt",
                "2016-09-26\\9_delay.txt"]
    title = "DIL - 14 GHz"
    fig, ax = amp_mean_phase_plot(fits, mask, excluded, title)
    fig.tight_layout()
    fig.savefig("DILM14.pdf")
    fig.savefig(os.path.join("..", "DILM14.pdf"))
    # DIL - 30 GHz
    mask = (fits['DL-Pro'] == 365840.7)
    excluded = ["2016-09-27\\7_delay.txt", "2016-09-27\\15_delay.txt"]
    title = "DIL - 30 GHz"
    fig, ax = amp_mean_phase_plot(fits, mask, excluded, title)
    fig.tight_layout()
    fig.savefig("DILM30.pdf")
    fig.savefig(os.path.join("..", "DILM30.pdf"))
    return


# main
# test_pam_plot()
amp_figs()
