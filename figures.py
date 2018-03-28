# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 07:46:44 2018

@author: labuser
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


# ==========
# Modulation and Mean vs. Pulsed Field
# ModvsField.pdf
# ==========
def phase_amp_mean_plot(data, title, ph_th, ax1, ax2):
    """Standard plotting for computed or experimental data.
    data DataFrame must have "Ep", "x0", "a", and "y0" keys."""
    # plot data
    ax1.axhline(0, color='grey')
    data.plot(x="Ep", y="a", ax=ax1, style="-o")
    data.plot(x="Ep", y="y0", ax=ax2, style="-o")
    # beautify
    # ax1.tick_params(which="minor", left="off")
    ax1.set(ylabel="Amp (pk-pk)", title=title)
    ax2.set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    # turn on grids
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.legend()
        ax.legend().remove()
    return ax1, ax2


def fsort_prep(fsort, excluded, title, ph_th, figname, ax1, ax2):
    fsort.sort_values(by=["Static"], inplace=True)
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # translate
    data = pd.DataFrame()
    data["Ep"] = fsort["Static"]
    data["a"] = fsort["a"]
    data["x0"] = fsort["phi"]
    data["y0"] = fsort["y0"]
    # phase threshold
    if ph_th is not None:
        # ph_th = 6*np.pi/6
        # Amplitude
        mask = (data["x0"] >= (ph_th - np.pi)) & (data["x0"] < ph_th)
        data.loc[mask, "a"] = -data[mask]["a"]
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "a"] = -data[mask]["a"]
        # phase
        mask = (data["x0"] < (ph_th - np.pi))
        data.loc[mask, "x0"] = data["x0"] + 2*np.pi
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "x0"] = data["x0"] - 2*np.pi
    # plot
    ax1, ax2 = phase_amp_mean_plot(data, title, ph_th, ax1, ax2)
    return data, ax1, ax2


def field_modulation():
    # read in all fits
    fname = os.path.join("..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # figure
    fig = plt.figure(figsize=(6, 9))
    gso = gridspec.GridSpec(3, 1)
    gsi0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gso[0])
    gsi1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gso[1])
    gsi2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gso[2])
    ax = np.array([None]*6)  # axes array for inner GridSpec
    # Add subplots from ax array with appropriate shared axes and labels.
    ax[0] = fig.add_subplot(gsi0[0])
    ax[1] = fig.add_subplot(gsi0[1], sharex=ax[0])
    ax[2] = fig.add_subplot(gsi1[0])
    ax[3] = fig.add_subplot(gsi1[1], sharex=ax[2])
    ax[4] = fig.add_subplot(gsi2[0])
    ax[5] = fig.add_subplot(gsi2[1], sharex=ax[4])
    # fig, ax = plt.subplots(nrows=6, figsize=(6, 9))
    # DIL + 2 GHz
    mask = (fits["DL-Pro"] == 365872.6) & (fits["Attn"] == 44)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    title = r"$W_0$ = DIL + 2 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_p2.pdf"
    data, ax[0], ax[1] = fsort_prep(fsort, excluded, title, ph_th, figname,
                                    ax[0], ax[1])
    ax[0].set(ylim=(-0.1, 0.1))
    ax[1].set(ylim=(0, 0.4))
    # DIL - 14 GHz
    mask = (fits["DL-Pro"] == 365856.7)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-09-23\\5_delay.txt", "2016-09-23\\11_delay.txt",
                "2016-09-23\\12_delay.txt", "2016-09-23\\16_delay.txt",
                "2016-09-23\\17_delay.txt", "2016-09-26\\8_delay.txt",
                "2016-09-26\\9_delay.txt"]
    title = r"$W_0$ = DIL - 14 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m14.pdf"
    data, ax[2], ax[3] = fsort_prep(fsort, excluded, title, ph_th, figname,
                                    ax[2], ax[3])
    ax[2].set(ylim=(-0.07, 0.07))
    ax[3].set(ylim=(0, 0.6))
    # DIL - 30 GHz
    mask = (fits["DL-Pro"] == 365840.7)
    fsort = fits[mask].sort_values(by=["Static"])
    excluded = ["2016-09-27\\7_delay.txt", "2016-09-27\\15_delay.txt"]
    title = r"$W_0$ = DIL - 30 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m30.pdf"
    data, ax[4], ax[5] = fsort_prep(fsort, excluded, title, ph_th, figname,
                                    ax[4], ax[5])
    ax[4].set(ylim=(-0.05, 0.05))
    ax[5].set(ylim=(0, 0.7))
    # letter labels
    props = dict(boxstyle='round', facecolor="white", alpha=1.0)
    ax[0].text(0.95, 0.95, "(a)", transform=ax[0].transAxes, fontsize=14,
               verticalalignment="center", horizontalalignment="center",
               bbox=props)
    ax[2].text(0.95, 0.95, "(b)", transform=ax[2].transAxes, fontsize=14,
               verticalalignment="center", horizontalalignment="center",
               bbox=props)
    ax[4].text(0.95, 0.95, "(c)", transform=ax[4].transAxes, fontsize=14,
               verticalalignment="center", horizontalalignment="center",
               bbox=props)
    # clean up
    gso.tight_layout(fig)
    plt.savefig('ModvsField.pdf')
    return
# ==========




field_modulation()
