# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 23:00:43 2018

@author: labuser
"""

# Model Results
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


def model_func(x, a, x0, y0):
    """Model for fitting cosine to convolved data"""
    return y0 + a*np.cos(x - x0)


def build_fitdata_from_params(obs):
    """Given an observation from params DataFrame, pull out a, x0, y0 and use
    model_func to the fit data.
    Returns xs, ys: np.array"""
    a = float(obs["a"])
    x0 = float(obs["x0"])
    y0 = float(obs["y0"])
    xs = np.arange(0, 2*np.pi, np.pi/100)
    ys = model_func(xs, a, x0, y0)
    return xs, ys


def Sim_plot(params, W0, Ep, ylims, title, ax):
    # settings
    xmin, xmax = 0, 2*np.pi
    (ymin, ymax) = ylims
    xticks, xticklabels = xticks_2p()
    # colors = ['C0', 'C1', 'C2']
    colors = ['C0', 'C1', 'C2']
    styles = ['--', ':', '-']
    # 0 line
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == W0)
    mask = mask & (params['Ep'] == Ep)  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    i = 2
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, styles[i], label="Signal", lw=2, c=colors[i])
    # plot 1/2 th_LRL = 0
    i = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, styles[i], label="Uphill", lw=3, c=colors[i])
    # plot 1/2 th_LRL = pi
    i = 1
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, styles[i], label="Downhill", lw=3, c=colors[i])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels,  # xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. Signal", title=title)
    return ax


def Sim0():
    """Model Results at W0 = 0 GHz"""
    # au = atomic_units()
    # import
    dname = os.path.join("..", '..', "2D-Comp-Model", "computation")
    fname = os.path.join(dname, "data_fit.txt")
    data = pd.read_csv(fname, index_col=0)
    fname = os.path.join(dname, "params_sums.txt")
    params = pd.read_csv(fname, index_col=0)
    # get needed values for W0 = 0 GHz, Epulse = 0, 36, 100 mV/cm
    W0 = np.sort(params['E0'].unique())[1]
    fields = np.sort(params['Ep'].unique()[[0, 5, 10]])
    # plot
    fig, axes = plt.subplots(nrows=3, sharex=True)
    ylims = (-0.05, 0.4)
    # 0 mV/cm
    i = 0
    title = r"$E_S = 0$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # 36 mV/cm
    i = 1
    title = r"$E_S = 36$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # 100 mV/cm
    i = 2
    title = r"$E_S = 100$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # save
    fig.tight_layout()
    fig.savefig("Sim0.pdf")
    fig.savefig(os.path.join("..", "Sim0.pdf"))
    return data, params


def SimM20():
    """Model Results at W0 = 0 GHz"""
    # au = atomic_units()
    # import
    dname = os.path.join("..", '..', "2D-Comp-Model", "computation")
    fname = os.path.join(dname, "data_fit.txt")
    data = pd.read_csv(fname, index_col=0)
    fname = os.path.join(dname, "params_sums.txt")
    params = pd.read_csv(fname, index_col=0)
    # get needed values for W0 = 0 GHz, Epulse = 0, 36, 100 mV/cm
    W0 = np.sort(params['E0'].unique())[0]    
    fields = np.sort(params['Ep'].unique()[[0, 1, 10]])
    # plot
    fig, axes = plt.subplots(nrows=3, sharex=True)
    ylims = (-0.05, 0.4)
    # 0 mV/cm
    i = 0
    title = r"$E_S = 0$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # 36 mV/cm
    i = 1
    title = r"$E_S = 7.2$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # 100 mV/cm
    i = 2
    title = r"$E_S = 100$ mV/cm"
    Ep = fields[i]
    axes[i] = Sim_plot(params, W0, Ep, ylims, title, axes[i])
    # save
    fig.tight_layout()
    fig.savefig("SimM20.pdf")
    fig.savefig(os.path.join("..", "SimM20.pdf"))
    return data, params


# main script
Sim0()
SimM20()
