# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 07:46:44 2018

@author: labuser
"""

import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def atomic_units():
    """Return a dictionary of atomic units, ['GHz'], ['mVcm'], and ['ns']"""
    au = {'GHz': 1.51983e-7, 'mVcm': 1.94469e-13, 'ns': 4.13414e7}
    return au


def progress(source, i, total):
    """print an updating report of 'source: i/total'"""
    # start on fresh line
    if i == 0:
        print()
    # progress
    print("\r{0}: {1} / {2}".format(source, i+1, total), end="\r")
    # newline if we've reached the end.
    if i+1 == total:
        print()
    return


# ==========
# Modulation and Mean vs. Pulsed Field
# ModvsField.pdf
# ==========
def phase_amp_mean_plot(data, title, ph_th, ax1, ax2):
    """Standard plotting for computed or experimental data.
    data DataFrame must have 'Ep', 'x0', 'a', and 'y0' keys."""
    # plot data
    ax1.axhline(0, color='grey')
    data.plot(x='Ep', y='a', ax=ax1, style='-o')
    data.plot(x='Ep', y='y0', ax=ax2, style='-o')
    # beautify
    # ax1.tick_params(which='minor', left='off')
    ax1.set(ylabel="Amp (pk-pk)", title=title)
    ax2.set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    # turn on grids
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.legend()
        ax.legend().remove()
    return ax1, ax2


def fsort_prep(fsort, excluded, title, ph_th, figname, ax1, ax2):
    fsort.sort_values(by=['Static'], inplace=True)
    # unmassage amps and phases
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
    # translate
    data = pd.DataFrame()
    data['Ep'] = fsort['Static']
    data['a'] = fsort['a']
    data['x0'] = fsort['phi']
    data['y0'] = fsort['y0']
    # phase threshold
    if ph_th is not None:
        # ph_th = 6*np.pi/6
        # Amplitude
        mask = (data['x0'] >= (ph_th - np.pi)) & (data['x0'] < ph_th)
        data.loc[mask, 'a'] = -data[mask]['a']
        mask = (data['x0'] >= (ph_th + np.pi))
        data.loc[mask, 'a'] = -data[mask]['a']
        # phase
        mask = (data['x0'] < (ph_th - np.pi))
        data.loc[mask, 'x0'] = data['x0'] + 2*np.pi
        mask = (data['x0'] >= (ph_th + np.pi))
        data.loc[mask, 'x0'] = data['x0'] - 2*np.pi
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
    mask = (fits['DL-Pro'] == 365872.6) & (fits['Attn'] == 44)
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
    mask = (fits['DL-Pro'] == 365856.7)
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
    mask = (fits['DL-Pro'] == 365840.7)
    fsort = fits[mask].sort_values(by=['Static'])
    excluded = ["2016-09-27\\7_delay.txt", "2016-09-27\\15_delay.txt"]
    title = r"$W_0$ = DIL - 30 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m30.pdf"
    data, ax[4], ax[5] = fsort_prep(fsort, excluded, title, ph_th, figname,
                                    ax[4], ax[5])
    ax[4].set(ylim=(-0.05, 0.05))
    ax[5].set(ylim=(0, 0.7))
    # letter labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    ax[0].text(0.95, 0.95, "(a)", transform=ax[0].transAxes, fontsize=14,
               verticalalignment='center', horizontalalignment='center',
               bbox=props)
    ax[2].text(0.95, 0.95, "(b)", transform=ax[2].transAxes, fontsize=14,
               verticalalignment='center', horizontalalignment='center',
               bbox=props)
    ax[4].text(0.95, 0.95, "(c)", transform=ax[4].transAxes, fontsize=14,
               verticalalignment='center', horizontalalignment='center',
               bbox=props)
    # clean up
    gso.tight_layout(fig)
    plt.savefig('ModvsField.pdf')
    return fits
# ==========


# ==========
# 1-D model for turning time
# up_and_down_orbits.pdf
# ==========
def interp_match(df1, df2, kx, ky):
    x = np.intersect1d(df1[kx], df2[kx])
    y1 = np.interp(x, df1[kx], df1[ky])
    y2 = np.interp(x, df2[kx], df2[ky])
    return x, y1, y2


def turning_time_figure():
    au = atomic_units()
    # import data
    dname = os.path.join("..", "2D-Comp-Model", "computation", "Turning Time")
    fname = os.path.join(dname, "data_raw.txt")
    data_tot = pd.read_csv(fname, index_col=0)
    fname = os.path.join(dname, "picked_w.txt")
    picked_tot = pd.read_csv(fname, index_col=0)
    # convert to lab units
    picked_tot['W'] = picked_tot['W']/au['GHz']
    picked_tot['field'] = picked_tot['field']/au['mVcm']
    # plot
    fig, axes = plt.subplots(nrows=2, figsize=(4, 6), sharex=True)
    ymin, ymax = -100, 100
    xmin, xmax = 0, 100
    # ==========
    # uphill figure
    # ==========
    ax = axes[0]
    colors = ['C2', 'C3', 'C9']
    # mask uphill values (Dir == -1)
    picked = picked_tot[picked_tot['Dir'] == -1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # turning time = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10.sort_values(by='field', inplace=True)
    dftt10.plot(x='field', y='W', label='tturn=10', color='k', lw=3, ax=ax)
    # upper binding time = 20 ns
    mask = (picked['kind'] == 'tplus=20')
    mask = mask & mask_NaN
    dftplus = picked[mask].copy(deep=True)
    dftplus.sort_values(by='field', inplace=True)
    dftplus.plot(x='field', y='W', label='tplus=20', color='k', lw=3, ax=ax)
    # lower binding time = 20 ns
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftbind = picked[mask].copy(deep=True)
    obs = dftbind.iloc[-1].copy(deep=True)
    obs['W'] = 0
    obs['field'] = 0
    dftbind = dftbind.append(obs, ignore_index=True).copy(deep=True)
    dftbind.sort_values(by='field', inplace=True)
    dftbind.plot(x='field', y='W', label='tb=20', color='k', lw=3, ax=ax)
    # fill between
    # If turning time < 10 ns, return is < 20 ns, survival is chaotic
    ax.fill_between(dftt10['field'], dftt10['W'], ymin, color=colors[2])
    # If between tbind and tplus, it's in a safe "long orbit"
    x, y1, y2 = interp_match(dftbind, dftplus, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    mask = (dftplus['field'] >= max(x))
    ax.fill_between(dftplus[mask]['field'], ymax, dftplus[mask]['W'],
                    color=colors[0])
    mask = (dftbind['field'] <= min(x))
    x, y1, y2 = interp_match(dftbind[mask], dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    # If above tbind or between tt10, it's lost
    ax.fill_between(dftbind['field'], ymax, dftbind['W'], color=colors[1])
    x, y1, y2 = interp_match(dftplus, dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[1])
    mask = (dftt10['field'] >= max(x))
    ax.fill_between(dftt10[mask]['field'], ymax, dftt10[mask]['W'],
                    color=colors[1])
    # text labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'center'}
    ax.text(80, 95, "(a)", **align, bbox=props)  # chaotic
    ax.text(50, 95, "(b)", **align, bbox=props)  # Immediate Ionization
    ax.text(32, 95, "(c)", **align, bbox=props)  # Goldylocks Zone
    ax.text(10, 95, "(d)", **align, bbox=props)  # Late Return & Ionization
    # touch ups
    ax.set(xlabel="Pulsed Field (mV/cm)", xlim=(xmin, xmax),
           ylabel=r"$W_0 + \Delta W_{MW}(\phi)$ (GHz)", ylim=(ymin, ymax),
           title="Uphill Electrons")
    ax.legend().remove()
    # ==========
    # downhill figure
    # ==========
    ax = axes[1]
    colors = ['C2', 'C3', 'C9']
    # mask uphill values (Dir == 1)
    picked = picked_tot[picked_tot['Dir'] == 1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # lower binding time = 20 ns
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftbind = picked[mask].copy(deep=True)
    obs = dftbind.iloc[-1].copy(deep=True)
    obs['W'] = 0
    obs['field'] = 0
    dftbind = dftbind.append(obs, ignore_index=True)
    dftbind.sort_values(by='field', inplace=True)
    dftbind.plot(x='field', y='W', label='tb=20', color='k', lw=3, ax=ax)
    # turning time = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10.sort_values(by='field', inplace=True)
    obs = dftbind.iloc[-4]
    dftt10 = dftt10.append(obs, ignore_index=True)
    dftt10.plot(x='field', y='W', label='tturn=10', color='k', lw=3, ax=ax)
    # DIL W = -2 E^0.5
    dfdil = pd.DataFrame({'field': np.arange(xmin, xmax+1, 1)})
    dfdil['W'] = -2*np.sqrt(dfdil['field']*au['mVcm'])/au['GHz']
    mask = (dfdil['field'] >= max(dftt10['field']))
    dfdil = dfdil[mask]
    dfdil.plot(x='field', y='W', label='DIL', color='k', lw=3, ax=ax)
    # fill between
    # Above binding time = 20 ns, Immediately Ionizes
    ax.fill_between(dftbind['field'], ymax, dftbind['W'], color=colors[1])
    ax.fill_between(dfdil['field'], ymax, dfdil['W'], color=colors[1])
    # Below turning time = 10 ns, Chaotic
    ax.fill_between(dftt10['field'], dftt10['W'], ymin, color=colors[2])
    mask = (dftbind['field'] >= max(dftt10['field']))
    ax.fill_between(dftbind[mask]['field'], dftbind[mask]['W'], ymin,
                    color=colors[2])
    ax.fill_between(dfdil['field'], dfdil['W'], ymin, color=colors[2])
    # Between binding time = 20ns and turnign time = 10 ns, Goldylocks
    x, y1, y2 = interp_match(dftbind, dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    # text labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'center',
             'horizontalalignment': 'left'}
    ax.text(2.5, -50, "(a)", **align, bbox=props)  # chaotic
    # ax.text(0, 50, "(b)", **align, bbox=props)  # Late Return and Ionize
    ax.text(10, 0, "(c)", **align, bbox=props)  # Goldylocks Zone
    ax.text(2.5, 50, "(d)", **align, bbox=props)  # Immediate Ionization
    # touch ups
    ax.set(xlabel="Pulsed Field (mV/cm)", xlim=(xmin, xmax),
           ylabel=r"$W_0 + \Delta W_{MW}(\phi)$ (GHz)", ylim=(ymin, ymax),
           title="Downhill Electrons")
    ax.legend().remove()
    # save
    fig.tight_layout()
    plt.savefig('up_and_down_orbits.pdf')
    return data_tot, picked_tot
# ==========


# ==========
# 2D model selections
# w0_2D.pdf
# ==========
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


def w0_2D():
    # au = atomic_units()
    # import
    dname = os.path.join("..", "2D-Comp-Model", "computation")
    fname = os.path.join(dname, "data_fit.txt")
    data = pd.read_csv(fname, index_col=0)
    fname = os.path.join(dname, "params_sums.txt")
    params = pd.read_csv(fname, index_col=0)
    # get needed values for W0 = 0 GHz, Epulse = 0, 36, 100 mV/cm
    w0 = np.sort(params['E0'].unique())[1]
    fields = np.sort(params['Ep'].unique()[[0, 5, 10]])
    # plot
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(4, 6))
    xmin, xmax = 0, 2*np.pi
    ymin, ymax = -0.05, 0.32
    xticks, xticklabels = xticks_2p()
    colors = ['C0', 'C1', 'C2']
    # ==========
    # 0 mV/cm
    # ==========
    ax = axes[0]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[0])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels,  # xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 0$ mV/cm")
    # ==========
    # 36 mV/cm
    # ==========
    ax = axes[1]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[1])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels,  # xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 36$ mV/cm")
    # ==========
    # 100 mV/cm
    # ==========
    ax = axes[2]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[2])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels, xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 100$ mV/cm")
    # ==========
    # save
    # ==========
    # fig.legend(bbox_to_anchor=(0.97, 1), framealpha=1)
    # fig.suptitle(r"$W_0 = 0$ GHz")
    # fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.tight_layout()
    plt.savefig('w0_2D.pdf')
    return data, params


def w20_2D():
    # au = atomic_units()
    # import
    dname = os.path.join("..", "2D-Comp-Model", "computation")
    fname = os.path.join(dname, "data_fit.txt")
    data = pd.read_csv(fname, index_col=0)
    fname = os.path.join(dname, "params_sums.txt")
    params = pd.read_csv(fname, index_col=0)
    # get needed values for W0 = 0 GHz, Epulse = 0, 7.2, 100 mV/cm
    w0 = np.sort(params['E0'].unique())[0]
    fields = np.sort(params['Ep'].unique()[[0, 1, 10]])
    # plot
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(4, 6))
    xmin, xmax = 0, 2*np.pi
    ymin, ymax = -0.05, 0.4
    xticks, xticklabels = xticks_2p()
    colors = ['C0', 'C1', 'C2']
    # ==========
    # 0 mV/cm
    # ==========
    ax = axes[0]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[0])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels,  # xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 0$ mV/cm")
    # ==========
    # 36 mV/cm
    # ==========
    ax = axes[1]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[1])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels,  # xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 7.2$ mV/cm")
    # ==========
    # 100 mV/cm
    # ==========
    ax = axes[2]
    ax.axhline(0, color='k')
    # masking
    mask = (params['E0'] == w0)  # Energy
    mask = mask & (params['Ep'] == fields[2])  # field
    mask = mask & np.isnan(params["dL"])  # dL combined
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label="Signal", lw=3, c=colors[2])
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Uphill", lw=3, ls='dashed', c=colors[0])
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label="Downhill", lw=3, ls='dashed', c=colors[1])
    # clean up
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=xticks,
           xticklabels=xticklabels, xlabel=r"Phase $\phi_0$ (rad)",
           ylabel="Norm. $e^-$ Signal", title=r"$E_{pulse} = 100$ mV/cm")
    # ==========
    # save
    # ==========
    # fig.legend(bbox_to_anchor=(0.97, 1), framealpha=1)
    # fig.suptitle(r"$W_0 = 0$ GHz")
    # fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.tight_layout()
    plt.savefig('w20_2D.pdf')
    return data, params
# ==========


# ==========
# Phase Delay scan with and without pulsed field of 14 mV/cm
# phase_delay.pdf
# ==========
def model_func_fit(x, y0, a, phi):
    """Sinusoidal plus offset model for delay scan phase dependence.
    "x" is the delay in wavelengths
    "y" is the normalized Rydberg signal.
    Returns model dataframe and fit parameters.
    """
    return y0 + a*np.sin(2*np.pi*x + phi)


def running_mean(df, n):
    cumsum = np.array(df['nsignal'].cumsum(skipna=True))
    y = (cumsum[n:] - cumsum[:-n]) / n
    cumsum = np.array(df['wavelengths'].cumsum(skipna=True))
    x = (cumsum[n:] - cumsum[:-n]) / n
    return x, y


def phase_delay():
    # load
    dname = os.path.join("..", "Data", "StaPD-Analysis")
    fname = os.path.join(dname, "moddata.txt")
    data_tot = pd.read_csv(fname, sep="\t", index_col=0)
    # pick out DIL + 2 GHz
    mask_p2 = (data_tot["DL-Pro"] == 365872.6)
    mask_p2 = mask_p2 & (data_tot["Attn"] == 44)
    # excluded data
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    for fname in excluded:
        mask_p2 = mask_p2 & (data_tot["Filename"] != fname)
    # plot
    fig, ax = plt.subplots()
    nave = 3
    # pick out field = 0 mV/cm
    mask = mask_p2 & (data_tot['Static'] ==
                      np.sort(data_tot['Static'].unique())[21])
    data = data_tot[mask].copy(deep=True)
    data.sort_values(by='wavelengths', inplace=True)
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='grey')
    x = np.array(data['wavelengths'])
    popt = np.array(data[['y0', 'a', 'phi']].iloc[0])
    y = model_func_fit(x, *popt)
    ax.plot(x, y, '-', c='k')
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # pick out field = -14 mV/cm
    # mask = mask_p2 & (data_tot['Static'] ==
    #                   np.sort(data_tot['Static'].unique())[17])
    # data = data_tot[mask].copy(deep=True)
    # data.sort_values(by='wavelengths', inplace=True)
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # pick out field = -14 mV/cm
    # mask = mask_p2 & (data_tot['Static'] ==
    #                   np.sort(data_tot['Static'].unique())[25])
    # data = data_tot[mask].copy(deep=True)
    # data.sort_values(by='wavelengths', inplace=True)
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # pick out field = 36 mV/cm
    mask = mask_p2 & (data_tot['Static'] ==
                      np.sort(data_tot['Static'].unique())[29])
    data = data_tot[mask].copy(deep=True)
    data.sort_values(by='wavelengths', inplace=True)
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='grey', markersize=3)
    x = np.array(data['wavelengths'])
    popt = np.array(data[['y0', 'a', 'phi']].iloc[0])
    y = model_func_fit(x, *popt)
    ax.plot(x, y, '-', c='blue')
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # pick out field = 108 mV/cm
    mask = mask_p2 & (data_tot['Static'] ==
                      np.sort(data_tot['Static'].unique())[-7])
    data = data_tot[mask].copy(deep=True)
    data.sort_values(by='wavelengths', inplace=True)
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='grey')
    x = np.array(data['wavelengths'])
    popt = np.array(data[['y0', 'a', 'phi']].iloc[0])
    y = model_func_fit(x, *popt)
    ax.plot(x, y, '-', c='k')
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # pick out field = 7.2 mV/cm
    mask = mask_p2 & (data_tot['Static'] ==
                      np.sort(data_tot['Static'].unique())[23])
    data = data_tot[mask].copy(deep=True)
    data.sort_values(by='wavelengths', inplace=True)
    x, y = running_mean(data, nave)
    ax.plot(x, y, '-', c='grey')
    x = np.array(data['wavelengths'])
    popt = np.array(data[['y0', 'a', 'phi']].iloc[0])
    y = model_func_fit(x, *popt)
    ax.plot(x, y, '-', c='k')
    # data.plot(x='wavelengths', y='nsignal', marker='.', ls='None', ax=ax)
    # tidy
    # ax.set(xlim=(-0.2, 3.5))
    ax.set(xlabel=r"Delay $\phi_0$ (rad)", ylabel="Norm. Signal",
           xticks=np.arange(0, 3.5, 0.5),
           xticklabels=["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$",
                        r"$5\pi$", r"$6\pi$"],
           ylim=(-0.03, 0.37))
    # twin axes
    ax2 = ax.twiny()
    ax2.tick_params('x')
    tps = 1/(15.932*1e9)*1e12
    tticks = np.arange(0, 3.5, 0.5)
    ax2.set(xticks=tticks, xticklabels=np.round(tticks*tps, 1),
            xlabel=r"Delay $t_0$ (ps)")
    # ax.legend().remove()
    # labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'center',
             'horizontalalignment': 'right'}
    ax2.text(3.2, 0.3, "0.0 mV/cm", **align, bbox=props)
    ax2.text(3.2, 0.18, "7.2 mV/cm", **align, bbox=props)
    ax2.text(3.2, 0.06, "36.0 mV/cm", **align, bbox=props)
    ax2.text(3.2, 0, "108.0 mV/cm", **align, bbox=props)
    # save
    fig.tight_layout(rect=(0, 0, 0.98, 1))
    plt.savefig("phase_delay.pdf")
    return data_tot, mask_p2, data
# ==========


# ==========
# Field vs IR Intensity
# fields.pdf
# ==========
def field_fig():
    # constants
    phi0 = 4*np.pi/6
    a = 0.9
    offmw = 2.0
    offir = 0.0
    freq = 15.932*1e9  # MW freq in Hz
    # omega = freq*(2*np.pi)
    # vectors
    phases = np.linspace(0, 5*np.pi, 10001)
    # times = phases/omega
    mw = a*np.sin(phases) + offmw
    ir = a*np.sin(phases + phi0) + offir
    # plot
    fig, ax = plt.subplots()
    ax.plot(phases, mw, c='C3', lw=3)
    ax.fill_between(phases, ir, -a + offir, color='C0')
    # markers
    largs = dict(c='k', lw=1)  # marker line arguments
    props = dict(boxstyle='round', color='white', alpha=0)
    align = {'verticalalignment': 'center',
             'horizontalalignment': 'center'}
    fs = 14  # text font size
    # zero references
    ax.axhline(-a + offir, color='grey', lw=1)
    ax.axhline(offmw, color='grey', lw=1)
    # phase offset
    ax.plot([np.pi]*2, offmw + np.array([0, 0.5]), **largs)
    ax.plot([np.pi + 3/2*np.pi - phi0]*2, [a + offir, offmw + 0.5], **largs)
    ax.plot([np.pi, np.pi + 3/2*np.pi - phi0], [offmw + 0.25]*2, **largs)
    ax.text(np.pi + (3/2*np.pi - phi0)/2, offmw + 0.5, r"$\omega t_0$",
            **align, bbox=props, fontsize=fs)
    # MW Period
    ax.plot([2.5*np.pi]*2, offmw + a + np.array([0, 0.3]), **largs)
    ax.plot([4.5*np.pi]*2, offmw + a + np.array([0, 0.3]), **largs)
    ax.plot([2.5*np.pi, 4.5*np.pi], [offmw + a + 0.15]*2, **largs)
    ax.text(3.5*np.pi, offmw + a + 0.3, str(np.round(1/freq*1e12, 1)) + " ps",
            **align, bbox=props, fontsize=fs)
    # y labels
    ax.text(-1.5, offmw, "MW Field", rotation=90, **align, fontsize=fs)
    ax.text(-1.5, offir, "IR Intensity", rotation=90, **align, fontsize=fs)
    # axes
    xticklabels = []
    xticks = np.arange(0, 5.5, 0.5)*np.pi
    for i in range(0, 13, 1):
        if (i % 2) == 0:
            label = str(int(i/2)) + r"$\pi$"
        else:
            label = ""
        xticklabels = xticklabels + [label]
    ax.set_xlabel("MW Phase $\omega t_0$ (rad.)", fontsize=fs)
    ax.set(xticks=xticks, xticklabels=xticklabels,
           yticks=[], yticklabels=[])
    ax.tick_params(labelsize=fs-2)
    ax.tick_params(axis='x', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # finishing
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig('fields.pdf')
    return phases, mw
# ==========


# ==========
# Sideways Pulsed Field
# circle_static.pdf
# ==========
def excluded_files(data):
    flist = ['Circle Static\\2015-11-28\\7_delay_botm200mV_35dB.txt',
             'Circle Static\\2015-11-29\\17_delay_hp200mV_vm020mV.txt']
    for fname in flist:
        mask = data['Filename'] != fname
        data = data[mask].copy(deep=True)
    return data


def massage_amp_phi(fsort, gate):
    """Given a series of fit phases, fix the phases and amplitudes so that all
    phases fall between [0, pi] + gate.
    Returns a fits DataFrame with modified 'phi'
    """
    # force all amps to be positive
    mask = (fsort['a'] < 0)
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + np.pi
    # force phases between 0 and 2pi
    mask = (fsort['phi'] > 2*np.pi)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] - 2*np.pi
    mask = (fsort['phi'] < 0)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + 2*np.pi
    # phases above gate
    mask = (fsort['phi'] > gate) & (fsort['phi'] <= 2*np.pi)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] - np.pi
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    # phases below gate - pi
    mask = (fsort['phi'] < (gate - np.pi)) & (fsort['phi'] > 0)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + np.pi
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    return fsort


def xi_and_sigma(line, x):
    sumx = sum(x**2)
    n = len(x)
    sigma = line['stderr']
    sigma_inter = sigma*np.sqrt(sumx/n)
    intercept = line['intercept']
    slope = line['slope']
    xi = -intercept/slope
    sigma_xi = np.sqrt(xi**2*((sigma/slope)**2 +
                              (sigma_inter/intercept)**2))
    return xi, sigma_xi


def circle_static():
    # import
    fname = os.path.join("..", "Data", "StaPD-Analysis", "Circle Static",
                         "fits.txt")
    fits = pd.read_csv(fname, index_col=0)
    # pk-pk amplitude
    fits['a'] = 2*fits['a']
    # degrees instead of radians
    fits['fa'] = fits['fa']*360/(2*np.pi)
    # get rid of bad files
    fits = excluded_files(fits)
    gate = {1: np.pi, 2: 2*np.pi, 3: 1.5*np.pi}
    # linear regressions on each group
    for group in [1, 2, 3]:
        mask = fits['group'] == group
        fsort = fits[mask].copy()
        fsort = massage_amp_phi(fsort, gate[group])
        slope, intercept, rvalue, pvalue, stderr = linregress(
                fsort['fa'], fsort['a'])
        line = {'slope': slope, 'intercept': intercept, 'rvalue': rvalue,
                'pvalue': pvalue, 'stderr': stderr}
        fits.loc[mask, 'line'] = [line]*sum(mask)
        fits.loc[mask, ['fa', 'a']] = fsort[['fa', 'a']]
    # master linear regression
    slope, intercept, rvalue, pvalue, stderr = linregress(
            fits['fa'], fits['a'])
    line = {'slope': slope, 'intercept': intercept, 'rvalue': rvalue,
            'pvalue': pvalue, 'stderr': stderr}
    fits['line_m'] = [line]*len(fits)
    # build xi and sigma_xi
    for group in [1, 2, 3]:
        # line = lines[group]
        mask = fits['group'] == group
        fsort = fits[mask]
        line = fsort.iloc[0]['line']
        xi, sigma_xi = xi_and_sigma(line, fsort['fa'])
        fits.loc[mask, 'xi'] = xi
        fits.loc[mask, 'sigma_xi'] = sigma_xi
    # master xi_m and sigma_xi_m
    line = fits.iloc[0]['line_m']
    xi, sigma_xi = xi_and_sigma(line, fits['fa'])
    fits['xi_m'] = xi
    fits['sigma_xi_m'] = sigma_xi
    # plotting
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    markers = {1: 'o', 2: '^', 3: 's'}
    fig, ax = plt.subplots()
    # total linear regression
    line = fits.iloc[0]['line_m']
    label = "{0} +/- {1} degrees".format(np.round(xi, 2),
                                         np.round(sigma_xi, 2))
    print("Total :\t", label)
    label = "Linear Regression"
    ax.plot(fits['fa'], line['intercept'] + line['slope']*fits['fa'], '-',
            c='k', lw=2, label=label)
    for group in [1, 2, 3]:
        mask = fits['group'] == group
        fsort = fits[mask]
        line = fits.iloc[0]['line']
        # label = labels[group]
        # sumx = sum((fsort['fa'])**2)
        # n = len(fsort)
        # sigma = line[4]
        # sigma_inter = sigma*np.sqrt(sumx/n)
        # intercept = line[1]
        xi = fsort.iloc[0]['xi']
        sigma_xi = fsort.iloc[0]['sigma_xi']
        label = "{0} +/- {1} degrees".format(np.round(xi, 2),
                                             np.round(sigma_xi, 2))
        print(group, " :\t", label)
        label = "Set " + str(group)
        ax.plot(fsort['fa'], fsort['a'], linestyle='none', color=colors[group],
                marker=markers[group], label=label)
        # x = fsorts[group]['fa']
        # ax.plot(x, lines[group][1] + lines[group][0]*x, '-', c=colors[group])
    ax.legend(fontsize=10)
    ax.set_xlabel("Field Angle (deg.)", fontsize=14)
    ax.set_ylabel("Pk-Pk Amplitude", fontsize=14)
    ax.tick_params(labelsize=12, direction='in')
    # twin axes with mV/cm
    ax2 = ax.twiny()
    ax2.tick_params('x')
    locs = ax.get_xticks()
    ax2.set_xlim(ax.get_xlim())
    locs2 = 200*0.1*0.72*np.sin(locs*2*np.pi/360)  # component in z direction
    ax2.set_xticklabels(np.round(locs2, 1))
    ax2.tick_params(labelsize=12, direction='in')
    ax2.set_xlabel("Vertical Field (mV/cm)", fontsize=14)
    # tidy up
    fig.tight_layout()
    plt.savefig('circle_static.pdf')
    return fits
# ==========

# ==========
# main script
# fits = field_modulation()
# data, picked = turning_time_figure()
# data, params = w0_2D()
# data, params = w20_2D()
# data_tot, mask, data = phase_delay()
# phases, mw = field_fig()
fits = circle_static()
