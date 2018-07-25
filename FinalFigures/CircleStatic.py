# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:27:14 2018

@author: labuser
"""

# Amplitude vs Field Angle plot.
# For Static Field Recombination paper

import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd


def excluded_files(data):
    """Files not to be used in the analysis."""
    flist = ['Circle Static\\2015-11-28\\7_delay_botm200mV_35dB.txt']
    #          'Circle Static\\2015-11-29\\17_delay_hp200mV_vm020mV.txt']
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
    """Uncertainty in zero crossing"""
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
    """Amplitude vs Field Angle from fits of delay scans while changing the
    field angle"""
    # import
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "Circle Static",
                         "fits.txt")
    fits = pd.read_csv(fname, index_col=0)
    # pk-pk amplitude
    fits['a'] = 2*fits['a']
    # degrees instead of radians
    fits['fa'] = fits['fa']*360/(2*np.pi)
    # get rid of bad files
    fits = excluded_files(fits)
    gate = {1: 2*np.pi, 2: 1*np.pi, 3: 0.5*np.pi}
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
    plt.savefig('CircleStatic.pdf')
    return fits


# main
circle_static()
