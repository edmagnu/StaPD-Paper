# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:00:56 2018

@author: labuser
"""

# Coulomb, Static, and MW field diagram
# For Static Field Recombination Paper

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def CoulMW():
    """Produce a plot of the potential seen by an electron in a static field,
    along with an inset of the MW field as a function of time.
    Returns DataFrames potential, dlimit
    """
    # Atomic units
    fAU1mVcm = 1.94469e-13
    enAU1GHz = 1.51983e-7
    # set up figure
    fig, ax = plt.subplots(figsize=(3.375, 3.375))
    ax2 = fig.add_axes([0.7, 0.45, 0.15, 0.4])
    zmax = 1000000
    xlims = (-zmax, zmax)
    ylims = (-100, 100)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ylims = (-100, 100)
    ax.set_ylim(ylims)
    # Potential DataFrame
    f = 40*fAU1mVcm
    dz = zmax/10000
    potential = pd.DataFrame({"z": np.arange(-zmax, 0, dz)})
    potential = potential.append(
            pd.DataFrame({'z': np.arange(dz, zmax + dz, dz)}))
    potential['C'] = -1/(np.abs(potential["z"]))/enAU1GHz
    potential['E'] = -f*potential['z']/enAU1GHz
    potential['V'] = potential['C'] + potential['E']
    # plot
    potential.plot(x="z", y="V", linewidth=2,
                   label=r"$-1/r - E_S \cdot z$", c='k', ax=ax)
    # beautify
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("z", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    # Excitation and direction arrows
    ax.plot(0, 0, color='k', marker='o')
    arrowprops = {'width': 2, 'headwidth': 10, 'headlength': 10, 'color': 'C3'}
    ax.annotate(s="", xy=(0, -10), xytext=(0, ylims[0]),
                arrowprops=arrowprops)
    arrowprops['color'] = 'C0'
    ax.annotate(s="", xy=(xlims[0]*0.3, 0), xytext=(xlims[0]*0.05, 0),
                arrowprops=arrowprops)
    arrowprops['color'] = 'C1'
    ax.annotate(s="", xy=(xlims[1]*0.3, 0), xytext=(xlims[1]*0.05, 0),
                arrowprops=arrowprops)
    # coordinate lines
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    arrowprops['color'] = 'k'
    ax.annotate(s="", xy=(xlims[1], ylims[0]), xytext=(xlims[0], ylims[0]),
                arrowprops=arrowprops)
    ax.annotate(s="", xy=(xlims[0], ylims[1]), xytext=(xlims[0], ylims[0]),
                arrowprops=arrowprops)
    # MW Field
    ax2 = MW_field(ax2)
    # finalize figure
    ax.legend().remove()
    # ax.axis('off')
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.savefig("CoulMW.pdf")
    plt.savefig(os.path.join("..", "CoulMW.pdf"))
    return potential


def MW_field(ax):
    """Inset MW field figure for CoulMW.pdf. Vertical time, horizontal MW field
    value."""
    # figure
    # fig, ax = plt.subplots()
    xlims = (-1.2, 1.2)
    ylims = (-np.pi/3, (2+2/3)*np.pi)
    # build data
    df = pd.DataFrame()
    df['t'] = np.arange(-np.pi/6, 2*np.pi + np.pi/6, np.pi/180)
    df['f'] = -np.sin(df['t'])
    # plot
    df.plot(x='f', y='t', c='C5', ax=ax)
    # manual axis lines
    arrowprops = {'width': 1, 'headwidth': 10, 'headlength': 10, 'color': 'k'}
    ax.annotate(s="", xy=(0, ylims[1]), xytext=(0, ylims[0]),
                arrowprops=arrowprops)
    ax.text(0, ylims[1] + np.pi/6, "t", fontsize=14,
            verticalalignment='bottom', horizontalalignment='center')
    ax.annotate(s="", xy=(xlims[0], 0), xytext=(xlims[1], 0),
                arrowprops=arrowprops)
    ax.text(xlims[1]+0.1, 0, r"$\vec{E}_{MW}$", fontsize=14,
            verticalalignment='center', horizontalalignment='left')
    # Es line
    ylims = (ylims[0]*3, ylims[1])
    ax.annotate(s="", xy=(xlims[0], ylims[0]), xytext=(xlims[1], ylims[0]),
                arrowprops=arrowprops)
    ax.text(xlims[1]+0.1, ylims[0], r"$\vec{E}_S$", fontsize=14,
            verticalalignment='center', horizontalalignment='left')
    # beautify
    ax.set(xlim=xlims, ylim=ylims)
    ax.legend().remove()
    ax.axis('off')
    # fig.tight_layout(rect=[0, 0, 0.9, 0.9])
    return ax


# main:
CoulMW()
# fig, ax = plt.subplots()
# MW_field(ax)
