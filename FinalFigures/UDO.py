# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:09:21 2018

@author: labuser
"""

# Uphill and Downhill orbit starts for +y angular momentum
# For Static Field Recombination paper.

import os
import numpy as np
import matplotlib.pyplot as plt


def ellipse(a, b):
    """Define the points on the ellipse."""
    c = np.sqrt(a**2 - b**2)
    x = np.arange(-a, a, 2*a/1e6)
    y = b/a*np.sqrt(a**2 - np.power(x, 2))
    return x, y, c


def udo_plot():
    """Plot an uphill and downhill orbit, with static field direction."""
    # get ellipse
    a = 10
    b = 5
    x, y, c = ellipse(a, b)
    # plotting
    fig, ax = plt.subplots()
    ax.axis('off')
    xlims = (-a*0.4, a*0.4)
    ylims = (-b, b)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # axis arrows
    arrowprops = {'width': 1, 'headwidth': 10, 'headlength': 10, 'color': 'k'}
    ax.annotate("", xy=(0, ylims[1]), xytext=(0, ylims[0]),
                arrowprops=arrowprops)
    ax.annotate("", xy=(xlims[1], 0), xytext=(xlims[0], 0),
                arrowprops=arrowprops)
    ax.text(xlims[1]*1.05, 0, "z", fontsize=14,
            horizontalalignment='left', verticalalignment='center')
    ax.text(0, ylims[1]*1.05, "x", fontsize=14,
            horizontalalignment='center', verticalalignment='bottom')
    # ellipses
    xd = x - c
    ax.plot(xd, y, '-.', c='C0')
    xu = x + c
    plt.plot(xu, -y, '--', c='C1')
    # starting arrows
    arrowprops = {'width': 0.8, 'headwidth': 7, 'headlength': 7, 'color': 'C0'}
    ax.annotate("", xy=(a - c, ylims[1]*0.3), xytext=(a - c, 0),
                arrowprops=arrowprops)
    ax.text(a - c, -ylims[1]*0.1, "(B)", fontsize=14,
            horizontalalignment='center', verticalalignment='center')
    arrowprops['color'] = 'C1'
    ax.annotate("", xy=(c-a, -ylims[1]*0.3), xytext=(c-a, 0),
                arrowprops=arrowprops)
    ax.text(c - a, ylims[1]*0.1, "(A)", fontsize=14,
            horizontalalignment='center', verticalalignment='center')
    # field arrow
    arrowprops = {'width': 1, 'headwidth': 10, 'headlength': 10, 'color': 'k'}
    ax.annotate("", xy=(xlims[1]*0.3, ylims[1]*0.7),
                xytext=(xlims[1]*0.8, ylims[1]*0.7),
                arrowprops=arrowprops)
    ax.text(xlims[1]*0.55, ylims[1]*0.72, r"$\vec{E}_S$", fontsize=14,
            horizontalalignment='center', verticalalignment='bottom')
    # core
    ax.plot(0, 0, '.', markersize=20, c='k')
    # finalize
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig("UDO.pdf")
    fig.savefig(os.path.join("..", "UDO.pdf"))
    return


# main script
udo_plot()
