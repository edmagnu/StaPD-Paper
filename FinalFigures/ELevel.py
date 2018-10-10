# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:32:35 2018

@author: labuser
"""

# Energy Level Diagram
# For Static Field Recombination Paper

import os
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def ELevel():
    """Energy level figure showing 2s -> 2p -> 3d -> n/e f"""
    plt.close()
    fig, ax = plt.subplots(figsize=(3.375, 3.375))
    w = 0.75
    dx = 0.5
    dy = 1
    delta = 0.95
    # 2s
    i = 0
    ax.plot(i*dx + np.array([0, w]), np.array([i, i])*dy, c='k')
    ax.text(i*dx, i*dy, "2S", fontsize=14,
            verticalalignment='bottom', horizontalalignment='left')
    arrowprops = {'width': 1, 'headwidth': 10, 'headlength': 10, 'color': 'C3'}
    rs = (i*dx + w/2, i*dy)
    rf = ((i+1)*dx + w/2, (i+1)*dy)
    dr = ((rf[0] - rs[0])*delta, (rf[1] - rs[1])*delta)
    rsn = (rs[0] + dr[0], rs[1] + dr[1])
    rfn = (rf[0] - dr[0], rf[1] - dr[1])
    ax.annotate("", xy=rsn, xytext=rfn, arrowprops=arrowprops)
    ax.text(i*dx + w, (i+0.4)*dy, "670 nm", fontsize=14)
    # 2p
    i = 1
    ax.plot(i*dx + np.array([0, w]), np.array([i, i])*dy, c='k')
    ax.text(i*dx, i*dy, "2P", fontsize=14,
            verticalalignment='bottom', horizontalalignment='left')
    arrowprops['color'] = 'C1'
    rs = (i*dx + w/2, i*dy)
    rf = ((i+1)*dx + w/2, (i+1)*dy)
    dr = ((rf[0] - rs[0])*delta, (rf[1] - rs[1])*delta)
    rsn = (rs[0] + dr[0], rs[1] + dr[1])
    rfn = (rf[0] - dr[0], rf[1] - dr[1])
    ax.annotate("", xy=rsn, xytext=rfn, arrowprops=arrowprops)
    ax.text(i*dx + w, (i+0.4)*dy, "610 nm", fontsize=14)
    # 3d
    i = 2
    ax.plot(i*dx + np.array([0, w]), np.array([i, i])*dy, c='k')
    ax.text(i*dx, i*dy, "3D", fontsize=14,
            verticalalignment='bottom', horizontalalignment='left')
    arrowprops['color'] = 'C4'
    rs = (i*dx + w/2, i*dy)
    rf = ((i+1)*dx + w/2, (i+1)*dy)
    dr = ((rf[0] - rs[0])*delta, (rf[1] - rs[1])*delta)
    rsn = (rs[0] + dr[0], rs[1] + dr[1])
    rfn = (rf[0] - dr[0], rf[1] - dr[1])
    ax.annotate("", xy=rsn, xytext=rfn, arrowprops=arrowprops)
    ax.text(i*dx + w, (i+0.4)*dy, "819 nm", fontsize=14)
    # n/e f
    i = 3
    ax.plot(i*dx + np.array([0, w]), ((i + np.array([0, 0]))*dy),
            c='k')
    ax.text(i*dx, i*dy, r"nf, $\epsilon$f", fontsize=14,
            verticalalignment='bottom', horizontalalignment='left')
    # beautify
    ax.axis('off')
    fig.tight_layout()
    fig.savefig("ELevel.pdf")
    fig.savefig(os.path.join("..", "Elevel.pdf"))
    return


# main
ELevel()
