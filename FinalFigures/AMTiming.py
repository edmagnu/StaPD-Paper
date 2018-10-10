# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:44:00 2018

@author: labuser
"""

# AM laser and MW timing figure.
# For Static Field Recombination paper

import os
import numpy as np
import matplotlib.pyplot as plt


def AMTiming():
    """Figure showing the period of the MW field and the timing of the AM
    modulated laser delay relative to the MW field."""
    # constants
    phi0 = -7*np.pi/6
    a = 0.9
    offmw = 2.0
    offir = 0.0
    freq = 15.932*1e9  # MW freq in Hz
    # omega = freq*(2*np.pi)
    # vectors
    xlims = (-np.pi/6, (5+1/6)*np.pi)
    phases = np.linspace(xlims[0], xlims[1], 10001)
    # times = phases/omega
    mw = a*np.sin(phases) + offmw
    ir = a*np.cos(phases + phi0) + offir
    # plot
    fig, ax = plt.subplots(figsize=(3.375, 0.8*3.375))
    ax.plot(phases, mw, c='C3', lw=3)
    ax.fill_between(phases, ir, -a + offir, color='C0')
    # markers
    largs = dict(c='k', lw=1)  # marker line arguments
    props = dict(boxstyle='round', color='white', alpha=0)
    align = {'verticalalignment': 'bottom',
             'horizontalalignment': 'center'}
    fs = 9  # text font size
    # zero references
    arrowprops = {'width': 1, 'headwidth': 10, 'headlength': 10,
                  'color': 'dimgray'}
    ax.annotate("", xy=(xlims[1]*1.1, -a + offir),
                xytext=(xlims[0]*1.1, -a + offir),
                arrowprops=arrowprops)
    ax.annotate("", xy=(xlims[1]*1.1, offmw),
                xytext=(xlims[0]*1.1, offmw),
                arrowprops=arrowprops)
    # phase offset
    # mw t=0 marker
    mw0 = 0
    ax.plot([mw0]*2, offmw + np.array([0, 1.4*a]), **largs)
    # IR t=0 marker
    ax.plot([-phi0]*2, [a + offir, offmw + 1.4*a], **largs)
    # horizontal span
    ax.plot([mw0, -phi0], [offmw + 1.2*a]*2, **largs)
    # label
    ax.text((mw0 - phi0)/2, offmw + 1.2*a, r"$\omega t_0$",
            **align, bbox=props, fontsize=fs)
    # MW Period
    # left peak marker
    ax.plot([2.5*np.pi]*2, offmw + np.array([a, 1.4*a]), **largs)
    # right peak marker
    ax.plot([4.5*np.pi]*2, offmw + np.array([a, 1.4*a]), **largs)
    # horizontal span
    ax.plot([2.5*np.pi, 4.5*np.pi], [offmw + 1.2*a]*2, **largs)
    # label
    ax.text(3.5*np.pi, offmw + 1.2*a, str(np.round(1/freq*1e12, 1)) + " ps",
            **align, bbox=props, fontsize=fs)
    # y labels
    align['verticalalignment'] = 'center'
    ax.text(-2.8, offmw, "MW Field", rotation=90, **align, fontsize=9)
    ax.text(-2.8, offir, "IR Intensity", rotation=90, **align, fontsize=9)
    # axes
    xticklabels = []
    xticks = np.arange(0, 5.5, 0.5)*np.pi
    for i in range(0, 13, 1):
        if (i % 2) == 0:
            label = str(int(i/2)) + r"$\pi$"
        else:
            label = ""
        xticklabels = xticklabels + [label]
    ax.set_xlabel("MW Phase (rad.)", fontsize=9)
    ax.set(xticks=xticks, xticklabels=xticklabels,
           yticks=[], yticklabels=[])
    ax.tick_params(labelsize=8, direction='in')
    ax.set(xlim=(xlims[0] - 0.1*np.diff(xlims)[0],
                 xlims[1] + 0.1*np.diff(xlims)[0]))
    ax.tick_params(labelsize=fs-2)
    ax.tick_params(axis='x', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # finishing
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig('AMTiming.pdf')
    fig.savefig(os.path.join("..", "AMTiming.pdf"))
    return phases, mw


AMTiming()