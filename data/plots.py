#!/usr/bin/env python 

import os
import os.path as op
import sys

import numpy as np
import pylab as pl 
from matplotlib.patches import Ellipse

from croaks import NTuple

def figure1():
    
    d = NTuple.fromtxt('selected_sn.list')
    
    # nearby sample 
    idx  = d['SURVEY'] == 'CSP'
    idx |= d['SURVEY'] == 'CalanTololo'
    idx |= d['SURVEY'] == 'CfAI'
    idx |= d['SURVEY'] == 'CfAII'
    idx |= d['SURVEY'] == 'CfAIII'
    idx |= d['SURVEY'] == 'lowz'
    d_nearby = d[idx]

    # SDSS 
    idx  = d['SURVEY'] == 'SDSS'
    d_sdss = d[idx]

    # SNLS 
    idx  = d['SURVEY'] == 'SNLS'
    d_snls = d[idx]
    
    # plot samples
    fig = pl.figure()    
    pl.errorbar(d_nearby['x1'], d_nearby['c'], 
                xerr=d_nearby['x1e'], yerr=d_nearby['ce'], 
                ls='', color='b', marker='o', alpha=0.25)
    pl.errorbar(d_sdss['x1'], d_sdss['c'], 
                xerr=d_sdss['x1e'], yerr=d_sdss['ce'], 
                ls='', color='g', marker='^', alpha=0.25)
    pl.errorbar(d_snls['x1'], d_snls['c'], 
                xerr=d_snls['x1e'], yerr=d_snls['ce'], 
                ls='', marker='s', color='orange',
                markerfacecolor=None, markeredgecolor='orange', alpha=0.25, )

    pl.xlabel('$X1$ [SALT2]')
    pl.ylabel('$Color$ [SALT2]')
    
    pl.axvline(-3, ls='--')
    pl.axvline(+3, ls='--')
    pl.axhline(0.3, ls='--')
    pl.axhline(-0.3, ls='--')
    
    pl.plot(-2., 0.2, marker='o', color='red', markersize=16)
    pl.annotate('faintest SN', (-2., 0.27), xytext=(-2,0.35),
                arrowprops={'facecolor': 'r', 'shrink': 0.05})
    pl.plot(+2., -0.2, marker='o', color='blue', markersize=16)    
    pl.annotate('brightest SN', (+2., -0.27), xytext=(2,-0.35),
                arrowprops={'facecolor': 'b', 'shrink': 0.05})

    e = Ellipse(xy=(0.,0.), width=5.6, height=0.56, fill=False, color='red', linestyle='dashdot')
    ax = pl.gca()
    ax.add_artist(e)


    fig.savefig('sn_parameter_space.pdf', bbox_inches='tight')
    fig.savefig('sn_parameter_space.png', bbox_inches='tight')
