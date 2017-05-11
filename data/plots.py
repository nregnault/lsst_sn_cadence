#!/usr/bin/env python 

import os
import os.path as op
import sys

import numpy as np
import pylab as pl 

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
    
    pl.plot(-3., 0.3, marker='o', color='red', markersize=16)
    pl.annotate('fiducial SN', (-2.98, 0.3), xytext=(-2,0.35),
                arrowprops={'facecolor': 'r', 'shrink': 0.01})
    
    fig.savefig('sn_parameter_space.pdf', bbox_inches='tight')
    fig.savefig('sn_parameter_space.png', bbox_inches='tight')
