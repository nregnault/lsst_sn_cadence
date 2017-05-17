#!/usr/bin/env python 

import os
import os.path 
import sys

import numpy as np
import pylab as pl

from saunerie import psf, snsim
from croaks import NTuple


def update_instrument_summary_data_from_obslog(instrument, obslog):
    data = instrument.data.copy()
    #    bands = np.unique(obslog.band)
    bands = np.unique(data['band'])
    for d in data:
        idx = obslog.band == d['band']
        d['iq'] = np.median(obslog.seeing[idx])
        d['mag_sky'] = np.median(obslog.mag_sky[idx])
        d['flux_sky'] = instrument.mag_to_flux(d['mag_sky'], [d['band']]) * instrument.pixel_area
    return data

def load_instruments():
    # load instruments
    lsst = psf.ImagingInstrument('LSST')
    lsst.precompute(full=1)
    lsst_pg = psf.ImagingInstrument('LSSTPG')
    lsst_pg.precompute(full=1)    
    
    lsst_pg_opsim = psf.ImagingInstrument('LSSTPG')
    log = snsim.OpSimObsLog(NTuple.fromtxt('Observations_DD_290_LSSTPG.txt'))
    d = update_instrument_summary_data_from_obslog(lsst_pg_opsim, log)
    lsst_pg_opsim.data[:] = d
    lsst_pg_opsim.precompute(full=1)
    
    # dump summary tables
    print "***** LSST [~LSE-40] ******"
    lsst.dump(exptime=30.)
    print "***** LSST [~SMTN-002 ] ******"
    lsst_pg.dump(exptime=30.)
    print "***** LSST [~SMTN-002 + OpSim default values ] ******"
    lsst_pg_opsim.dump(exptime=30.)
    
    return lsst, lsst_pg, lsst_pg_opsim

def plot_transmissions(lsst, lsst_pg):
    # plot transmissions
    fig = pl.figure()
    lsst.plot()
    pl.xlim((3000., 11000.))
    pl.grid(1)
    fig.savefig('lse_40_passbands.png', bbox_inches='tight')
    
    fig = pl.figure()
    lsst_pg.plot()
    pl.xlim((3000., 11000.))
    pl.grid(1)
    fig.savefig('smtn002_passbands.png', bbox_inches='tight')

    
def summary_plot(lsst, lsst_pg, lsst_pg_opsim=None, legend=True):
    
    # summary plot (instrument ZP's & m5)
    fig = pl.figure()
    ax_zp = ax = pl.subplot(221)
    pl.plot(lsst.data['leff'],   lsst.data['zp'], 'bs--', label='LSE-40')
    pl.plot(lsst_pg.data['leff'], lsst_pg.data['zp'], 'ro-', label='SMTN-002')
    if lsst_pg_opsim is not None:
        pl.plot(lsst_pg_opsim.data['leff'], lsst_pg_opsim.data['zp'], 'g^:', label='SMTN-002 [OpSim]')
    ax.get_xaxis().set_ticks(lsst.data['leff'])
    ax.get_xaxis().set_ticklabels(['$%s$' % tr.Band for tr in lsst.data['tr']], fontsize=14)
    ax.set_xlim((3000., 10250.))
    pl.ylim((25.5, 29.))
    pl.ylabel('ZP [AB, flux in e/s]', fontsize=14)
    pl.grid(1)
    if legend:
        pl.legend(loc='lower center', fontsize=12)

    # summary plot IQ 
    ax_iq = ax = pl.subplot(222)
    pl.plot(lsst.data['leff'],   lsst.data['iq'], 'bs--')
    pl.plot(lsst_pg.data['leff'], lsst_pg.data['iq'], 'ro-')
    if lsst_pg_opsim is not None:
        pl.plot(lsst_pg_opsim.data['leff'], lsst_pg_opsim.data['iq'], 'g^:')
    ax.get_xaxis().set_ticks(lsst.data['leff'])
    ax.get_xaxis().set_ticklabels(['$%s$' % tr.Band for tr in lsst.data['tr']], fontsize=14)
    ax.set_xlim((3000., 10250.))
    pl.ylim((0.6, 1.2))
    pl.ylabel('seeing [FWHM, arcsec]', fontsize=14)
    pl.grid(1)

    # summary plot mag sky
    ax = pl.subplot(223, sharex=ax_zp)
    pl.plot(lsst_pg.data['leff'], lsst_pg.data['mag_sky'], 'ro-')
    pl.plot(lsst.data['leff'],   lsst.data['mag_sky'], 'bs--')
    if lsst_pg_opsim is not None:
        pl.plot(lsst_pg_opsim.data['leff'],   lsst_pg_opsim.data['mag_sky'], 'g^:')
    ax.get_xaxis().set_ticks(lsst.data['leff'])
    ax.get_xaxis().set_ticklabels(['$%s$' % tr.Band for tr in lsst.data['tr']], fontsize=14)
    ax.set_xlim((3000., 10250.))
    pl.ylim((17.5, 23.25))
    pl.ylabel('$m_{\mathrm{sky}}$ [AB, mag/arcsec$^2$]', fontsize=14)
    pl.grid(1)    
    
    # summary plot: limiting mag
    ax = pl.subplot(224, sharex=ax_iq)
    d = lsst.data
    exptime = [30.] * len(d)
    skyflux = lsst.mag_to_flux(d['mag_sky'], d['band'])
    m5 = [float(lsst.mag_lim(t,m,s,b,snr=5.)) \
              for t,m,s,b in zip(exptime, skyflux, d['iq'], d['band'])]
    pl.plot(d['leff'],   m5, 'bs--')
    
    d = lsst_pg.data
    exptime = [30.] * len(d)
    skyflux = lsst_pg.mag_to_flux(d['mag_sky'], d['band'])
    m5 = [float(lsst_pg.mag_lim(t,m,s,b,snr=5.)) \
              for t,m,s,b in zip(exptime, skyflux, d['iq'], d['band'])]    
    pl.plot(d['leff'], m5, 'ro-')

    if lsst_pg_opsim is not None:
        d = lsst_pg_opsim.data
        exptime = [30.] * len(d)
        skyflux = lsst_pg_opsim.mag_to_flux(d['mag_sky'], d['band'])
        m5 = [float(lsst_pg_opsim.mag_lim(t,m,s,b,snr=5.)) \
                  for t,m,s,b in zip(exptime, skyflux, d['iq'], d['band'])]    
        pl.plot(d['leff'], m5, 'g^:')
        

    ax.get_xaxis().set_ticks(lsst.data['leff'])
    ax.get_xaxis().set_ticklabels(['$%s$' % tr.Band for tr in lsst.data['tr']], fontsize=14)
    ax.set_xlim((3000., 10250.))
    pl.ylabel('$5\sigma$-mag [30-s visit]', fontsize=14)
    pl.grid(1)    
    
    pl.subplots_adjust(left=0.1, right=0.95, wspace=0.3, top=0.95)

    fig.savefig('lsst_model_summary.png', bbox_inches='tight')
    fig.savefig('lsst_model_summary.pdf', bbox_inches='tight')
