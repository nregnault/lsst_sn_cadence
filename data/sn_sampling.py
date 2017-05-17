#!/usr/bin/env python 

import os
import os.path 
import sys

import numpy as np
import pylab as pl

from saunerie import psf
import os
import os.path 
import sys

import numpy as np
import pylab as pl

from croaks import NTuple
from saunerie import psf, snsim, salt2
from pycosmo import cosmo 


_instruments = {}
model_components = None


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


def get_instrument(instrument_name, log=None):
    global _instruments
    if instrument_name not in _instruments:
        instr = psf.ImagingInstrument(instrument_name)
        instr.precompute(full=1)
        _instruments[instrument_name] = instr
    return _instruments[instrument_name]


def init_lcmodel(log, filename='salt2.npz'):
    global model_components
    if model_components is None:
        if filename is None:
            model_components = salt2.ModelComponents.load_from_saltpath()
        else:
            model_components = salt2.ModelComponents(filename)
    fs = salt2.load_filters(np.unique(log.band))
    lcmodel = snsim.SALT2LcModel(fs,model_components)
    return lcmodel

def bands(instrument, excl=[]):
    bands = instrument.data['band']
    cut = np.in1d(bands, excl)
    return bands[~cut]

def update_log_with_seeing_and_sky(log, seeing={}, mag_sky={}):
    bands = np.unique(log.band)
    for b in bands:
        idx = log.band == b
        if b in seeing:
            log.data['seeing'][idx] = seeing[b]
            print seeing[b], ' !! '
        if b in mag_sky:
            log.data['sky'][idx] = mag_sky[b]
            print mag_sky[b], ' !! '
    
    
def gen_simple_obslog(instrument, opsim_log=None, cadence=1., tmin=-20, tmax=60., 
                      excl_bands=['LSSTPG::u'], 
                      texp={'LSSTPG::g': 300., 'LSSTPG::r': 600., 'LSSTPG::i': 600., 'LSSTPG::z': 780., 'LSSTPG::y': 600.}):
    """
    """
    
    def select(data, excl):
        bands = data['band']
        cut = np.in1d(bands, excl)
        return data[~cut]
    
    # 
    if opsim_log is None:
        data = select(instrument.data, excl=excl_bands)
    else:
        data = update_instrument_summary_data_from_obslog(instrument, opsim_log)
        data = select(data, excl=excl_bands)
    seq = np.arange(tmin, tmax+cadence, cadence)
    nbands = len(data)
    n = len(seq)
    
    # 
    band = np.repeat(data['band'], n)
    mjd = np.tile(seq, nbands)
    exptime = np.repeat([texp[b] for b in data['band']], n)    
    seeing = np.repeat(data['iq'], n)
    moon_frac = np.zeros(len(band))
    mag_sky = np.repeat(data['mag_sky'], n)
    flux_sky = instrument.mag_to_flux(mag_sky, band)
    kAtm = np.zeros(len(band))
    airmass = np.ones(len(band))
    m5sigmadepth = instrument.mag_lim(exptime, flux_sky, seeing, band, snr=5.0)
    Nexp = np.ones(len(band))
    
    data = np.rec.array((band, mjd, exptime, seeing, moon_frac, mag_sky, kAtm, airmass, m5sigmadepth, Nexp), 
                        names = ['band', 'mjd', 'exptime', 'seeing', 'moon_frac', 'sky', 'kAtm', 'airmass', 'm5sigmadepth', 'Nexp'])
    return snsim.OpSimObsLog(data)


def generate_sample(obslog):
    """
    generate a sample, with 1 sn per redshift bin
    """
    cosmo_model = cosmo.CosmoLambda()
    norm = 1.E-9 / (4. * np.pi)
    
    z = np.arange(0.05, 1.2, 0.05)
    nsn_tot = len(z)
    mjd_start, mjd_end = obslog.mjd.min(), obslog.mjd.max()
    dl = cosmo_model.dL(z)
    X1 = np.zeros_like(z)
    Color = np.zeros_like(z)
    #    DayMax = np.random.uniform(mjd_start, mjd_end, size=nsn_tot)
    DayMax = np.zeros_like(z)
    ra = np.zeros_like(z)
    dec = np.zeros_like(z)
    sample = np.rec.fromarrays((z, dl, DayMax, X1, Color, ra, dec), 
                               names=['z', 'dL', 'DayMax', 'X1', 'Color', 'ra', 'dec'])
    return sample
    

def reso(lcfit, plot=1):
    r = []
    for lc,W in lcfit:
        C = np.linalg.inv(W)
        r.append(np.sqrt(np.diag(C)))
    r = np.rec.fromrecords(r, names=['eX0', 'eX1', 'eColor', 'eDayMax'])
    return r


def reso_vs_lc_amplitude():
    opsim = NTuple.fromtxt('Observations_DD_290_LSSTPG.txt')
    idx = opsim['band'] != 'LSSTPG::u'
    log = snsim.OpSimObsLog(opsim[idx])
    r = log.split()
    lcmodel = init_lcmodel(log)
    
    s = snsim.SnSurveyMC(obslog=r[2], filename='lsst_survey.conf')
    sne = s.generate_sample()
    lc = s.generate_lightcurves(sne, lcmodel, fit=1)
    bands = np.unique(log.band)
    r = []
    for l in lc:
        C = l.covmat()
        sa = []
        for b in bands:
            sa.append(l.amplitude_snr(b))
        r.append(np.sqrt(np.diag(C)).tolist() + sa)
    return sne, np.rec.fromrecords(r, names=['eX0', 'eX1', 'eColor', 'eDayMax'] + ['sig_' + b for b in bands])


        

        

def main_opsim_log(filename='Observations_DD_290_LSSTPG.txt', i_season=0,  
                   X1=0., Color=0., label=None, tofile=None, zlim=None):
    
    d = NTuple.fromtxt(filename)
    cut = (d['band'] != 'LSSTPG::u')
    log = snsim.OpSimObsLog(d[cut])
    #    update_log_with_seeing_and_sky(log, 
    #                                   seeing = {'LSSTPG::u':  0.92, 'LSSTPG::g': 0.87,  'LSSTPG::r':  0.83,  'LSSTPG::i':  0.80, 'LSSTPG::z':  0.78, 'LSSTPG::y': 0.76},
    #                                   mag_sky= {'LSSTPG::u': 22.95, 'LSSTPG::g': 22.24, 'LSSTPG::r': 21.20 , 'LSSTPG::i': 20.47, 'LSSTPG::z': 19.60, 'LSSTPG::y': 18.63})
    #    update_log_with_seeing_and_sky(log, 
    #                                   mag_sky= {'LSSTPG::y': 18.63})
    
    seasons = log.split()
    log = seasons[i_season]
    lcmodel = init_lcmodel(log)
    
    instr = [get_instrument(nm) for nm in np.unique(log.instrument)]
    s = snsim.SnSurveyMC(obslog=log, filename='lsst_survey.conf', instruments=instr)
    
    sne = s.generate_sample()
    sne['X1'] = X1 # -3. 
    sne['Color'] = Color # 0.3 
    
    lc = s.generate_lightcurves(sne, lcmodel, fit=True)
    res = reso(lc)
    
    fig = pl.figure()
    #    pl.plot(sne['z'], res['eColor'], marker='.', color='gray', ls='')
    p_first = (s.obslog.mjd.min()-sne['DayMax']) / (1. + sne['z'])
    p_last  = (s.obslog.mjd.max()-sne['DayMax']) / (1. + sne['z'])
    idx = (p_first<-15.) & (p_last>20.)
    print len(idx), idx.sum()
    pl.plot(sne['z'][idx], res['eColor'][idx], marker='.', color='b', ls='')
    
    pl.xlabel('$z$', fontsize=18)
    pl.ylabel('$\sigma{\cal{C}}$', fontsize=18)
    pl.ylim((0., 0.15))
    pl.axhline(0.03, color='r', ls='--')
    if label:
        pl.annotate(label, xy=(0.1, 0.8), xycoords='axes fraction', 
                    fontsize=18)
    pl.grid(1)
    if zlim:
        pl.axvline(zlim, ls='-.', color='gray', lw=2)
    
    if tofile:
        fig.savefig(tofile, bbox_inches='tight')
    
    pl.title('season=%d' % i_season)
        
    return s, sne, res, lc


def all_opsim_telecon():
    s, sne, res, lc = main_opsim_log(i_season=0,
                                     X1=-3, Color=0.3, zlim=0.65, label='[X1=-3, Color=0.3]', 
                                     tofile='color_resolution_season_0_sublumi.png')
    print ' (*) survey duration [days, yr]', s.duration
    s, sne, res, lc = main_opsim_log(i_season=0,
                                     X1=0., Color=0., zlim=0.8, label='[X1=0, Color=0]', 
                                     tofile='color_resolution_season_0_normal.png')
    print ' (*) survey duration [days, yr]', s.duration
    s, sne, res, lc = main_opsim_log(i_season=2,
                                     X1=-3, Color=0.3, zlim=0.6, label='[X1=-3, Color=0.3]', 
                                     tofile='color_resolution_season_2_sublumi.png')
    print ' (*) survey duration [days, yr]', s.duration
    s, sne, res, lc = main_opsim_log(i_season=2,
                                     X1=0., Color=0., zlim=0.62, label='[X1=0, Color=0]', 
                                     tofile='color_resolution_season_2_normal.png')
    print ' (*) survey duration [days, yr]', s.duration



def main(instrument_name='LSSTPG', cadence=1., opsim_log=None):
    """
    """
    instr = get_instrument(instrument_name)
    log = gen_simple_obslog(instr, cadence=cadence, opsim_log=opsim_log)
    lcmodel = init_lcmodel(log)
    
    s = snsim.SnSurveyMC(obslog=log, filename='lsst_survey.conf', instruments=instr)
    
    sne = s.generate_sample()
    sne['X1'] = 0.
    sne['Color'] = 0.
    sne['DayMax'] = 0.
    print len(sne)
    
    lc = s.generate_lightcurves(sne, lcmodel, fit=True)
    res = reso(lc)
    
    pl.plot(sne['z'], res['eColor'], 'r.')
    
    return s, sne, res


def plot_log_median_conditions(log, instrument=None, bands=["LSSTPG::" + b for b in "ugrizy"]):
    # seeing 
    fig = pl.figure()
    iplot = 1
    for band in bands:
        try:
            pl.subplot(3, 2, iplot)
            idx = log.data['band'] == band
            pl.hist(log.data['seeing'][idx], histtype='stepfilled', alpha=0.5, color='b', normed=True)
            med = np.median(log.data['seeing'][idx])
            pl.axvline(med, color='r', ls='-')
            idx = instrument.data['band'] == band
            pl.axvline(instrument.data[idx]['iq'], color='r', ls='--')
            iplot += 1
            pl.annotate(band.split('::')[-1], 
                        xy=(0.85, 0.8), xycoords='axes fraction', 
                        fontsize=16)
            pl.xlim((0.5, 1.5))
            pl.yticks([])
        except:
            continue
    fig.suptitle('seeing', fontsize=18)
    
    fig = pl.figure()
    iplot = 1
    for band in bands:
        pl.subplot(3, 2, iplot)
        idx = log.data['band'] == band
        d = log.data['sky'][idx]
        rng = np.floor(d.min()), np.ceil(d.max())
        pl.hist(d, histtype='stepfilled', range=rng, bins=10, alpha=0.5, color='b', normed=True)
        med = np.median(log.data['sky'][idx])
        pl.axvline(med, color='r', ls='-')
        idx = instrument.data['band'] == band
        reference_value = instrument.data[idx]['mag_sky']
        pl.axvline(reference_value, color='r', ls='--')
        iplot += 1
        pl.annotate(band.split('::')[-1], 
                    xy=(0.85, 0.8), xycoords='axes fraction', 
                    fontsize=16)
        pl.yticks([])
    fig.suptitle('mag sky', fontsize=18)



# def main(instrument_name, even_sample=False, cadence=1.):
    
#     instr = get_instrument(instrument_name)
#     log = gen_simple_obslog(instr, cadence=cadence)
#     lcmodel = init_lcmodel(log)
#     s = snsim.SnSurveyMC(obslog=log, 
#                          # filename='lsst_survey.conf', 
#                          INSTRUMENTS={instr.instrument.name: instr})
#     s.zrange = (0.01, 1.2)
        
#     if even_sample:
#         sample = generate_sample(log)
#     else:
#         sample = s.generate_sample()
#         sample['DayMax'] = 0.
#         sample['X1'] = 0.
#         sample['Color'] = 0.
        
    
#     lc = s.generate_lightcurves(sample, lcmodel, fit=True)
#     r = reso(lc)
    
#     #    pl.figure()
#     pl.plot(sample['z'], r['eColor'], 'r.')
    

#     return s, sample, lc

    
