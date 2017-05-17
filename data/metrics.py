#!/usr/bin/env python 

"""

"""


import os
import os.path as op
import re
import glob

import numpy as np
import pylab as pl

from croaks import NTuple
from pycosmo.cosmo import CosmoLambda
from saunerie import salt2, snsim, psf


model_components = None
cosmo = CosmoLambda()


class SimpleCadenceMetric():
    """
    
    """
    def __init__(self, mjd, dt=21):
        """
        Given a lightcurve shape, and a observation log, compute the
        relative variance of the light curve amplitude (in one band)
        in a sliding window. 
        
        Args:
          - log (OpSimObsLog-like): observation log 
          - shape (callable): something that gives the shape of the light curve
               in a given band
          - instrument (ImagingInstrument): used to convert the 5sigma mags 
               into 5 sigma fluxes.
        """
        self.mjd, self.dt = mjd, int(dt)
        self._grid = self._obs_grid(mjd)
        self.obs, self.bins = np.histogram(self.mjd, 
                                           bins=self._grid)
        ix = np.arange(0, self.dt, 1)
        self.ix = np.vstack(ix+i for i in xrange(len(self._grid)-self.dt))
        self.xx = np.arange(np.floor(mjd.min()), np.ceil(mjd.max()), 1.)
        self.wiLidi = self.obs[self.ix]
        self.sumw = np.sqrt(np.sum(self.wiLidi, axis=1)) / self.dt
        
    def _obs_grid(self, mjd):
        """
        return a grid, with 1-day wide cells, encompassing the
        duration of the run. Add margins to this grid, to taken into
        account the SN rise time at the beginning, and the SN decline
        time at the end of the survey.
        """
        rng = np.floor(mjd.min()), np.ceil(mjd.max())
        return np.arange(rng[0]-self.dt, rng[1], 1.)

    def __call__(self, mjd):
        return np.interp(mjd+0.5 * self.dt, self.xx, self.sumw)




class CadenceMetric():
    """
    
    """
    def __init__(self, mjd, f5, band, shape):
        """
        Given a lightcurve shape, and a observation log, compute the
        relative variance of the light curve amplitude (in one band)
        in a sliding window. 
        
        Args:
          - log (OpSimObsLog-like): observation log 
          - shape (callable): something that gives the shape of the light curve
               in a given band
          - instrument (ImagingInstrument): used to convert the 5sigma mags 
               into 5 sigma fluxes.
        """
        self.mjd, self.f5, self.band, self.shape = mjd, f5, band, shape
        self._grid = self._obs_grid(mjd, shape)
        #        self.index = np.digitize(self.mjd, self.grid)
        self.obs, self.bins = np.histogram(self.mjd, 
                                           bins=self._grid, 
                                           weights=1./self.f5**2)
        sz = len(shape)
        ix = np.arange(0, sz, 1)
        self.ix = np.vstack(ix+i for i in xrange(len(self._grid)-sz))
        #        self.mjd_offset = np.argmax(self.shape) - len(self.shape)
        self.xx = np.arange(np.floor(mjd.min()), np.ceil(mjd.max()), 1.)
        self.wiLidi = self.obs[self.ix] * self.shape**2
        self.sumw = 5. * np.sqrt(np.sum(self.wiLidi, axis=1))

    def _obs_grid(self, mjd, shape):
        """
        return a grid, with 1-day wide cells, encompassing the
        duration of the run. Add margins to this grid, to taken into
        account the SN rise time at the beginning, and the SN decline
        time at the end of the survey.
        """
        rng = np.floor(mjd.min()), np.ceil(mjd.max())
        # we assume the shape to rise, reach a max and decline
        imax, sz = np.argmax(shape), len(shape)
        return np.arange(rng[0]-imax, rng[1]+sz-imax, 1.)

    def mean_L(self):
        wi = self.obs[self.ix]
        return self.wiLidi.sum(axis=1) / wi.sum(axis=1)
    
    def __call__(self, mjd):
        return np.interp(mjd, self.xx, self.sumw)


def band_colors(instrument):
    data = instrument.data
    cols = [pl.cm.jet(int(256 * (wl-3000.) / (11000.-3000.))) for wl in data['leff']]
    return dict(zip(data['band'], cols))


def init_lcmodel(bands, filename='salt2.npz'):
    """
    Utility function to load a SALT2 light curve model. 
    The model components are cached. 

    This should be a function of the snsim module.
    """
    global model_components
    if model_components is None:
        print 'we have to reload model_components'
        if filename is None:
            model_components = salt2.ModelComponents.load_from_saltpath()
        else:
            model_components = salt2.ModelComponents(filename)
    fs = salt2.load_filters(np.unique(bands))
    lcmodel = snsim.SALT2LcModel(fs, model_components)
    return lcmodel


def get_pulse_shapes(bands, 
                     restframe_phase_range = (-20., 40.),
                     z=1.1, X1=0., Color=0., DayMax=0., 
                     filename='salt2.npz',
                     cosmo=cosmo, plot=False):
    """
    Call the SALT model for the fiducial SN specified in argument, on
    a grid of regularly spaced mjd (observer frame), corresponding to
    ``restframe phase range``
    
    .. note : slow because no 
    """
    pmin, pmax = restframe_phase_range
    mjd_min = np.floor(pmin * (1.+z) + DayMax)
    mjd_max = np.ceil(pmax * (1.+z) + DayMax)
    mjd = np.arange(mjd_min, mjd_max, 1.)
    
    sn = np.rec.fromrecords([(z, cosmo.dL(z), X1, Color, DayMax)], 
                            names=['z', 'dL', 'X1', 'Color', 'DayMax'])    
    
    ret = {}
    lcmodel = init_lcmodel(bands, filename=filename)
    for bn in bands:
        b = [bn] * len(mjd)
        #        lcmodel = init_lcmodel(b)
        ret[bn] = lcmodel(sn, mjd, b)
        
    return mjd, ret


def plot_pulse_shapes(mjd, shapes, colors=None, bands=None):
    pl.figure()
    fmax = None
    if bands is None:
        bands = shapes.keys()
        #    for bn, shape in shapes.items():
    for bn in bands:
        shape = shapes[bn]
        c = 'r'
        if colors is not None:
            c = colors[bn]
        pl.plot(mjd, shape, marker='.', ls=':', 
                color=c, 
                label=bn.split(':')[-1])
        if fmax is None or fmax<shape.max():
            fmax = shape.max()
    pl.xlabel('mjd', fontsize=16)
    pl.ylabel('flux [e$^-$/s]', fontsize=16)
    pl.legend(loc='best')
    pl.ylim((0., fmax*1.1))
    

def plot_weighted_cadence(z=0.8, band='LSSTPG::z'):
    mjd, shapes = mjd, shapes = get_pulse_shapes(['LSSTPG::' + b for b in "grizy"], z=z)
    cadence, weighted_cadence = [], []
    for delta in xrange(1,100):
        d = np.zeros_like(mjd)
        d[::delta] = 1.
        cadence.append(np.sum(d) / len(d))
        weighted_cadence.append(np.sum(d * shapes[band]**2) / np.sum(shapes[band]**2))
    cadence = np.asarray(cadence)
    weighted_cadence = np.asarray(weighted_cadence)
    return np.arange(1,100), cadence, weighted_cadence
#    pl.plot(cadence, weighted_cadence, 'r.')



def figure_1_pulse_shapes(X1=0., Color=0.):
    lsstpg = psf.ImagingInstrument('LSSTPG')
    colors = band_colors(lsstpg)

    bands = ['LSSTPG::' + b for b in "rizy"]
    mjd, shapes = get_pulse_shapes(bands=bands, z=0.7, X1=X1, Color=Color)
    plot_pulse_shapes(mjd, shapes, colors=colors, bands=bands)
    pl.title('[z=0.7, X1=%6.2f, Color=%6.2f]' % (X1, Color))
    pl.ylim((0., 65.))
    pl.gcf().savefig('pulse_shape_z_07.pdf', bbox_inches='tight')
    
    bands=['LSSTPG::' + b for b in "izy"]
    mjd, shapes = get_pulse_shapes(bands=bands, z=0.9, X1=X1, Color=Color)
    plot_pulse_shapes(mjd, shapes, colors=colors, bands=bands)
    pl.title('[z=0.9, X1=%6.2f, Color=%6.2f]' % (X1, Color))
    pl.ylim((0., 65.))
    pl.gcf().savefig('pulse_shape_z_09.pdf', bbox_inches='tight')

    bands=['LSSTPG::' + b for b in "izy"]    
    mjd, shapes = get_pulse_shapes(bands=bands, z=1.1, X1=X1, Color=Color)
    plot_pulse_shapes(mjd, shapes, colors=colors, bands=bands)
    pl.title('[z=1.1, X1=%6.2f, Color=%6.2f]' % (X1, Color))
    pl.ylim((0., 65.))
    pl.gcf().savefig('pulse_shape_z_11.pdf', bbox_inches='tight')    


def simple_cadence_metrics(eps, shape):
    n = len(shape)
    bins = np.arange(n+1.) - 0.5
    xx = 0.5 * (bins[1:] + bins[:-1])
    norm = np.max(shape)
    sh = shape / norm
    r = []
    for e in eps:
        x = np.arange(0., n, e)
        v,b = np.histogram(x, bins=bins)
        if v.sum() <= 0.01:
            print ' [+] ', e, v, x
        v = v * sh**2
        if v.sum() <= 0.01:
            print ' [-] ', e, v, x
        if e > 50.:
            print v
        r.append(v.sum()/sh.sum())
    return np.asarray(r)
    

def f5_cadence_specs(SNR=20):
    bands=['LSSTPG::' + b for b in "grizy"]
    
    # limit 
    print " LIMITS "
    for z in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]:
        mjd, shapes = get_pulse_shapes(z=z, X1=-3, Color=0.3, bands=bands)
        L = {}
        line = "  %6.1f  & " % z
        for b in bands:
            Li2 = np.sqrt(np.sum(shapes[b]**2))
            lim = 5. * Li2 / SNR
            line += " %6.0f & " % (lim,)
        line += '\\\\'
        print line 

    # sum_i L_i^2 
    print " SUM Li2 "
    for z in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]:
        mjd, shapes_worst_case = get_pulse_shapes(z=z, X1=-3, Color=0.3, bands=bands)
        mjd, shapes_average    = get_pulse_shapes(z=z, X1=0., Color=0.0, bands=bands)
        L = {}
        line = "  %6.1f  & " % z
        for b in bands:
            Li2_average    = np.sqrt(np.sum(shapes_average[b]**2))
            Li2_worst_case = np.sqrt(np.sum(shapes_worst_case[b]**2))
            line += " %6.0f / %6.0f & " % (Li2_average, Li2_worst_case)
        line += '\\\\'
        print line 


def f5_cadence_lims(SNR=20, 
                    zs=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                    X1=-3., Color=0.3, 
                    bands=["LSSTPG::" + b for b in "grizy"]):
    """
    like the functin above, but just returns a dict 
    with the cadence limits for the requested SNR
    """
    lims = {}
    for z in zs:
        mjd, shapes = get_pulse_shapes(z=z, X1=X1, Color=Color, bands=bands)
        lims[z] = {}
        for b in bands:
            Li2 = np.sqrt(np.sum(shapes[b]**2))
            lim = 5. * Li2 / SNR
            lims[z][b] = lim
    return lims


# 
# now implemented in log
# 
# def median_values_from_logs(logs, band):
#     for log in logs:
#         r = log.split()
#         for l in r:
#             idx = l.band == bands
#             seeing = np.median(l.seeing[idx])
#             mag_sky = np.median(l.mag_sky[idx])
#             flux_sky = instr.mag_to_flux([mag_sky], [band])
#             mjd = l.mjd[idx]
#             c,b = np.histogram(mjd, bins=bins)
#             cadence = 1./c.mean()
#             texp = np.median(l.exptime[idx])
#             m5 = lsstpg.mag_lim(texp, flux_sky, seeing, band)
#             r.append((band, seeing, mag_sky, flux_sky, cadence, texp, m5))
#     return np.fromrecords(r, names=['band', 'seeing', 'mag_sky', 'flux_sky', 'cadence', 'texp', 'm5'])

            
def f5_cadence_plot(lsstpg, band, lims=None, median_log_summary=None, 
                    mag_range=(23., 26.5), 
                    dt_range=(0.5, 15.), 
                    target={},
                    SNR=20):
    dt = np.linspace(dt_range[0], dt_range[1], 100)
    m5 = np.linspace(mag_range[0], mag_range[1], 100)
    b = [band] * len(m5)
    f5 = lsstpg.mag_to_flux(m5, b)
    
    F5,DT = np.meshgrid(f5, dt)
    M5,DT = np.meshgrid(m5, dt)
    metric = np.sqrt(DT) * F5
    
    # draw limits 
    pl.figure()
    pl.imshow(metric, extent=(mag_range[0], mag_range[1], 
                              dt_range[0], dt_range[1]), 
              aspect='auto', alpha=0.25)
    
    if lims is not None:
        fmt = {}
        ll = [lims[zz][band] for zz in lims.keys()]
        cs = pl.contour(M5, DT, metric, ll, colors='k')
        
        strs = ['$z=%3.1f$' % zz for zz in lims.keys()]
        for l,s in zip(cs.levels, strs):
            fmt[l] = s
        pl.clabel(cs, inline=1, fmt=fmt, fontsize=16)
    
    if median_log_summary is not None:
        idx = median_log_summary['band'] == band
        m = median_log_summary[idx]
        pl.plot(m['m5'], m['cadence'], 'r+')
        
    t = target.get(band, None)
    print target, t
    if t is not None:
        pl.plot(t[0], t[1], 
                color='r', marker='*', 
                markersize=15)
    
    pl.xlabel('$m_{5\sigma}$', fontsize=18)
    pl.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
    pl.title('$%s$' % band.split(':')[-1], fontsize=18)
    pl.xlim(mag_range)
    pl.ylim(dt_range)
    pl.grid(1)


def texp_cadence_plot(lsstpg, band, lims, log=None, texp_range=(30., 1200.), dt_range=(0.5, 15.), SNR=20):
    dt    = np.linspace(dt_range[0], dt_range[1], 100)
    texp = np.linspace(texp_range[0], texp_range[1], 100)
    b = [band] * len(texp)
    
    cadence_from_log, texp_from_log = [], []
    if log is not None:
        idx = log.band == band
        seeing = np.median(log.seeing[idx])
        mag_sky = np.median(log.mag_sky[idx])
        flux_sky = lsstpg.mag_to_flux([mag_sky], [band])
        r = log.split()
        for rr in r:
            idx = rr.band == band
            mjd = rr.mjd[idx]
            bins = np.arange(np.floor(mjd.min()), np.ceil(mjd.max()), 1.)
            c,b = np.histogram(mjd, bins=bins)
            cadence_from_log.append(1./c.mean())
            texp_from_log.append(np.median(rr.exptime[idx]))
    else:
        idx = lsstpg.data['band'] == band
        seeing = lsstpg.data[idx]['iq']
        mag_sky = lsstpg.data[idx]['mag_sky']
        flux_sky = lsstpg.mag_to_flux(mag_sky, [band]) # lsstpg.data[idx]['flux_sky']
        print " --> ", mag_sky, flux_sky, 
        
    f5 = lsstpg.flux_lim(texp, flux_sky, seeing, band)
    
    F5,DT = np.meshgrid(f5, dt)
    TEXT,DT = np.meshgrid(texp, dt)
    metric = np.sqrt(DT) * F5
    
    # draw limits 
    pl.figure()
    pl.imshow(metric, extent=(texp_range[0], texp_range[1], 
                              dt_range[0], dt_range[1]), 
              aspect='auto', alpha=0.25)
    
    if lims is not None:
        fmt = {}
        ll = [lims[zz][band] for zz in lims.keys()]
        print ll 
        cs = pl.contour(TEXT, DT, metric, ll, colors='k')
        
        strs = ['$z=%3.1f$' % zz for zz in lims.keys()]
        for l,s in zip(cs.levels, strs):
            fmt[l] = s
        pl.clabel(cs, inline=1, fmt=fmt, fontsize=16)

    # plot the real cadence achieved with OpSim
    pl.plot(texp_from_log, cadence_from_log, 'r+')
        
    pl.xlabel('$T_{exp}$', fontsize=18)
    pl.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
    pl.title('$%s$' % band.split(':')[-1], fontsize=18)
    pl.grid(1)
    
    
    
def plot_lc_snr(filename='Observations_DD_290_LSSTPG.txt', 
                z=1.1, X1=0., Color=0., DayMax=0., 
                bands=['LSSTPG::' + b for b in "rizy"], 
                rest_frame_margins=(-15., 30.),
                snr_min = 20.,
                fig=None):
    """Return a metric from a log and a fiducial supernova. 

    Args:
      z, X1, Color, DayMax: fiducial supernova 
      
    """
    lsstpg = psf.find('LSSTPG')
    colors = band_colors(lsstpg)
    mjd, shapes = get_pulse_shapes(bands, z=z, X1=X1, Color=Color)
    mjd_margins = rest_frame_margins[0] * (1.+z), rest_frame_margins[1] * (1.+z)
    
    log = snsim.OpSimObsLog(NTuple.fromtxt(filename))
    r = log.split(delta=100.)
    
    #    f5 = lsstpg.mag_to_flux(log.m5sigmadepth, log.band)
    flux_sky = lsstpg.mag_to_flux(log.mag_sky, log.band)
    f5 = lsstpg.flux_lim(log.exptime, flux_sky, log.seeing, log.band)
    #    print lsstpg.mag_lim(log.exptime, flux_sky, log.seeing, log.band)
    #    print lsstpg.mag_to_flux(lsstpg.mag_lim(log.exptime, flux_sky, log.seeing, log.band), log.band)
    
    if fig is None:
        fig = pl.figure(figsize=(12,12))
    ax = None

    mjd = np.arange(log.mjd.min(), log.mjd.max(), 1.)

    for i,bn in enumerate(bands):
        if ax is None:
            ax = pl.subplot(len(bands), 1, i+1)
            pl.title('$SN[X1=%5.1f,C=%5.1f]\ at\ z=%4.2f$ [%s]' % (X1, Color, z, filename))
        else:
            ax = pl.subplot(len(bands), 1, i+1)

        # runs 
        for rr in r:
            pl.axvspan(rr.mjd.min(), rr.mjd.max(), color='gray', alpha=0.25)
            pl.axvspan(rr.mjd.min() - rest_frame_margins[0], 
                       rr.mjd.max() - rest_frame_margins[1], 
                       color='gray', alpha=0.35)
        
        # plot the cadence metric 
        idx = log.band == bn
        c = CadenceMetric(log.mjd[idx], f5[idx], log.band[idx], shapes[bn])
        y = c(mjd)
        pl.plot(mjd, y, color=colors[bn], marker='.', ls=':')
        pl.ylabel('$SNR [%s]$' % bn.split(':')[-1], 
                  fontsize=16) #  color=colors[bn])
        pl.axhline(snr_min, ls='--', color='r')
        pl.ylim((0., max((y.max(), snr_min+1))))
        
        # plot the average number of observations averaged over a
        # window of ~ 21 days.
        c_sched = SimpleCadenceMetric(log.mjd[idx], 21.)
        print c_sched.sumw.min(), c_sched.sumw.max()
        y_sched = c_sched(mjd)
        ax = pl.gca().twinx()
        ax.plot(mjd, y_sched, color='gray', ls='-')
        pl.ylim((0., 0.28))
        pl.ylabel("Cadence [day$^{-1}$]", 
                  color='black')
        cad = 1. / (4. * (1.+z))
        pl.axhline(cad, ls=':', color='gray')
        
        if i<len(bands)-1:
            ax.get_xaxis().set_ticklabels([])
    pl.subplots_adjust(hspace=0.06)
    pl.xlabel('$MJD$ [days]', fontsize=16)
    
    return c


def check_m5sigma_depth():
    """
    Compare Philippe Gris' m5 with mine.
    """
    log = snsim.OpSimObsLog(NTuple.fromtxt('Observations_DD_290_LSSTPG.txt'))
    lsstpg = psf.find('LSSTPG')
    sky_flux = lsstpg.mag_to_flux(log.mag_sky, log.band)
    m5 = lsstpg.mag_lim(30., sky_flux, log.seeing, log.band)
    
    pl.figure()
    pl.plot(log.mjd, log.m5sigmadepth, marker='.', color='gray', ls='')
    pl.plot(log.mjd, m5, marker='.', color='r', ls='')
    pl.xlabel('MJD')

    for b in np.unique(log.band):
        idx = log.band == b
        pl.figure()
        pl.plot(log.mjd[idx], m5[idx]-log.m5sigmadepth[idx], marker='.', color='gray', ls='')
        pl.title(b)
    pl.xlabel('MJD')
    pl.xlabel('$\Delta m_{5\sigma} [%s]$' % b)
    

def main():
    
    z_lim=1.0
    fn = glob.glob('OpSimLogs/*_DD_*.txt')
    for f in fn:
        plot_lc_snr(filename=f, z=z_lim, X1=-3, Color=0.3, bands=['LSSTPG::' + b for b in "izy"])
        fig = pl.gcf()
        num = re.search('Observations_DD_(\d+).txt', f).group(1)
        fig.savefig('metric_DD_' + num + '.png', bbox_inches='tight')
        fig.savefig('metric_DD_' + num + '.pdf', bbox_inches='tight')
        
    z_lim = 0.4
    fn = glob.glob('OpSimLogs/*_WFD_*.txt')
    for f in fn:
        plot_lc_snr(filename=f, z=z_lim, X1=-3, Color=0.3, bands=['LSSTPG::' + b for b in "gri"])
        fig = pl.gcf()
        num = re.search('Observations_WFD_(\d+).txt', f).group(1)
        fig.savefig('metric_WFD_' + num + '.png', bbox_inches='tight')
        fig.savefig('metric_WFD_' + num + '.pdf', bbox_inches='tight')


def main_depth_ddf(instr_name='LSSTPG', 
                   bands=['r', 'i', 'z', 'y'], 
                   target={'LSSTPG::i': (25.75, 8.),
                           'LSSTPG::z': (25.06, 8.),
                           'LSSTPG::y': (25.10, 8.) }):
    
    instr = psf.find(instr_name)
    
    # DDF survey 
    logs = [snsim.OpSimObsLog(NTuple.fromtxt(fn)) for fn in glob.glob('OpSimLogs/*_DD_*.txt')]
    m = np.hstack([log.median_values() for log in logs])
    lims = f5_cadence_lims(zs=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1], 
                           bands=[instr_name + '::' + b for b in bands])
    for bn in bands:
        print target 
        f5_cadence_plot(instr, instr_name + '::' + bn, 
                        lims, 
                        target=target,
                        mag_range=(23., 26.), 
                        median_log_summary=m)
        pl.gcf().savefig('m5_cadence_limits_%s.png' % bn.split(':')[-1], 
                         bbox_inches='tight')
        pl.gcf().savefig('m5_cadence_limits_%s.pdf' % bn.split(':')[-1], 
                         bbox_inches='tight')
        
def main_depth_wide(instr_name='LSSTPG', 
                    bands=['g', 'r', 'i', 'z'], 
                    target={'LSSTPG::g': (24.4, 5.),
                            'LSSTPG::r': (23.0, 5.),
                            'LSSTPG::i': (22.6, 5.) }):
    
    instr = psf.find(instr_name)
    
    # Wide survey 
    logs = [snsim.OpSimObsLog(NTuple.fromtxt(fn)) for fn in glob.glob('OpSimLogs/*_WFD_*.txt')]
    m = np.hstack([log.median_values() for log in logs])
    lims = f5_cadence_lims(zs=[0.1, 0.2, 0.3, 0.4, 0.5],
                           bands=[instr_name + '::' + b for b in bands])
    for bn in bands:
        f5_cadence_plot(instr, instr_name + '::' + bn, 
                        lims, 
                        mag_range=(21., 24.8),
                        dt_range=(0.5, 30.),
                        median_log_summary=m,
                        target=target)
        pl.gcf().savefig('m5_cadence_limits_wide_%s.png' % bn.split(':')[-1], 
                         bbox_inches='tight')
        pl.gcf().savefig('m5_cadence_limits_wide_%s.pdf' % bn.split(':')[-1], 
                         bbox_inches='tight')
        
