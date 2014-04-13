"""
Load the Kepler light curve from FITS files. Also do a bit of preprocessing.
"""
import inputs as inp


def loadlc(files, usepdc=False, **kwargs):
    """
    Load Kepler light curves.

    Parameters
    ----------
    files : list of strings
        The locations of the light curves to load together.
    usepdc : bool, optional
        Set to True to load the PDC light curves instead of the
        default SAP.

    Returns
    -------
    time : ndarray
        Kepler times of center of exposure
    flux : ndarray
        Kepler normalized fluxes
    fluxerr : ndarray
        Kepler flux errors
    cadence : ndarray
        Kepler cadence number
    quarter : ndarray
        Kepler quarter
    quality : ndarray
        Kepler quality flag
    """
    import astropy.io.fits as pyfits
    import numpy as np

    # load the first file
    ifile = files[0]
    data = pyfits.getdata(ifile)

    # get the times and fluxes
    time = data['time']+54833e0
    flux = data['sap_flux']
    fluxerr = data['sap_flux_err']
    if usepdc:
        flux = data['pdcsap_flux']
        fluxerr = data['pdcsap_flux_err']
    # where the times and fluxes are finite
    good = (np.isfinite(time) & np.isfinite(flux))

    # get the good values of everything
    time = time[good]
    flux = flux[good]
    fluxerr = fluxerr[good]
    quality = data['sap_quality'][good]
    cadence = data['cadenceno'][good]
    # pull the quarter from the header and set it up as an array
    quart = pyfits.getval(ifile, 'quarter', 0)
    quarter = np.zeros(len(time)) + quart

    # normalize the fluxes
    fluxerr /= np.median(flux)
    flux /= np.median(flux)

    # add in subsequent files
    for i in np.arange(len(files)-1)+1:
        ifile = files[i]
        data = pyfits.getdata(ifile)
        # get the times and fluxes
        itime = data['time']+54833e0
        iflux = data['sap_flux']
        ifluxerr = data['sap_flux_err']
        if usepdc:
            iflux = data['pdcsap_flux']
            ifluxerr = data['pdcsap_flux_err']

        # where the times and fluxes are finite
        good = (np.isfinite(itime) & np.isfinite(iflux))

        # get the good values of everything
        itime = itime[good]
        iflux = iflux[good]
        ifluxerr = ifluxerr[good]
        iquality = data['sap_quality'][good]
        icadence = data['cadenceno'][good]
        # pull the quarter from the header and set it up as an array
        quart = pyfits.getval(ifile, 'quarter', 0)
        iquarter = np.zeros(len(itime)) + quart

        # normalize the fluxes
        ifluxerr /= np.median(iflux)
        iflux /= np.median(iflux)

        time = np.concatenate((time, itime))
        flux = np.concatenate((flux, iflux))
        fluxerr = np.concatenate((fluxerr, ifluxerr))
        quality = np.concatenate((quality, iquality))
        cadence = np.concatenate((cadence, icadence))
        quarter = np.concatenate((quarter, iquarter))

    # guarantee the light curve in sequential order
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    fluxerr = fluxerr[order]
    quality = quality[order]
    cadence = cadence[order]
    quarter = quarter[order]

    return time, flux, fluxerr, cadence, quarter, quality


def preparelc(KIC, dataloc=inp.keplerdata, fill=True,
              badflags=(128, 2048), ignorelist=inp.baddata,
              **kwargs):
    """
    Load Kepler light curves, then process them for analysis.

    Parameters
    ----------
    KIC : int
        The Kepler ID for the system to look at
    dataloc : string, optional
        Directory point to the location of the Kepler light curves.
        Default can be changed in the module initialization.
    fill : boolean, optional
        Should we fill in all missing cadences? If true, will
        interpolate times to all missing cadences and assign them
        flux with np.inf errors. Necessary for QATS requiring
        continuous data. Default True.
    badflags : tuple, optional
        Flags that can be set by Kepler that we should take seriously
        and ignore.
        Set all cadences with these flags to have infinite errors.
        Default 128 and 2048.
    ignorelist : string, optional
        File containing regions of time to ignore. File contents should
        be 2 columns, with start and end times (in times already
        adjusted by inp.timeoffset). Defaults to the file listed in the
        module.

    Returns
    -------
    time : ndarray
        Kepler times of center of exposure
    flux : ndarray
        Kepler normalized fluxes
    fluxerr : ndarray
        Kepler flux errors
    cadence : ndarray
        Cadence number, starting at 0
    quarter : ndarray
        Kepler quarter
    quality : ndarray
        Kepler quality flag
    """
    from glob import glob
    import numpy as np
    from scipy import interpolate

    # load the lightcurve from FITS files
    KICstr = str(int(KIC))
    files = glob(dataloc + 'kplr*' + KICstr + '*llc.fits')
    time, flux, fluxerr, cad, quart, qual = loadlc(files, **kwargs)
    time -= inp.timeoffset

    # make sure cadences start at 0
    cad -= cad[0]

    if fill:
        # fill in the missing cadences and interpolate their times and
        # fluxes (though the flux errors will be infinite)
        newcad = np.arange(cad[-1]+1)
        time = np.interp(newcad, cad, time)

        newfluxerr = newcad * 0. + np.inf
        newfluxerr[cad] = fluxerr

        # fill in the old fluxes, etc to the new grid
        newflux = newcad * 0. + 1.
        newflux[cad] = flux

        newqual = newcad * 0
        newqual[cad] = qual
        # default to quarter -1 for filled in gaps
        newquart = newcad * 0. - 1.
        newquart[cad] = quart

        cad = newcad
        flux = newflux
        fluxerr = newfluxerr
        qual = newqual
        quart = newquart

        # fill in the infinite flux errors with interpolated values
        # to make plotting look better
        func = interpolate.interp1d(time[np.isfinite(fluxerr)],
                                    flux[np.isfinite(fluxerr)],
                                    bounds_error=False, fill_value=1.)
        flux[~np.isfinite(fluxerr)] = func(time[~np.isfinite(fluxerr)])

    # ignore the places with these bad flags
    for ii in badflags:
        bad = np.where(qual & ii)[0]
        fluxerr[bad] = np.inf

    # ignore these regions for whatever reason
    if ignorelist is not None:
        tstart, tend = np.loadtxt(ignorelist, unpack=True, ndmin=2)
        for ii in np.arange(len(tstart)):
            igsrch = np.where((time >= tstart[ii]) & (time <= tend[ii]))[0]
            fluxerr[igsrch] = np.inf

    return time, flux, fluxerr, cad, quart, qual
