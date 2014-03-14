"""
Take in oribtal and stellar parameters and turn them into an eclipsing light curve.

Also run an MCMC analysis and/or find an initial fit.

This time we use isochrones and observed magnitudes. Also use age constraints
to make sure the masses/MS ages/etc are consistent.

Vectorized to run faster.
"""
import ekruse as ek
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
from glob import glob
from scipy import optimize as opt
import matplotlib.ticker as plticker
import cPickle
from scipy import interpolate
import scipy.stats
import sys
import numpy.polynomial.polynomial as poly

# NOTE: make sure I'm not tweaking t, f, ferr or any other inputs

# isochrone files
padsdssfiles = '/astro/users/eakruse/microlens/padova/*sdss'
padwisefiles = '/astro/users/eakruse/microlens/padova/*wise'

wdfiles = '/astro/users/eakruse/microlens/wdmodels/Table_Mass*'
# what's in the isochrone and what its index is
wdinds = {'g':13,'r':14,'i':15,'teff':0,'logg':1,'age':26}

# whether or not to use the adjustments for crowding
usecrowd = True
# crowding value in Kepler for each quarter (1-17)
quartcontam = np.array([0.9619, 0.9184, 0.9245, 0.9381, 0.9505, 0.9187, 0.9246, 0.9384, 0.9598, 0.9187, 0.9248, 0.9259, 0.9591, 0.9186, 0.9244, 0.9383, 0.9578])
# quarter for each event
equarts = np.array([ 1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,  16, 16, 17])

# use the scipy.optimize to get an initial fit
findfit = False
# run the MCMC
domcmc = False
# where to save the MCMC output
outfile = './chain_final_isochrones.txt'

# fit the limb darkening parameters
# if False, uses the Sing fits based on the stellar parameters
fitlimb = False

# subsample/npert: change sampling of times per cadence
subsample  = 10

# time/fluxes/flux errors for all events to fit
infile = './KOI3278_events_sap.txt'

# number of MCMC iterations
niter = 100000

# save the sampler object to inspect later
saveresults = False
picklefile = None
thin = 10

# multiply the errors by this amount to get a reduced chi-square closer to 1.
expanderror = 1.13

# =========================================================================

# set up the crowding parameters for each event
crowding = np.ones(len(equarts))
if usecrowd:
    for ii in np.arange(len(crowding)):
        crowding[ii] = quartcontam[equarts[ii]-1]

# this is the full list from Eric
magname = ['u','g','r','i','z','U','B','V','R','I','J','H','Ks','Kp','D51','W1','W2']
magobs = np.array([16.990,15.245,14.617,14.435,14.330,16.145,15.647,14.920,14.620,14.020,13.447,13.039,12.977, 14.71, 15.094, 12.959, 13.020])
# sig_mag= np.array([ 0.008, 0.003, 0.004, 0.003, 0.004, 0.025, 0.025, 0.018, 0.430, 1.000, 0.024, 0.022, 0.032,  0.50,  0.020,  0.026,  0.029])
# The u, U, Kp & D51 bands seem to have the largest discrepancy between Dartmouth & Padova, so I will ignore these:
magerr = np.array([np.inf ,0.003,  0.004, 0.003, 0.004, np.inf , 0.025, 0.018, 0.430, 1.000, 0.024, 0.022, 0.032, np.inf , np.inf,  0.026,  0.029])
# extinction measure
maglam = np.array([ 0.460, 0.359, 0.248, 0.184, 0.137, 0.446, 0.395, 0.291, 0.230, 0.165, 0.078, 0.050, 0.034,  0.25,  0.325,  0.021,  0.016])

# this is what I currently have in the isochrones and want to use
magname = ['g','r','i','z','J','H','Ks','W1','W2']
magobs = np.array([15.245,14.617,14.435,14.330,13.447,13.039,12.977, 12.959, 13.020])
magerr = np.array([0.003, 0.004, 0.003, 0.004, 0.024, 0.022, 0.032,  0.026,  0.029])
# extinction measure
maglam = np.array([0.359, 0.248, 0.184, 0.137, 0.078, 0.050, 0.034, 0.021,  0.016])

# we want to add in these components
magname.append('Rad')
magname.append('logg')
magname.append('teff')

magname = np.array(magname)
# ============================================================================ #
# isochrone loading section
try:
    loaded
except NameError:
    loaded = 1
    sdssisos = glob(padsdssfiles)
    sdssisos.sort()
    wiseisos = glob(padwisefiles)
    wiseisos.sort()

    if len(sdssisos) != len(wiseisos):
        print 'Error! Mismatched isochrones.'
        sys.exit(1)

    # load and concatenate the isochrones
    for ii in np.arange(len(sdssisos)):
        iso1 = np.loadtxt(sdssisos[ii])
        iso2 = np.loadtxt(wiseisos[ii])
        # first 8 columns are the same, then you get into the bands
        # next 3 are repeats of JHK, so ignore those too
        together = np.concatenate((iso1,iso2[:,11:]),axis=1)
        if ii == 0:
            fulliso = together * 1.
        else:
            fulliso = np.concatenate((fulliso, together))
    # what's in the isochrone and what its index is
    inds = {'feh':0,'age':1,'M':2,'Mact':3,'lum':4,'teff':5,'logg':6,'mbol':7,'Kp':8,'g':9,'r':10,'i':11,'z':12,'D51':13,'J':14,'H':15,'Ks':16,'int_IMF1':17,'3.6':18,'4.5':19,'5.8':20,'8.0':21,'24':22,'70':23,'160':24,'W1':25,'W2':26,'W3':27,'W4':28,'int_IMF2':29,'Rad':30}

    # which ones we want to solve for eventually
    maginds = []
    for ii in magname:
        maginds.append(inds[ii])
    maginds = np.array(maginds)
    maginds.astype(int)

    # convert from Padova Z into metallicity
    zs = fulliso[:,inds['feh']]
    fesol = 0.0147
    hsol = 0.7106
    hnew = 1. - 0.2485 - 2.78 * zs
    fulliso[:,inds['feh']] = np.log10((zs/hnew)/(fesol/hsol))

    # calculate radii two different ways
    G = 6.67e-8 # cgs units
    Msun = 1.9884e33 # g
    Rsun = 6.955e10 # cm
    R = np.sqrt(G * fulliso[:,inds['M']] * Msun / 10.**fulliso[:,inds['logg']]) / Rsun
    sigma = 5.6704e-5 # cgs
    Lsun = 3.846e33 # erg/s
    R2 = np.sqrt(10.**fulliso[:,inds['lum']] * Lsun / (4. * np.pi * sigma * (10.**fulliso[:,inds['teff']])**4.)) / Rsun
    R = (R + R2)/2.
    # add in a radius measurement
    fulliso = np.concatenate((fulliso,R[:,np.newaxis]),axis=1)

    # what are the metallicities and ages in this set?
    fehs = np.unique(fulliso[:,inds['feh']])
    ages = np.unique(fulliso[:,inds['age']])
    minfeh = fehs.min()
    maxfeh = fehs.max()
    minage = ages.min()
    maxage = ages.max()


    # set up the mass interpolations
    interps = np.zeros((len(fehs),len(ages)),dtype=object)
    maxmasses = np.zeros((len(fehs),len(ages)))

    for ii in np.arange(len(fehs)):
        for jj in np.arange(len(ages)):
            small = np.where((fulliso[:,inds['feh']] == fehs[ii]) & (fulliso[:,inds['age']] == ages[jj]))[0]
            interps[ii,jj] = interpolate.interp1d(fulliso[small,inds['M']],fulliso[small][:,maginds],axis=0,bounds_error=False)
            maxmasses[ii,jj] = fulliso[small,inds['M']].max()


    # set up the WD section
    files = glob(wdfiles)

    for ct, ii in enumerate(files):
        iwdmods = np.loadtxt(ii)
        # pull the mass out of the file name
        imass = float(ii[-3:])
        # only grab the H WDs, ignore the He ones
        iwdmods = iwdmods[:np.diff(iwdmods[:,wdinds['teff']]).argmin()+1,:]
        imass = np.ones(len(iwdmods[:,0])) * imass

        if ct == 0:
            wdmods = iwdmods * 1.
            mass = imass * 1.
        else:
            mass = np.concatenate((mass,imass))
            wdmods = np.concatenate((wdmods,iwdmods))

    kpmag = np.zeros(len(wdmods[:,0]))
    blue = wdmods[:,wdinds['g']] - wdmods[:,wdinds['r']] <= 0.3
    kpmag[blue] = 0.25 * wdmods[blue,wdinds['g']] + 0.75 * wdmods[blue,wdinds['r']]
    kpmag[~blue] = 0.3 * wdmods[~blue,wdinds['g']] + 0.7 * wdmods[~blue,wdinds['i']]

    # WD models contains the WD mass, age, and Kp magnitude
    wdmodels = np.zeros((len(mass),3))
    wdmodels[:,0] = mass
    # log age like everything else
    wdmodels[:,1] = np.log10(wdmods[:,wdinds['age']])
    wdmodels[:,2] = kpmag
    wdmagfunc = interpolate.LinearNDInterpolator(wdmodels[:,0:2],wdmodels[:,2])

    limits = (minfeh,maxfeh,minage,maxage)
    # what we're going to feed all the functions
    isobundle = (magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc)

print 'Done loading isochrones'
# ============================================================================ #

# current parametrization and starting fit
labels = ['$P$ (days)','$t_{tran}$ (days)','$e\cos\omega$','$e\sin\omega$','$b$','$M_{2,init}$','$M_2$','$M_1$','[Fe/H]','Age (Gyr)','Distance (pc)','$\sigma_{sys}$','h (pc)','$A_\lambda$ scale']
# median solution from the MCMC analysis (without crowding)
p = np.array([  8.81804953e+01,   8.54189557e+01,   1.47075839e-02, 3.88439171e-03,   7.10199165e-01,   2.20962752e+00, 6.11162960e-01,   1.03747625e+00,   3.80623768e-01, 1.91048678e+00,   8.09286733e+02,   2.41907976e-02, 1.17995034e+02,   1.00266715e+00])
# absolute minimum chi-square found in MCMC (without crowding)
p = np.array([  8.81806192e+01,   8.54188445e+01,   1.47156750e-02, 6.15356399e-03,   7.05240224e-01,   2.38417117e+00, 5.96523680e-01,   1.02311930e+00,   3.09126840e-01, 1.71942645e+00,   7.87965211e+02,   1.73709656e-02, 1.13214456e+02,   9.97091028e-01])

# median solution from the MCMC analysis (with crowding)
#p = np.array([  8.81805192e+01,   8.54189812e+01,   1.47126486e-02, 6.75188973e-04,   7.06050040e-01,   2.35077177e+00, 6.33579046e-01,   1.04081228e+00,   3.90799792e-01, 1.67782257e+00,   8.08645617e+02,   2.48914843e-02, 1.17309541e+02,   1.00106015e+00])
p = np.array([  8.81805180e+01,   8.54189900e+01,   1.47132293e-02, 4.83767012e-04,   7.05595086e-01,   2.40081224e+00, 6.33573877e-01,   1.04177206e+00,   3.94625983e-01, 1.62016796e+00,   8.08342999e+02,   2.46057348e-02, 1.17068978e+02,   1.00122149e+00])


# absolute minimum chi-square found in MCMC (with crowding)
#p = np.array([  8.81804882e+01,   8.54191929e+01,   1.47396630e-02, 5.30922862e-04,   7.15370985e-01,   1.99619503e+00, 6.27306194e-01,   1.02279313e+00,   3.22443203e-01, 2.34152377e+00,   7.99956109e+02,   1.74440753e-02, 1.15856745e+02,   9.82378626e-01])
p = np.array([  8.81805979e+01,   8.54189422e+01,   1.47105950e-02, 5.83059972e-03,   7.02722610e-01,   2.35546161e+00, 6.26868773e-01,   1.03255051e+00,   3.46963869e-01, 1.71307399e+00,   7.99324162e+02,   1.51296591e-02, 1.23274350e+02,   1.00831069e+00])


if fitlimb:
    p = np.concatenate((p,np.array([5.64392567e-02, 5.07460729e-01])))
    labels.append('$u_{S1,1}$')
    labels.append('$u_{S1,2}$')




def initrange(p):
    """
    Return initial error estimates in each parameter.
    See light_curve_model for the order of the parameters.
    """
    # labels = ['$P$ (days)','$t_{tran}$ (days)','$e\cos\omega$','$e\sin\omega$','$b$','$M_{2,init}$','$M_2$','$M_1$','[Fe/H]','Age (Gyr)','Distance (pc)','$\sigma_{sys}$','h (pc)','$A_\lambda$ scale','$u_{S1,1}$','$u_{S1,2}$']
    if len(p) == 14:
        return np.array([  2.41373687e-04,   2.15625144e-03,   5.42134862e-05, 4.95601162e-02,   4.33065931e-02,   1.19787163e-01, 1.03564141e-01,   9.25556734e-02,  2.34386318e-01,  2.34386318e-01,  2.34386318e+01,  1.34386318e-03, 5.26059466e+00,   1.00059466e-02])

    if len(p) == 18:
        return np.array([  2.41373687e-04,   2.15625144e-03,   5.42134862e-05, 4.95601162e-02,   4.33065931e-02,   1.19787163e-01, 1.03564141e-01,   9.25556734e-02,   2.34386318e-01,  2.34386318e-01,  2.34386318e+01,  1.34386318e-03, 5.26059466e+00,   1.00059466e-02, 0.1, 0.1, 0.1, 0.1])

    if len(p) == 16:
        return np.array([  2.41373687e-04,   2.15625144e-03,   5.42134862e-05, 4.95601162e-02,   4.33065931e-02,   1.19787163e-01, 1.03564141e-01,   9.25556734e-02,   2.34386318e-01,  2.34386318e-01,  2.34386318e+01,  1.34386318e-03, 5.26059466e+00,   1.00059466e-02, 0.1, 0.1])

# load in the sections of the light curve near transits
t,f,ferr = np.loadtxt(infile,unpack=True)
ferr *= expanderror

def isointerp(M,FeH,age,isobundle,testvalid = False):
    """
    """
    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle
    minfeh, maxfeh, minage, maxage = limits

    if age >= minage and age <= maxage and FeH >= minfeh and FeH <= maxfeh:
        fehinds = np.digitize([FeH],fehs)
        fehinds = np.concatenate((fehinds-1,fehinds))
        ageinds = np.digitize([age],ages)
        ageinds = np.concatenate((ageinds-1,ageinds))

        # bilinear interpolation
        fehdiff = np.diff(fehs[fehinds])[0]
        agediff = np.diff(ages[ageinds])[0]

        interp1 = (interps[fehinds[0],ageinds[0]](M) * (fehs[fehinds[1]] - FeH) + interps[fehinds[1],ageinds[0]](M) * (FeH - fehs[fehinds[0]])) / fehdiff
        interp2 = (interps[fehinds[0],ageinds[1]](M) * (fehs[fehinds[1]] - FeH) + interps[fehinds[1],ageinds[1]](M) * (FeH - fehs[fehinds[0]])) / fehdiff

        result = (interp1 * (ages[ageinds[1]] - age) + interp2 * (age - ages[ageinds[0]])) / agediff
    else:
        result = np.zeros(len(magobs)+3)
        result[:] = np.nan
    if testvalid:
        return np.isfinite(result).all()
    else:
        return result

def wdkpmag(M,age,isobundle):
    """

    """
    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle
    minfeh, maxfeh, minage, maxage = limits

    # WD models contains the WD mass, age, and Kp magnitude
    #mag = interpolate.griddata(wdmodels[:,0:2],wdmodels[:,-1],np.array([[M,age]]))
    #magfunc = interpolate.interp2d(wdmodels[:,0],wdmodels[:,1],wdmodels[:,2])
    #mag = magfunc(M,age)
    mag = wdmagfunc(np.array([[M,age]]))

    return mag[0]


def msage(M,FeH,isobundle):
    """
    Return the liftime of the star of mass M and metallicity FeH based on the
    isochrones in the isobundle.

    Returns log10(age [years])
    """
    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle
    minfeh, maxfeh, minage, maxage = limits

    if FeH >= minfeh and FeH <= maxfeh:
        fehinds = np.digitize([FeH],fehs)
        fehinds = np.concatenate((fehinds-1,fehinds))

        twoages = np.zeros(len(fehinds))
        # for each metallicity
        for ii in np.arange(len(fehinds)):
            # where the max mass is still bigger than the current guess
            srch = np.where(maxmasses[fehinds[ii],:] >= M)[0]
            # very short MS lifetime, not even on the isochrones
            if len(srch) == 0:
                return ages.min()
            srch = srch[-1]
            # hasn't evolved yet! return error?
            if srch == len(ages) - 1:
                return ages.max()
            bounds = maxmasses[fehinds[ii],srch:srch+2]
            diff = bounds[0] - bounds[1]
            twoages[ii] = (bounds[0] - M)/diff * ages[srch+1] + (M - bounds[1])/diff * ages[srch]

        diff = fehs[fehinds[1]] - fehs[fehinds[0]]
        finalage = (fehs[fehinds[1]] - FeH)/diff * twoages[0] + (FeH - fehs[fehinds[0]])/diff * twoages[1]
        return finalage
    return 0.

def kepler_problem(M,e):
    """
    Simple Kepler solver.
    Iterative solution via Newton's method taken from Charbonneau's
    Astronomy 110 problem set. Could likely be sped up, but this works for
    now.

    Input
    -----
    M : ndarray
    e : float or ndarray of same size as M

    Returns
    -------
    E : ndarray
    """
    import numpy as np
    # start with this guess
    M = np.array(M)
    E = M * 1.
    err = M * 0. + 1.
    while err.max() > 1e-8:
        # solve via Newton's method
        guess = E - (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
        err = np.abs(guess - E)
        E = guess
    return E

def light_curve_model(t,p,isobundle,npert=1):
    """
    Given the orbital parameters in p, computer a model light curve at times
    t, sampling at the rate of npert.
    Currently p must contain:
    period, time of center of transit, ecos(omega), esin(omega), impact
    parameter, flux ratios, mass of star 2, mass of star 1,
    metallicity of star 1, log of age of the system, distance, and systematic magnitude errors.
    May contain additional 2 or 4 quadratic limb darkening parameters
    (to fit one star or both).

    Input
    -----
    t : ndarray
        Times to return the model light curve.
    p : ndarray
        Orbital parameters. See above for the order. Must be length 9/11/13.
    isobundle : tuple
        (magobs, magerr, maglam, interps, limits, fehs, ages)
        Contains everything needed for the isochrones
    npert : int, optional
        Sampling rate per cadence. Final light curve will average each
        cadence over this many samples.

    Returns
    -------
    fluxes : ndarray
        Light curve corresponding to the times in t.
    """

    # fix limb darkening
    if len(p) == 14:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult = p
        u20 = 0.
        u21 = 0.
    # fit limb darkening for both stars
    if len(p) == 18:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11, u20, u21 = p
    # fit limb darkening for primary star
    if len(p) == 16:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11 = p
        u20 = 0.
        u21 = 0.

    # to get in log(age) like the interpolations need
    age = np.log10(age * 1e9)

    wdage = np.log10(10.**age - 10.**(msage(M2init,FeH,isobundle)))
    wdmag = wdkpmag(M2,wdage,isobundle)

    mags = isointerp(M1,FeH,age,isobundle)

    R1 = mags[-3]
    logg = mags[-2]
    Teff = 10.**mags[-1]
    if len(p) == 14:
        # get the limb darkening from the fit to Sing?
        u10 = 0.44657704  -0.00019632296 * (Teff-5500.) +   0.0069222222 * (logg-4.5) +    0.086473504 *FeH
        u11 = 0.22779778  - 0.00012819556 * (Teff-5500.) - 0.0045844444  * (logg-4.5)  -0.050554701 *FeH
    u1 = np.array([u10,u11])
    u2 = np.array([u20,u21])

    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle

    F2F1 = 0.
    if np.isfinite(wdmag):
        # get the Kp magnitude of the main star
        gind = np.where(magname == 'g')[0][0]
        rind = np.where(magname == 'r')[0][0]
        iind = np.where(magname == 'i')[0][0]
        if mags[gind] - mags[rind] <= 0.3:
            kpmag1 = 0.25 * mags[gind] + 0.75 * mags[rind]
        else:
            kpmag1 = 0.3 * mags[gind] + 0.7 * mags[iind]

        F2F1 = 10.**((wdmag - kpmag1)/(-2.5))

    if not np.isfinite(F2F1):
        F2F1 = 0.

    # reconvert into more useful orbital elements
    e = np.sqrt(ecosw**2. + esinw**2.)
    omega = np.arctan2(esinw,ecosw)
    a = ((period * 86400.)**2. * 6.67e-11 * (M1 + M2) * 1.988e30 / (4.*np.pi**2.))**(1./3) # in m
    a = a / (6.955e8 * R1) # in radii of the first star
    # Simple conversion
    inc = np.arccos(b/a)
    # Compute the size of the WD using the Nauenberg relation:
    MCh = 1.454
    # in Solar radii
    R2 = .0108*np.sqrt((MCh/M2)**(2./3.)-(M2/MCh)**(2./3.))
    rrat = R2 / R1

    n = 2. * np.pi / period

    # cadence for this data set
    medt = np.median(t[1:]-t[:-1])
    # generate npert subcadences, equally spaced
    tmfine = np.linspace(-medt/2.,+medt/2.,npert+1)
    tmfine = tmfine[:-1] + (tmfine[1] - tmfine[0])/2.
    # all times to evaluate fluxes at
    # has shape (t, npert)
    newt = t[:,np.newaxis] + tmfine
    # has to be a vector for Mandel-Agol function
    t = newt.reshape((-1,))

    # Sudarsky 2005 Eq. 9 to convert between center of transit and pericenter passage (tau)
    edif = 1.-e**2.
    fcen = np.pi/2. - omega
    tau = ttran + np.sqrt(edif)*period / (2.*np.pi) * ( e*np.sin(fcen)/(1.+e*np.cos(fcen)) - 2./np.sqrt(edif) * np.arctan(np.sqrt(edif)*np.tan(fcen/2.)/(1.+e)))

    # define the mean anomaly
    M = (n * (t - tau)) % (2. * np.pi)
    E = kepler_problem(M,e)

    # solve for f
    tanf2 = np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.)
    fanom = (np.arctan(tanf2)*2.) % (2. * np.pi)

    r = a * (1. - e**2.) / (1. + e * np.cos(fanom))
    # projected distance between the stars (in the same units as a)
    projdist = r * np.sqrt(1. - np.sin(omega + fanom)**2. * np.sin(inc)**2.)

    # positive z means body 2 is in front (transit)
    Z = r * np.sin(omega + fanom ) * np.sin(inc)

    # get the lens depth given this separation at transit
    # 1.6984903e-5 gives 2*Einstein radius^2/R1^2 = 8GMZ/(c^2 R^2) with M, Z, R all
    # scaled to solar values
    lensdep = 1.6984903e-5 * M2 * np.abs(Z) / R1 - rrat**2.
    # then get it into the form I use
    lensdep = (lensdep / rrat**2.) + 1.

    # fluxes of each body, adjusted by their relative fluxes
    F1t = t * 0. + 1.
    F2t = t * 0. + F2F1

    # object 2 passes in front of object 1
    transits = np.where((projdist < 1. + rrat) & (Z > 0.))[0]
    if len(transits) > 0:
        # limb darkened light curves for object 1
        ldark  = ek.mandel_agol(projdist[transits],u1[0],u1[1],rrat)
        # object 1 also has microlensing effects
        F1t[transits] *= (ldark + (1. - ldark)*lensdep[transits])

    # object 1 passes in front of object 2
    occults = np.where((projdist < 1. + rrat) & (Z < 0.))[0]
    if len(occults) > 0:
        # must be in units of the blocked star/object radius for Mandel/Agol function
        # so divide by the radius ratio
        ldark = ek.mandel_agol(projdist[occults]/rrat,u2[0],u2[1],1./rrat)
        F2t[occults] *= ldark

    # get back to the proper shape
    F1t = F1t.reshape(newt.shape)
    F2t = F2t.reshape(newt.shape)
    # get the average value for each cadence
    F1t = F1t.mean(axis=1)
    F2t = F2t.mean(axis=1)

    # return a normalized light curve
    normed = (F1t + F2t)/(1. + F2F1)
    return normed

def logprior(p,isobundle):
    """
    Priors on the input parameters.

    Input
    -----
    p : ndarray
        Orbital parameters. See light_curve_model for the order.
        Must be length 9/11/13.
    isobundle : tuple
        (magobs, magerr, maglam, interps, limits)
        Contains everything needed for the isochrones

    Returns
    -------
    prior : float
        Log likelihood of this set of input parameters based on the
        priors.
    """

    # fix limb darkening
    if len(p) == 14:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult = p
        u20 = 0.
        u21 = 0.
        # for the sake of the limits below just make up a valid number for these
        u10 = 0.1
        u11 = 0.1
    # fit limb darkening for both stars
    if len(p) == 18:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11, u20, u21 = p
    # fit limb darkening for primary star
    if len(p) == 16:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11 = p
        u20 = 0.
        u21 = 0.

    # to get in log(age) like the interpolation needs
    age = np.log10(age * 1e9)

    # check to make sure that it's valid within the models.
    if not isointerp(M1,FeH,age,isobundle,testvalid=True):
        return -np.inf

    # reconvert into more useful orbital elements
    e = np.sqrt(ecosw**2. + esinw**2.)
    a = ((period * 86400.)**2. * 6.67e-11 * (M1 + M2) * 1.988e30 / (4.*np.pi**2.))**(1./3) # in m
    # not worth running the isochrone fit to get the one R1 number
    #a = a / (6.955e8 * R1) # in radii of the first star

    # if any of the parameters are unphysical, return negative infinity
    # log likelihood (impossible)
    if period < 0. or e < 0. or e >= 1. or a < 1. or u10 + u11 >= 1 or u20 + u21 >= 1 or M2 < 0. or M2init < 0. or height < 0. or M1 < 0. or dist < 0. or syserr < 0.:
        return -np.inf
    # otherwise return a uniform prior (except modify the eccentricity to
    # ensure the prior is uniform in e)
    return 0. - np.log(e)


def loglikeli(p,t,f,ferr,cuts,crowding,isobundle, minimize = False,retmodel = False,retpoly=False, indchi=False,**kwargs):
    """
    Compute the log likelihood of a microlensing signal with these orbital
    parameters given the data. By default returns this value, but can
    optionally return the full model light curve or just the polynomial
    portion of the light curve instead.

    Input
    -----
    p : ndarray
        Orbital parameters. See light_curve_model for the order.
        Must be length 9/11/13.
    t, f, ferr : ndarray
        times, fluxes, and flux errors of the data.
    cuts : ndarray
        Beginning of a new segment of data to analyze separately.
        Last element must be len(t).
    isobundle : tuple
        (magobs, magerr, maglam, interps, limits)
        Contains everything needed for the isochrones
    crowding : ndarray
        Must be an array of len(cuts)-1
        indicating what fraction of the light is due to the binary system.
        1 - crowding is the contamination from outside sources.
        If purely light from the system in question, should be just
        np.ones(len(cuts)-1)
    minimize : boolean, optional
        If True, we are trying to minimize the chi-square rather than
        maximize the likelihood. Default False.
    retmodel : boolean, optional
        If True, return the model fluxes instead of the log likelihood.
        Default False.
    retpoly : boolean, optional
        If True, return the polynomial portion of the model fluxes
        instead of the log likelihood. Default False.
    indchi : boolean, optional
        If True, return the chi-square of each individual event.
        Default False.

    Return
    ------
    likeli : float
        Log likelihood that the model fits the data.
    """
    # don't modify the originals
    tt = t * 1.
    ff = f * 1.
    fferr = ferr * 1.

    # compute the model light curve
    model = light_curve_model(t,p,isobundle,**kwargs)

    ncuts = cuts[-1] + 1

    # add in the contamination from outside light sources
    if crowding is not None:
        model = model * crowding[cuts] + 1. - crowding[cuts]

    # now has shape (ncuts, tper)
    tt = tt.reshape((ncuts,-1))
    ff = ff.reshape((ncuts,-1))
    fferr = fferr.reshape((ncuts,-1))
    model = model.reshape((ncuts,-1))

    tmeds = np.median(tt,axis=1)
    tt -= tmeds[:,np.newaxis]


    # marginalize over the polynomial detrending

    # these are just all orders we want to compute
    pord = 2
    pords = np.arange(pord+1)
    # 1-d and 3-d blank arrays to allow for numpy array broadcasting
    # in a bit
    ones = np.ones((pord+1))
    ones3d = np.ones((ncuts,1,pord+1,pord+1))

    # every time to every polynomial order power
    # has shape (ncuts, pert, pords)
    tpow = tt[:,:,np.newaxis] ** pords
    # this is the same for every polynomial order.
    # has shape (ncuts, pert, pords)
    prefix = ((ff/model) / (fferr/model)**2.)[:,:,np.newaxis] * ones

    # get the data side of the equation. Just has shape (ncuts, pords) because we summed over (pert)
    Bmat = np.sum(prefix * tpow,axis=(1,))

    # get the time**pords for both the j and k indices
    # has shape (ncuts, pert, 1, pords)
    j = tt[:,:,np.newaxis,np.newaxis] ** pords[np.newaxis,np.newaxis,np.newaxis,:]
    # has shape (ncuts, pert, pords, 1)
    k = tt[:,:,np.newaxis,np.newaxis] ** pords[np.newaxis,np.newaxis,:,np.newaxis]
    # has shape (ncuts, pert, pords, pords)
    Mbig = j*k

    # this gets divided in. should be the same for each pord, but needs to be the right shape
    # has shape (ncuts, pert, pords, pords)
    divider = ((fferr/model)**2.)[:,:,np.newaxis,np.newaxis] * ones3d
    Mbig = Mbig / divider
    # sum over all times so now
    # has shape (ncuts,pords,pords)
    Mfinal = np.sum(Mbig,axis=1)

    solution = np.array([np.linalg.lstsq(Mfinal[ii,:,:],Bmat[ii,:])[0] for ii in np.arange(ncuts)]).swapaxes(0,1)

    solution = solution[:,cuts]

    # get the optimal polynomial model for this segment of data
    polymodel = poly.polyval(tt.reshape((-1,)),solution,tensor=False)

    # compute the chi-square of this segment
    totchisq = np.sum(((ff - model * polymodel.reshape((ncuts,-1)))/fferr)**2.,axis=1)

    # add these to the return arrays if necessary
    if retmodel:
        return model.reshape((-1,)) * polymodel
    if retpoly:
        return polymodel
    if indchi:
        return totchisq

    totchisq = np.sum(totchisq)

    # fix limb darkening
    if len(p) == 14:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult = p
        u20 = 0.
        u21 = 0.
    # fit limb darkening for both stars
    if len(p) == 18:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11, u20, u21 = p
    # fit limb darkening for primary star
    if len(p) == 16:
        period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11 = p

    # to get in log(age) like the interpolation needs
    age = np.log10(age * 1e9)

    # calculate the chi-square from the magnitudes
    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle
    mags = isointerp(M1,FeH,age,isobundle)
    # take off the radius, logg, teff measurements
    mags = mags[:-3]

    #mag_model=5d0*alog10(distance/10d0)+alam_max*(1d0-exp(-distance*sin(10.2869/180d0*!dpi)/120d0))+input[18:34]
    magmodel = 5. * np.log10(dist / 10.) + alammult * maglam * (1. - np.exp(-dist * np.sin(10.2869 * np.pi / 180.) / height)) + mags

    #chi_square=chi_square+total((mag_model[inz]-mag_obs[inz])^2/(sig_mag[inz]^2+input[nparam-1]^2)) + total(alog(sig_mag[inz]^2+input[nparam-1]^2))
    # regular chi-square plus a penalty for large systematic error
    totchisq += np.sum((magmodel - magobs)**2. / (magerr**2. + syserr**2.)) + np.sum(np.log(magerr**2. + syserr**2.))

    # 10% error on Kalirai predicted/final mass relationship for WDs
    predM2 = 0.109 * M2init + 0.394
    totchisq += (M2 - predM2)**2./(0.1 * predM2)**2.

    # 15pc uncertainty in dust scale height (Jones, West, & Foster 2011)
    totchisq += (height - 119.)**2. / (15**2.)

    # 3.5% uncertainty in A_lambda max # From IRSA uncertainty on E(B-V)=0.0946+-0.0033
    totchisq += (alammult - 1.)**2. / (0.035**2.)

    # Add constraint that G dwarf age should be greater than or equal to spin-down age of 0.89+-0.15 Gyr:
    age = (10.**age) / 1e9
    if age < 0.89:
        totchisq += (age - 0.89)**2. / (0.15**2.)

    # TIP: remember that I added in this metallicity prior!
    # totchisq += (FeH - 0.)**2. / (0.3**2.)

    # if we're minimizing chi-square instead of maximizing likelihood
    if minimize:
        return totchisq
    # log likelihood is just -chisq/2
    return -totchisq/2.

def logprob(p,t,f,ferr,cuts,crowding,npert,isobundle,minimize=False):
    """
    Get the log probability of the data given the priors and the model.
    See loglikeli for the input parameters.

    Also requires npert (the number of subsamples to divide a cadence into).

    Returns
    -------
    prob : float
        Log likelihood of the model given the data and priors, up to a
        constant.
    """
    lp = logprior(p,isobundle)
    if not np.isfinite(lp):
        # minimization routines don't handle infinities very well, so
        # just penalize impossible parameter space
        if minimize:
            return 100000.
            #return np.inf
        return -np.inf
    if minimize:
        return -lp + loglikeli(p,t,f,ferr,cuts,crowding,isobundle,minimize = minimize, npert = npert)
    return lp + loglikeli(p,t,f,ferr,cuts,crowding,isobundle,minimize = minimize, npert = npert)











good = np.isfinite(ferr)

# just define segments of data as any data gap more than 4 days
edges = np.where(np.abs(t[1:] - t[:-1]) > 4.)[0] + 1
cuts = np.zeros(len(t)).astype(np.int)
cuts[edges] = 1
cuts = np.cumsum(cuts)

# make sure these functions are working
print loglikeli(p,t,f,ferr,cuts,crowding,isobundle,npert=subsample,minimize=True)
print logprob(p,t,f,ferr,cuts,crowding,subsample,isobundle,minimize=True)
print loglikeli(p,t,f,ferr,cuts,crowding,isobundle,npert=subsample)
print logprob(p,t,f,ferr,cuts,crowding,subsample,isobundle)

if domcmc:
    ndim = len(p)
    nwalkers = 50
    # set up the walkers in a ball near the optimal solution
    startlocs = [p + initrange(p)*np.random.randn(ndim)*0.6 for i in np.arange(nwalkers)]

    # set up the MCMC code
    sampler = emcee.EnsembleSampler(nwalkers,ndim,logprob,args=(t,f,ferr,cuts,crowding,subsample,isobundle))

    # clear the file
    ofile = open(outfile,'w')
    ofile.close()

    # run the MCMC, recording parameters for every walker at every step
    for result in sampler.sample(startlocs,iterations=niter,storechain=saveresults,thin=thin):
        position = result[0]
        iternum = sampler.iterations
        ofile = open(outfile,'a')
        for k in np.arange(position.shape[0]):
            ofile.write('{0} {1} {2} {3}\n'.format(iternum,k,str(result[1][k])," ".join([str(x) for x in position[k]])))
        ofile.close()
        print iternum

    # if we want to save the sampler for later inspection (autocorrelation, etc)
    if saveresults:
        output = open(picklefile,'wb')
        cPickle.dump(sampler,output)
        output.close()

# try to find an optimal solution
if findfit:
    #result = opt.fmin(logprob,p,args=(t,f,ferr,cuts,crowding,))
    result2 = opt.minimize(logprob,p,args=(t,f,ferr,cuts,crowding,subsample,isobundle,True),method='TNC',options = {'maxiter':1000, 'disp':True}) # TNC works / BFGS
    result = result2['x']
    print logprob(result,t,f,ferr,cuts,crowding,subsample,isobundle,minimize=True)
    print 'Fit:'
    print result
    #print result2
    resfullmod = loglikeli(result,t,f,ferr,cuts,crowding,isobundle,npert=subsample,retmodel=True)
    plt.plot(t[good],resfullmod[good],'r',lw=2,label='New Model')
    p = result

# some diagnostic plots
plt.figure(2)
plt.clf()
plt.plot(t[good],f[good],'b',lw=2)

print loglikeli(p,t,f,ferr,cuts,crowding,isobundle,npert=subsample)
print 'Reduced chi-square: ', loglikeli(p,t,f,ferr,cuts,crowding,isobundle,npert=subsample,minimize=True)/(len(t[good]) + len(magobs) - len(p)-1)
print logprob(p,t,f,ferr,cuts,crowding,subsample,isobundle)


pfullmod = loglikeli(p,t,f,ferr,cuts,crowding,isobundle,retmodel=True,npert=500)
polymodel = loglikeli(p,t,f,ferr,cuts,crowding,isobundle,retpoly=True,npert=500)
indchis = loglikeli(p,t,f,ferr,cuts,crowding,isobundle,indchi=True,npert=500)
#plt.plot(t[good],pfullmod[good])


plt.plot(t[good],polymodel[good],label='Polynomial Only')
plt.plot(t[good],pfullmod[good],label='Starting model')
plt.legend()

# fix limb darkening
if len(p) == 14:
    period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult = p
    u20 = 0.
    u21 = 0.
# fit limb darkening for both stars
if len(p) == 18:
    period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11, u20, u21 = p
# fit limb darkening for primary star
if len(p) == 16:
    period, ttran, ecosw, esinw, b, M2init, M2, M1, FeH, age, dist, syserr, height, alammult, u10, u11 = p
    u20 = 0.
    u21 = 0.

# to get in log(age) like the interpolation needs
age = np.log10(age * 1e9)

plt.figure(3)
plt.clf()


# the old version
cuts = np.where(np.abs(t[1:] - t[:-1]) > 4.)[0] + 1
cuts = np.concatenate(([0],cuts,[len(t)]))

#model = model * crowding[ii] + 1. - crowding[ii]
#model = (modelo + crowding - 1.) / crowding
#torig[cuts[ii]:cuts[ii+1]]
fullmodel = np.ones(len(f))
for ii in np.arange(len(cuts)-1):
    fullmodel[cuts[ii]:cuts[ii+1]] = (((pfullmod/polymodel) + crowding[ii] - 1.) / crowding[ii])[cuts[ii]:cuts[ii+1]]
rescale = fullmodel / (pfullmod/polymodel)


gs = gridspec.GridSpec(2,2,height_ratios=[2,1],wspace=0.03)
gs.update(hspace = 0.0)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

maxresid = 0.
miny = 5.
maxy = 0.
minx = np.array([10000.,10000.])
maxx = np.array([-10000.,-10000.])

tocc = np.array([])
focc = np.array([])
ferrocc = np.array([])
resocc = np.array([])
tpul = np.array([])
fpul = np.array([])
ferrpul = np.array([])
respul = np.array([])
qualpul = np.array([])
qualocc = np.array([])

KIC = ek.koitokic(3278)
usepdc = False
time,flux,fluxerr,cad,quart,qual,segnum = ek.preparelc(KIC,usepdc=usepdc)
# quality flag of each cadence
quals = np.zeros(len(t),dtype=int)
quarts = np.zeros(len(t))
for ii in np.arange(len(t)):
    best = np.abs((t[ii] - time)).argmin()
    quals[ii] = qual[best]
    quarts[ii] = quart[best]

eventquarts = np.zeros(len(cuts)-1)
for ii in np.arange(len(cuts)-1):
    used = np.arange(cuts[ii],cuts[ii+1])
    igood = np.isfinite(ferr[used])
    if ii % 2:
        tocc = np.concatenate((tocc,t[used][igood]%period))
        focc = np.concatenate((focc,f[used][igood]*rescale[used][igood]/polymodel[used][igood]))
        ferrocc = np.concatenate((ferrocc,ferr[used][igood]*rescale[used][igood]/polymodel[used][igood]))
        resocc = np.concatenate((resocc,f[used][igood]-pfullmod[used][igood]))
        qualocc = np.concatenate((qualocc,quals[used][igood]))
    else:
        tpul = np.concatenate((tpul,t[used][igood]%period))
        fpul = np.concatenate((fpul,f[used][igood]*rescale[used][igood]/polymodel[used][igood]))
        ferrpul = np.concatenate((ferrpul,ferr[used][igood]*rescale[used][igood]/polymodel[used][igood]))
        respul = np.concatenate((respul,f[used][igood]-pfullmod[used][igood]))
        qualpul = np.concatenate((qualpul,quals[used][igood]))
    eventquarts[ii] = scipy.stats.mstats.mode(quarts[used][igood])[0][0]

npts = 42
# the subtraction is so the bins don't straddle ingress/egress
tbinpul = np.linspace(min(tpul),max(tpul),npts) - 0.005
bwidpul = tbinpul[1]-tbinpul[0]
digits = np.digitize(tpul,tbinpul)
# get the binned light curve
fbinpul = np.array([np.median(fpul[digits == foo]) for foo in range(1,len(tbinpul))])
resbinpul = np.array([np.median(respul[digits == foo]) for foo in range(1,len(tbinpul))])
errbinpul = np.array([np.std(fpul[digits == foo])/np.sqrt(len(fpul[digits == foo])) for foo in range(1,len(tbinpul))])
tbinpul = tbinpul[1:] - bwidpul/2.
#ax0.scatter(tbinpul,fbinpul,c='r',zorder=3,s=40)
ax0.errorbar(tbinpul,fbinpul,yerr=errbinpul,ls='none',color='#dd0000',marker='o',mew=0,zorder=3,ms=np.sqrt(50),elinewidth=4,capthick=0,capsize=0)
ax2.errorbar(tbinpul,resbinpul,yerr=errbinpul,ls='none',color='#dd0000',marker='o',mew=0,zorder=3,ms=np.sqrt(50),elinewidth=4,capthick=0,capsize=0)
# just to see where the bin edges are exactly
#for ii in np.arange(len(tbinpul)):
    #ax0.plot([tbinpul[ii]-bwidpul/2.,tbinpul[ii]-bwidpul/2.],[0,2],c='r')

#ax2.scatter(tbinpul,resbinpul,c='r',zorder=3,s=40)
#binpulmod = loglikeli(p,tbinpul,fbinpul,np.ones(len(fbinpul))*np.median(ferr),cuts,crowding,isobundle,retmodel=True,npert=500)
#ax0.scatter(tbinpul,binpulmod,c='g',zorder=3,s=40)
#ax2.scatter(tbinpul,fbinpul-binpulmod,c='r',zorder=3,s=40)

# the subtraction is so the bins don't straddle ingress/egress
tbinocc = np.linspace(min(tocc),max(tocc),npts) - 0.007
bwidocc = tbinocc[1]-tbinocc[0]
digits = np.digitize(tocc,tbinocc)
# get the binned light curve
fbinocc = np.array([np.median(focc[digits == foo]) for foo in range(1,len(tbinocc))])
resbinocc = np.array([np.median(resocc[digits == foo]) for foo in range(1,len(tbinocc))])
errbinocc = np.array([np.std(focc[digits == foo])/np.sqrt(len(focc[digits == foo])) for foo in range(1,len(tbinocc))])
tbinocc = tbinocc[1:] - bwidocc/2.
ax1.errorbar(tbinocc,fbinocc,yerr=errbinocc,ls='none',color='#dd0000',mew=0,marker='o',zorder=3,ms=np.sqrt(50),elinewidth=4,capthick=0,capsize=0)
#ax1.scatter(tbinocc,fbinocc,c='r',zorder=3,s=40)
#ax3.scatter(tbinocc,resbinocc,c='r',zorder=3,s=40)
ax3.errorbar(tbinocc,resbinocc,yerr=errbinocc,ls='none',color='#dd0000',mew=0,marker='o',zorder=3,ms=np.sqrt(50),elinewidth=4,capthick=0,capsize=0)


# just to see where the bin edges are exactly
#for ii in np.arange(len(tbinocc)):
    #ax1.plot([tbinocc[ii]-bwidocc/2.,tbinocc[ii]-bwidocc/2.],[0,2],c='r')



"""
# show the bad quality flags and print their values
ax0.scatter(tpul[qualpul != 0],fpul[qualpul != 0],s=18,zorder=4,c='r',lw=0)
ax1.scatter(tocc[qualocc != 0],focc[qualocc != 0],s=18,zorder=4,c='r',lw=0)
for ii in np.arange(len(tpul)):
    if qualpul[ii] != 0:
        ax0.text(tpul[ii],fpul[ii],str(qualpul[ii]),bbox = {'fc':'r','ec':'none','pad':0})
for ii in np.arange(len(tocc)):
    if qualocc[ii] != 0:
        ax1.text(tocc[ii],focc[ii],str(qualocc[ii]),bbox = {'fc':'r','ec':'none','pad':0})
"""
ax0.scatter(tpul,fpul,s=3,zorder=2,c='k',lw=0)
ax1.scatter(tocc,focc,s=3,zorder=2,c='k',lw=0)
ax0.set_xlim(tpul.min(),tpul.max())
ax1.set_xlim(tocc.min(),tocc.max())
maxflux = np.array([fpul.max(),focc.max()]).max()
minflux = np.array([focc.min(),fpul.min()]).min()
ax0.set_ylim(minflux,maxflux)
ax1.set_ylim(minflux,maxflux)
ax0.set_ylim(0.9982,1.0018)
ax1.set_ylim(0.9982,1.0018)
ax0.ticklabel_format(useOffset=False)
ax1.set_yticklabels([])
ax3.set_yticklabels([])
ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax0.set_ylabel('Relative flux',fontsize=24)
ax0.tick_params(labelsize=18,width=2,length=5)
ax1.tick_params(labelsize=18,width=2,length=5)
ax2.tick_params(labelsize=18,width=2,length=5)
ax3.tick_params(labelsize=18,width=2,length=5)

ax2.scatter(tpul,respul,s=3,zorder=2,c='k',lw=0)
ax3.scatter(tocc,resocc,s=3,zorder=2,c='k',lw=0)
ax2.set_ylabel('Residuals',fontsize=24)
ax2.set_xlabel('BJD - 2455000',fontsize=24)
ax3.set_xlabel('BJD - 2455000',fontsize=24)
ax2.set_xlim(tpul.min(),tpul.max())
ax3.set_xlim(tocc.min(),tocc.max())
maxresid = np.array([np.abs(respul).max(),np.abs(resocc).max()]).max()
maxresid = 0.0015
ax2.set_ylim(-maxresid, maxresid)
ax3.set_ylim(-maxresid, maxresid)

# this plots the model convolved at the Kepler cadence
goodmod = np.where(polymodel > 0)[0]
tmod = t[goodmod]
pmod = pfullmod[goodmod]*rescale[goodmod]/polymodel[goodmod]
order = np.argsort(tmod % period)
ax0.plot(tmod[order] % period,pmod[order],c='#666666',lw=4,zorder=1)
ax1.plot(tmod[order] % period,pmod[order],c='#666666',lw=4,zorder=1)
ax2.plot(tmod[order] % period,np.zeros(len(tmod)),c='#666666',lw=4,zorder=1)
ax3.plot(tmod[order] % period,np.zeros(len(tmod)),c='#666666',lw=4,zorder=1)





inpul = np.where(pmod > 1.00002)[0]
inocc = np.where(pmod < 0.99996)[0]
flat = np.where((pmod < 1.00002) & (pmod > 0.99996))[0]
allresids = f[goodmod] - pfullmod[goodmod]
print np.std(allresids[flat]), np.std(allresids[inocc]), np.std(allresids[inpul])
plt.figure(7)
plt.clf()
plt.hist(allresids[flat],bins=350,alpha=0.5,facecolor='k',label='Out of Events')
plt.hist(allresids[inocc],bins=30,alpha=0.5,facecolor='r',label='Occultation')
plt.hist(allresids[inpul],bins=30,alpha=0.5,facecolor='g',label='Pulse')
plt.legend()
plt.xlabel('Residuals')

"""
# this plots an absolutely perfect model (before contamination stuff)
t2 = np.linspace(tpul.min(),tpul.max(),10000)
modelfine = light_curve_model(t2,p,isobundle,npert=50)
ax0.plot(t2,modelfine,c='#666666',lw=3,zorder=1)
ax2.plot(t2,np.zeros(len(t2)),c='#666666',lw=2,zorder=1)

t3 = np.linspace(tocc.min(),tocc.max(),10000)
modelfine2 = light_curve_model(t3,p,isobundle,npert=50)
ax1.plot(t3,modelfine2,c='#666666',lw=3,zorder=1)
ax3.plot(t3,np.zeros(len(t3)),c='#666666',lw=2,zorder=1)
#plt.plot(t2,light_curve_model(t2,result),'c')
"""


# plot the individual events
fig4 = plt.figure(4)
fig4.clf()

fig5 = plt.figure(5,figsize=(7,5))
fig5.clf()
f4ct = 1
f5ct = 1

for ii in np.arange(len(cuts)-1):
    used = np.arange(cuts[ii],cuts[ii+1])
    igood = np.isfinite(ferr[used])
    if ii % 2 and len(t[used][igood]) > 1:
        plt.figure(4)
        ax = fig4.add_subplot(4,4,f4ct)

        plt.scatter(t[used][igood], f[used][igood]/polymodel[used][igood],c='k',s=40,zorder=1)
        #plt.plot(t,pfullmod/polymodel,c='r')
        t3 = np.linspace(t[used][igood].min(),t[used][igood].max(),10000)
        modelfine3 = light_curve_model(t3,p,isobundle,npert=50)
        plt.plot(t3,modelfine3,c='r',lw=3,zorder=2)
        ax.set_ylim(0.9982,1.0015)
        ax.set_xlim(t[used][igood].min(),t[used][igood].max())
        ax.ticklabel_format(useOffset=False)
        if f4ct % 4 != 1:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=5,prune='both'))
        ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=5,prune='both'))
        ax.tick_params(labelsize=18,width=2,length=5)
        f4ct += 1

    elif len(t[used][igood]) > 1:
        plt.figure(5)
        ax = fig5.add_subplot(4,4,f5ct)

        plt.scatter(t[used][igood], f[used][igood]/polymodel[used][igood],c='k',s=40,zorder=1)
        #plt.plot(t,pfullmod/polymodel,c='r')
        t3 = np.linspace(t[used][igood].min(),t[used][igood].max(),10000)
        modelfine3 = light_curve_model(t3,p,isobundle,npert=50)
        plt.plot(t3,modelfine3,c='r',lw=3,zorder=2)
        ax.set_ylim(0.999,1.0018)
        ax.set_xlim(t[used][igood].min(),t[used][igood].max())
        ax.ticklabel_format(useOffset=False)
        if f5ct % 4 != 1:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=5,prune='both'))
        ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=5,prune='both'))
        ax.tick_params(labelsize=18,width=2,length=5)
        f5ct += 1

fig4.subplots_adjust(wspace=0.03)
fig5.subplots_adjust(wspace=0.03)

fig4.text(0.5,0.05,'BJD - 2455000',ha='center',va='center',fontsize=24)
fig5.text(0.5,0.05,'BJD - 2455000',ha='center',va='center',fontsize=24)
fig4.text(0.07,0.5,'Relative Flux',ha='center',va='center',rotation=90,fontsize=24)
fig5.text(0.07,0.5,'Relative Flux',ha='center',va='center',rotation=90,fontsize=24)


M2s = np.linspace(0.1,1.453,1000)


wdage = np.log10(10.**age - 10.**(msage(M2init,FeH,isobundle)))
wdmag = wdkpmag(M2,wdage,isobundle)

mags = isointerp(M1,FeH,age,isobundle)

R1 = mags[-3]
logg = mags[-2]
Teff = 10.**mags[-1]
if len(p) == 14:
	# get the limb darkening from the fit to Sing?
	u10 = 0.44657704  -0.00019632296 * (Teff-5500.) +   0.0069222222 * (logg-4.5) +    0.086473504 *FeH
	u11 = 0.22779778  - 0.00012819556 * (Teff-5500.) - 0.0045844444  * (logg-4.5)  -0.050554701 *FeH
u1 = np.array([u10,u11])
u2 = np.array([u20,u21])

magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle

F2F1 = 0.
if np.isfinite(wdmag):
	# get the Kp magnitude of the main star
	gind = np.where(magname == 'g')[0][0]
	rind = np.where(magname == 'r')[0][0]
	iind = np.where(magname == 'i')[0][0]
	if mags[gind] - mags[rind] <= 0.3:
		kpmag1 = 0.25 * mags[gind] + 0.75 * mags[rind]
	else:
		kpmag1 = 0.3 * mags[gind] + 0.7 * mags[iind]

F2F1 = 10.**((wdmag - kpmag1)/(-2.5))
if not np.isfinite(F2F1):
	F2F1 = 0.

# reconvert into more useful orbital elements
e = np.sqrt(ecosw**2. + esinw**2.)
omega = np.arctan2(esinw,ecosw)
a = ((period * 86400.)**2. * 6.67e-11 * (M1 + M2) * 1.988e30 / (4.*np.pi**2.))**(1./3) # in m
a = a / (6.955e8 * R1) # in radii of the first star
# Simple conversion
inc = np.arccos(b/a)
# Compute the size of the WD using the Nauenberg relation:
MCh = 1.454
# in Solar radii
R2 = .0108*np.sqrt((MCh/M2)**(2./3.)-(M2/MCh)**(2./3.))
R2s = .0108*np.sqrt((MCh/M2s)**(2./3.)-(M2s/MCh)**(2./3.))
rrat = R2 / R1
rrats = R2s / R1

n = 2. * np.pi / period

# Sudarsky 2005 Eq. 9 to convert between center of transit and pericenter passage (tau)
edif = 1.-e**2.
fcen = np.pi/2. - omega
tau = ttran + np.sqrt(edif)*period / (2.*np.pi) * ( e*np.sin(fcen)/(1.+e*np.cos(fcen)) - 2./np.sqrt(edif) * np.arctan(np.sqrt(edif)*np.tan(fcen/2.)/(1.+e)))

# define the mean anomaly
M = (n * (ttran - tau)) % (2. * np.pi)
E = kepler_problem(M,e)

# solve for f
tanf2 = np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.)
fanom = (np.arctan(tanf2)*2.) % (2. * np.pi)

r = a * (1. - e**2.) / (1. + e * np.cos(fanom))
# projected distance between the stars (in the same units as a)
projdist = r * np.sqrt(1. - np.sin(omega + fanom)**2. * np.sin(inc)**2.)

# positive z means body 2 is in front (transit)
Z = r * np.sin(omega + fanom) * np.sin(inc)
# get the lens depth given this separation at transit
# 1.6984903e-5 gives 2*Einstein radius^2 = 8GMZ/(c^2 R^2) with M, Z, R all
# scaled to solar values

lensdeps = 1.6984903e-5 * M2s * np.abs(Z) / R1 - rrats**2.
lensonly = 1.6984903e-5 * M2s * np.abs(Z) / R1
tranonly =  -rrats**2.

Merrs = np.array([0.579,0.681])
Rerrs = .0108*np.sqrt((MCh/Merrs)**(2./3.)-(Merrs/MCh)**(2./3.))
rraterrs = Rerrs / R1
lensdeperrs = 1.6984903e-5 * Merrs * np.abs(Z) / R1 - rraterrs**2.
#lensdeperrs2 = np.array([0.00097,0.00107])
lensdeperrs2 = np.array([0.000954,0.001056])

plt.figure(6)
plt.clf()
ax = plt.subplot(111)
plt.plot(M2s,lensonly,ls='-.',c='k',lw=4,zorder=3)
plt.plot(M2s,lensdeps,c='k',lw=4,zorder=3)
plt.plot(M2s,tranonly,ls='--',c='k',lw=4,zorder=3)
plt.plot([-1,4],[0.,0.],'k',lw=2,zorder=2)
plt.xlim(0.,M2s.max())

# shaded region for our error bars
ax.fill_between(Merrs,[ax.get_ylim()[0],ax.get_ylim()[0]],[ax.get_ylim()[1],ax.get_ylim()[1]],facecolor='#8281f7',alpha=1,edgecolor='none',zorder=1)
#ax.fill_between([ax.get_xlim()[0],ax.get_xlim()[1]],lensdeperrs[0],lensdeperrs[1],facecolor='g',alpha=0.3)
ax.fill_between([ax.get_xlim()[0],ax.get_xlim()[1]],lensdeperrs2[0],lensdeperrs2[1],facecolor='#8281f7',alpha=1,edgecolor='none',zorder=1)

plt.plot([ax.get_xlim()[0],ax.get_xlim()[1]],[lensdeperrs2[0],lensdeperrs2[0]],lw=1,zorder=2,color='k')
plt.plot([ax.get_xlim()[0],ax.get_xlim()[1]],[lensdeperrs2[1],lensdeperrs2[1]],lw=1,zorder=2,color='k')
plt.plot([Merrs[0],Merrs[0]],[ax.get_ylim()[0],ax.get_ylim()[1]], zorder=2,lw=1,color='k')
plt.plot([Merrs[1],Merrs[1]],[ax.get_ylim()[0],ax.get_ylim()[1]], zorder=2,lw=1,color='k')

ax.tick_params(labelsize=16,width=2,length=5)
plt.xlabel(r'$M_{WD} (M_\odot)$',fontsize=20)
plt.ylabel('Magnification - 1',fontsize=20)
