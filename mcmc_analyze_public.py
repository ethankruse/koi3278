"""
Analyze the results of MCMC analysis.
"""

import ekruse as ek
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from glob import glob
import cPickle
from scipy import interpolate


# isochrone files
padsdssfiles = '/astro/users/eakruse/microlens/padova/*sdss'
padwisefiles = '/astro/users/eakruse/microlens/padova/*wise'



#infile = './chain_mr_iso.txt'
infile = './chain_agelim.txt'
infile = './chain_agelim_fullmetal.txt'
infile = './chain_agelim_fullmetal_crowding_thin.txt'
infile = './chain_agelim_fullmetal_crowding.txt'
infile = './chain_final_isochrones.txt'
#infile = './chain_agelim_fullmetal_crowding_metalprior.txt'
#infile = './chain_mr_reduced.txt'

fitlimb = False

picklefile = None

# output the results to a TeX file
texout = None
#texout = './MCMC_fit_iso.tex'
#texout = './MCMC_fit_agelim.tex'
#texout = './MCMC_fit_agelim_fullmetal.tex'
#texout = './MCMC_fit_agelim_fullmetal_crowding.tex'
texout = './MCMC_fit_final_isochrones.tex'
#texout = './MCMC_fit_agelim_fullmetal_crowding_metalprior.tex'

# whether or not to evaluate all the isochrones
gettemps = True

# iteration where burn-in stops
burnin = 20000

# after the burn in, only use every thin amount for speed
nthin = 1

maketriangle = False

# current order
if fitlimb:
    labels = ['$P$ (d)','$t_0$ (d)','$e\cos\omega$','$e\sin\omega$','$b_0$','$M_{2,init}$','$M_2$','$M_1$','[Fe/H]$_1$','$t_1$ (Gyr)','D (pc)','$\sigma_{sys}$','h (pc)','$A_\lambda$','$u_{S1,1}$','$u_{S1,2}$']
else:
    labels = ['$P$ (d)','$t_0$ (d)','$e\cos\omega$','$e\sin\omega$','$b_0$','$M_{2,init}$','$M_2$','$M_1$','[Fe/H]$_1$','$t_1$ (Gyr)','D (pc)','$\sigma_{sys}$','h (pc)','$A_\lambda$']


wdfiles = './wdmodels/Table_Mass*'
# what's in the isochrone and what its index is
wdinds = {'g':13,'r':14,'i':15,'teff':0,'logg':1,'age':26}


# =========================================================================== #

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


nparams = len(labels)

x = np.loadtxt(infile)
print 'File loaded'

iteration = x[:,0]
walkers = x[:,1]
uwalkers = np.unique(walkers)
loglike = x[:,2]

x = x[:,3:]

thin = np.arange(0,iteration.max(),nthin)
good = np.in1d(iteration,thin)

x = x[good,:]
iteration = iteration[good]
walkers = walkers[good]
loglike = loglike[good]


plt.figure(1)
plt.clf()
#period, ttran, e, u10, u11, u20, u21, a, rrat, inc, omega, F2F1, lensdep = p
for ii in np.arange(nparams+1):

    plt.subplot(np.ceil((nparams+1)/3.),3,ii+1)
    for jj in uwalkers:
        this = np.where(walkers == jj)[0]
        if ii < nparams:
            if len(iteration[this]) > 25000:
                plt.plot(iteration[this][::10],x[this,ii].reshape((-1,))[::10])
            else:
                plt.plot(iteration[this][::10],x[this,ii].reshape((-1,))[::10])

        else:
            if len(iteration[this]) > 25000:
                plt.plot(iteration[this][::10],loglike[this][::10])
            else:
                plt.plot(iteration[this],loglike[this])
    #if ii < nparams:
        #plt.plot([iteration.min(),iteration.max()],[p[ii],p[ii]],lw=2)

    plt.plot([burnin,burnin],plt.ylim(),lw=2)
    if ii < nparams:
        plt.ylabel(labels[ii])
    else:
        plt.ylabel('Log Likelihood')

    #plt.xlabel('Iterations')
#plt.tight_layout()

pastburn = np.where(iteration > burnin)[0]
#pastburn = np.where((iteration > burnin) & (loglike > -1772.))[0]
iteration = iteration[pastburn]
walkers = walkers[pastburn]
loglike = loglike[pastburn]
x = x[pastburn,:]
lsort = np.argsort(loglike)
lsort = lsort[::-1]
iteration = iteration[lsort]
walkers = walkers[lsort]
loglike = loglike[lsort]
x = x[lsort,:]


x[:,4] = np.abs(x[:,4])

if maketriangle:
    #fig = triangle.corner(x,labels=labels)
    plt.figure(2)
    plt.clf()

    maxes = np.zeros(len(x[0,:])) - 9e9
    mins = np.zeros(len(x[0,:])) + 9e9
    nbins = 50
    for jj in np.arange(len(x[0,:])):
        for kk in np.arange(len(x[0,:])):
            if kk < jj:
                ax = plt.subplot(len(x[0,:]),len(x[0,:]),jj * len(x[0,:]) + kk + 1)

                sigmas = np.array([0.9544997,0.6826895])
                sigmas = np.array([0.9973002,0.9544997,0.6826895])

                hist2d, xedge, yedge = np.histogram2d(x[:,jj],x[:,kk],bins = [nbins,nbins], normed=False)
                hist2d /= len(x[:,jj])
                flat = hist2d.flatten()
                fargs = flat.argsort()[::-1]
                flat = flat[fargs]
                cums = np.cumsum(flat)
                #levels = sigmas * 0.
                levels = []

                for ii in np.arange(len(sigmas)):
                        above = np.where(cums > sigmas[ii])[0][0]
                        #levels[ii] = flat[above]
                        levels.append(flat[above])
                levels.append(1.)

                above = np.where(hist2d > levels[0])
                thismin = xedge[above[0]].min()
                if thismin < mins[jj]:
                    mins[jj] = thismin
                thismax = xedge[above[0]].max()
                if thismax > maxes[jj]:
                    maxes[jj] = thismax
                thismin = yedge[above[1]].min()
                if thismin < mins[kk]:
                    mins[kk] = thismin
                thismax = yedge[above[1]].max()
                if thismax > maxes[kk]:
                    maxes[kk] = thismax

                plt.contourf(yedge[1:]-np.diff(yedge)/2.,xedge[1:]-np.diff(xedge)/2.,hist2d,levels=levels,colors=('k','#444444','#888888'))

                #plt.xlim(x[:,kk].min(),x[:,kk].max())
                #plt.ylim(x[:,jj].min(),x[:,jj].max())
            if jj == kk:
                ax = plt.subplot(len(x[0,:]),len(x[0,:]),jj * len(x[0,:]) + kk + 1)
                plt.hist(x[:,jj],bins=nbins,facecolor='k')
                #plt.xlim(x[:,kk].min(),x[:,kk].max())

    diffs = maxes - mins
    mins -= 0.1 * diffs
    maxes += 0.1 * diffs

    for jj in np.arange(len(x[0,:])):
        for kk in np.arange(len(x[0,:])):
            if kk < jj or jj == kk:
                ax = plt.subplot(len(x[0,:]),len(x[0,:]),jj * len(x[0,:]) + kk + 1)

                if kk < jj:
                    ax.set_ylim(mins[jj],maxes[jj])
                ax.set_xlim(mins[kk],maxes[kk])

                ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=4,prune='both'))
                #ax.ticklabel_format(useOffset=False)
                if kk != 0 or jj == 0:
                    ax.set_yticklabels([])
                else:
                    plt.ylabel(labels[jj])
                    locs,labs = plt.yticks()
                    plt.setp(labs,rotation=0,va='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.yaxis.set_major_formatter(yformatter)

                ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=4,prune='both'))
                if jj != len(x[0,:])-1:
                    ax.set_xticklabels([])
                else:
                    plt.xlabel(labels[kk])
                    locs,labs = plt.xticks()
                    plt.setp(labs,rotation=90,ha='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.xaxis.set_major_formatter(yformatter)


    plt.subplots_adjust(hspace=0.0,wspace=0.0)



best = np.median(x,axis=0)
devs = np.std(x,axis=0)


# ============================================================================ #
# isochrone loading section
if gettemps and texout is not None:
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
# ============================================================================ #

if texout is not None:

    e = (np.sqrt(x[:,2]**2. + x[:,3]**2.)).reshape((len(x[:,0]),1))
    x = np.concatenate((x,e),axis=1)
    labels.append('$e$')
    omega = np.arctan2(x[:,3],x[:,2]).reshape((len(x[:,0]),1)) * 180./np.pi
    x = np.concatenate((x,omega),axis=1)
    labels.append('$\omega$ (deg)')

    if gettemps:
        FeH = x[:,8]
        # convert to log(age) for the isochrone
        age = np.log10(x[:,9] * 1e9)
        M = x[:,7]

        results = np.zeros((len(FeH),len(maginds)))

        for ii in np.arange(len(FeH)):
            results[ii,:] = isointerp(M[ii],FeH[ii],age[ii],isobundle)

        Teff = (10.**results[:,-1]).reshape((len(x[:,0]),1))
        x = np.concatenate((x,Teff),axis=1)
        labels.append('$T_{eff,1}$ (K)')

        logg = (results[:,-2]).reshape((len(x[:,0]),1))
        x = np.concatenate((x,logg),axis=1)
        labels.append('log(g)')

        R1 = (results[:,-3]).reshape((len(x[:,0]),1))
        x = np.concatenate((x,R1),axis=1)
        labels.append('$R_1$')

        a = ((x[:,0] * 86400.)**2. * 6.67e-11 * (x[:,6] + x[:,7]) * 1.988e30 / (4.*np.pi**2.))**(1./3) # in m
        aau = a * 6.685e-12 # in AU
        aau = aau.reshape((len(x[:,0]),1)) # in AU
        x = np.concatenate((x,aau),axis=1)
        labels.append('$a$ (AU)')


        a = (a / (6.955e8 * x[:,-2])).reshape((len(x[:,0]),1)) # in radii of the first star
        x = np.concatenate((x,a),axis=1)
        aind = len(labels)
        labels.append('$a/R_1$')

        # Eq. 7 of Winn chapter from Exoplanets
        #inc = np.arccos(b/a * ((1. + esinw)/(1.-e**2.)))
        inc = (np.arccos(x[:,4]/x[:,aind])).reshape((len(x[:,0]),1)) * 180./np.pi
        x = np.concatenate((x,inc),axis=1)
        labels.append('$i$ (deg)')

        results = results[:,:-3]
        magname = magname[:-3]
        x = np.concatenate((x,results),axis=1)
        for ii in magname:
            labels.append(ii)

        # predicted Kp magnitude of the MS star
        kpmag = np.zeros(len(results[:,0]))
        blue = results[:,0] - results[:,1] <= 0.3
        kpmag[blue] = 0.25 * results[blue,0] + 0.75 * results[blue,1]
        kpmag[~blue] = 0.3 * results[~blue,0] + 0.7 * results[~blue,2]

        wdage = np.zeros(len(age))
        msages = np.zeros(len(age))
        wdmag = np.zeros(len(age))
        F2F1 = np.zeros(len(age))
        for ii in np.arange(len(age)):
            msages[ii] = msage(x[ii,5],x[ii,8],isobundle)
            wdage[ii] = np.log10(10.**age[ii] - 10.**(msages[ii]))
            wdmag[ii] = wdkpmag(x[ii,6],wdage[ii],isobundle)
            F2F1[ii] = 10.**((wdmag[ii] - kpmag[ii])/(-2.5))

        F2F1 = F2F1.reshape((len(x[:,0]),1))
        x = np.concatenate((x,F2F1),axis=1)
        labels.append('$F_2/F_1$')

        wdpreds = np.zeros((len(wdage),2))
        wdpreds[:,0] = x[:,6]
        wdpreds[:,1] = wdage

        wdtemp = interpolate.griddata(wdmodels[:,0:2],wdmods[:,wdinds['teff']],wdpreds)
        wdtemp = wdtemp.reshape((len(x[:,0]),1))
        x = np.concatenate((x,wdtemp),axis=1)
        labels.append('$T_{eff,2}$ (K)')

        # from Seager's Exoplanets Eq. 2.13
        K = 29.775 / np.sqrt(1. - e[:,0]**2.) * x[:,6] * np.sin(inc[:,0] * np.pi/180.) / np.sqrt(x[:,6] + x[:,7]) / np.sqrt(a[:,0] * R1[:,0] * 0.004649) # km/s
        K = K.reshape((len(x[:,0]),1))
        x = np.concatenate((x,K),axis=1)
        labels.append('$K_1$ (km/s)')

        parallax = 1000./x[:,10]
        parallax = parallax.reshape((len(x[:,0]),1))
        x = np.concatenate((x,parallax),axis=1)
        labels.append('$\pi$ (mas)')

        a1 = x[:,6] / (x[:,6] + x[:,7]) * a[:,0] * R1[:,0] * 0.004649 # in AU
        reflex = 1000. * a1 / x[:,10] # in mas (half-amplitude)
        reflex = reflex.reshape((len(x[:,0]),1))
        x = np.concatenate((x,reflex),axis=1)
        labels.append(r'$\alpha_1$ (mas)')


        u1 = 0.44657704  -0.00019632296 * (Teff[:,0]-5500.) +   0.0069222222 * (logg[:,0]-4.5) +    0.086473504 *FeH
        u2 = 0.22779778  - 0.00012819556 * (Teff[:,0]-5500.) - 0.0045844444  * (logg[:,0]-4.5)  -0.050554701 *FeH
        u1 = u1.reshape((len(x[:,0]),1))
        x = np.concatenate((x,u1),axis=1)
        labels.append('$u_1$')
        u2 = u2.reshape((len(x[:,0]),1))
        x = np.concatenate((x,u2),axis=1)
        labels.append('$u_2$')

        MCh = 1.454
        # in Solar radii
        R2 = .0108*np.sqrt((MCh/x[:,6])**(2./3.)-(x[:,6]/MCh)**(2./3.))
        R2 = R2.reshape((len(x[:,0]),1))
        x = np.concatenate((x,R2),axis=1)
        labels.append('$R_2$')

        msages = (10.**msages).reshape((len(x[:,0]),1)) / 1e9
        wdage = (10.**wdage).reshape((len(x[:,0]),1)) / 1e9


        n = 2. * np.pi / x[:,0]

        # Sudarsky 2005 Eq. 9 to convert between center of transit and pericenter passage (tau)
        edif = 1.-e[:,0]**2.
        fcen = np.pi/2. - omega[:,0] * np.pi/180.
        tau = x[:,1] + np.sqrt(edif)*x[:,0] / (2.*np.pi) * ( e[:,0]*np.sin(fcen)/(1.+e[:,0]*np.cos(fcen)) - 2./np.sqrt(edif) * np.arctan(np.sqrt(edif)*np.tan(fcen/2.)/(1.+e[:,0])))

        # define the mean anomaly
        M = (n * (x[:,1] - tau)) % (2. * np.pi)
        E = kepler_problem(M,e[:,0])


        # solve for f
        tanf2 = np.sqrt((1.+e[:,0])/(1.-e[:,0])) * np.tan(E/2.)
        f = (np.arctan(tanf2)*2.) % (2. * np.pi)

        r = a[:,0] * (1. - e[:,0]**2.) / (1. + e[:,0] * np.cos(f))
        # positive z means body 2 is in front (transit)
        Z = r * np.sin(omega[:,0]*np.pi/180. + f) * np.sin(inc[:,0]*np.pi/180.)
        # 1.6984903e-5 gives 2*Einstein radius^2/R1^2
        Rein = np.sqrt(1.6984903e-5 * x[:,6] * np.abs(Z) * R1[:,0] / 2.)

        Rein = Rein.reshape((len(x[:,0]),1))
        x = np.concatenate((x,Rein),axis=1)
        labels.append('$R_E$')

        x = np.concatenate((x,msages),axis=1)
        labels.append('WD MS Age (Gyr)')
        x = np.concatenate((x,wdage),axis=1)
        labels.append('WD Cooling Age (Gyr)')

        wdlum = (R2**2.) * ((wdtemp / 5777.)**4.)
        x = np.concatenate((x,wdlum),axis=1)
        labels.append('$L_{WD} (L_\odot)$')

        lensdeps = 1.6984903e-5 * x[:,6] * np.abs(Z) / R1[:,0] - (R2[:,0]/R1[:,0])**2.
        lensdeps = lensdeps.reshape((len(x[:,0]),1))
        x = np.concatenate((x,lensdeps),axis=1)
        labels.append('Magnification - 1')

    stds = [15.87,50.,84.13]
    neg1, med, plus1 = np.percentile(x,stds,axis=0)

    ofile = open(texout,'w')
    ofile.write('\\documentclass{article}\n\\begin{document}\n\n\\begin{tabular}{| c | c |}\n\\hline\n')

    # what decimal place the error bar is at
    sigfigslow = np.floor(np.log10(np.abs(plus1-med)))
    sigfigshigh = np.floor(np.log10(np.abs(med-neg1)))
    sigfigs = sigfigslow * 1
    # take the smallest of the two sides of the error bar
    lower = np.where(sigfigshigh < sigfigs)[0]
    sigfigs[lower] = sigfigshigh[lower]
    # go one digit farther
    sigfigs -= 1
    # no decimals if not necessary
    #sigfigs[sigfigs > 0] = 0
    sigfigs *= -1.
    sigfigs = sigfigs.astype(int)

    for ii in np.arange(len(labels)):
        if sigfigs[ii] >= 0:
            val = '%.'+str(sigfigs[ii])+'f'
        else:
            val = '%.'+str(0)+'f'
        ofile.write(labels[ii]+' & $'+ str(val % np.around(med[ii],decimals=sigfigs[ii])) + '^{+'+str(val % np.around(plus1[ii]-med[ii],decimals=sigfigs[ii]))+'}_{-' +str(val % np.around(med[ii]-neg1[ii],decimals=sigfigs[ii]))  + '}$ \\\\\n\\hline\n')

    ofile.write('\\end{tabular}\n\\end{document}')
    ofile.close()

if picklefile is not None:
    pfile = open(picklefile,'rb')
    pkl = cPickle.load(pfile)
    pfile.close()




