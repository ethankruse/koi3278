"""
Make a river plot of a known KOI or a found transit.

This is under the assumption that fill=True, so no cadences are missing in the
loaded light curve.

Basically a copy of the river_plot.py in the ~/circumbinary/ directory, but wanted
to tweak things slightly for the KOI-3278 paper (e.g. adjust titles).
"""
import ekruse as ek
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
import matplotlib.cm as cm
from scipy import signal as sgnl
from matplotlib.colors import LinearSegmentedColormap as LSC

KOI = 3278
#prefix = './koisRun2/KOI' + str(KOI)
prefix = '/astro/users/eakruse/microlens/KOI' + str(KOI)
#prefix = './microlens/KOI' + str(KOI)
#prefix = './microlens_fill/KOI' + str(KOI)
#prefix = './koisFitted/KOI' + str(KOI)
maskprefix = '/astro/users/eakruse/circumbinary/knownkois/KOI' + str(KOI)
maskprefix = '/astro/users/eakruse/circumbinary/koitransittimes/KOI' + str(KOI)
usetimes = True

KIC = ek.koitokic(KOI)

# which potential planet(s) to look at (starting at 1)
# can be a list too to compare several
resnum = [1]

# save the figure?
saveresult = False

# for poster figures
bigfont = True
# otherwise save as png
dopdf = False
# make it anonymous?
anonymous = False
# if so, start and end it a little early
fstart = 0.01
fend = 0.9

dosmooth = False

# plot the known (masked) KOI planets as well
usekoi = True


fixspsds = False
usepdc = False

#usetxt = './injections/InjectKIC'+str(KIC)+'.txt'
usetxt = None

koilist = ek.koilist

# how many transit durations to plot on either side
plotrange = 2.

# ===============================

# seahawks 1
cdict = { 'red': ((0,0,0),  \
                  (0.5,0.702,0.702),  \
                  (1.,0.37255,0)), \
          'green': ((0,0,0.1333),\
                    (0.5,0.7098,0.7098),\
                    (1.,0.7412,0)),\
          'blue': ((0,0,0.2588),\
                    (0.5,0.7216,0.7216),\
                    (1.,0.2706,0))\
          }

# seahawks 2
cdict = { 'red': [(0.,0.702,0.702),  \
                  (0.5,0,0),  \
                  (1.,0.37255,0)], \
          'green': [(0,0.7098,0.7098),\
                    (0.5,0.1333,0.1333),\
                    (1.,0.7412,0)],\
          'blue': [(0.,0.7216,0.7216),\
                    (0.5,0.2588,0.2588),\
                    (1.,0.2706,0)]\
          }

"""
# UW 1
cdict = { 'red': ((0.,0,0),  \
                  (0.5,0.2235,0.2235),  \
                  (1.,0.7804,0.7804)), \
          'green': ((0.,0.,0.),\
                    (0.5,0.15294,0.15294),\
                    (1.,0.600,0.600)),\
          'blue': ((0.,0.,0.),\
                    (0.5,0.3569,0.3569),\
                    (1.,0.,0.)),\
          }
"""
locs, reds, greens, blues = np.loadtxt('./cubehel4.txt',unpack=True)
rchans = []
gchans = []
bchans = []
for ii in np.arange(len(locs)):
    rchans.append((locs[ii],reds[ii],reds[ii]))
    gchans.append((locs[ii],greens[ii],greens[ii]))
    bchans.append((locs[ii],blues[ii],blues[ii]))
cdict = {'red':rchans, 'green':gchans, 'blue':bchans}



seahawks = LSC('Seahawks',cdict)

cmap = cm.winter

cmap = seahawks
cmap.set_bad(color='#ffffff',alpha=1)
cmap.set_bad(color='#cccccc',alpha=1)
#cmap = cm.cubehelix
#cmap = cm.CMRmap

if not anonymous:
    fstart = 0.
    fend = 1.
    anonoff = 0.

# get the light curve
time,flux,fluxerr,cad,quart,qual,segnum = ek.preparelc(KIC,cutjumps=fixspsds,usepdc=usepdc,usetxt = usetxt,fill=True)


try:
    len(resnum)
    resnum = np.array(resnum)
except TypeError:
    resnum = np.array([resnum])

ithist0 = []
ithispd = []
itcads = []
imtran = []
idur = []
idepth = []
icwidth = []
ipord = []
ilabels = []
iiskoi = []
# if we have searched for this KOI and have exact cadence info for it
# rather than just assuming it's perfectly periodic
iexactkoi = []

# go through the candidates
for jj in np.arange(len(resnum)):
    # for KOI searches
    plotfile = prefix + '.QATSinds' + str(resnum[jj])
    _, tcads = np.loadtxt(plotfile,unpack=True,skiprows=1)
    # make sure it can be used as indexing into arrays
    tcads = tcads[tcads > 0]
    tcads = tcads.astype(np.int)
    mtran = len(tcads)

    # get the info about the planet
    resfile = prefix + '.TopResults.txt'
    durs, depths, pords, cwidths, noises = np.loadtxt(resfile,unpack=True,usecols=(3,4,9,10,11),ndmin=2)
    # get it back in days
    durs /= 24.

    if len(durs) < resnum[jj]:
        print "Can't find this planet in the given location!"
        sys.exit(1)

    # look at one particular planet
    cwidth = cwidths[resnum[jj]-1]
    pord = pords[resnum[jj]-1]
    dur = durs[resnum[jj]-1]
    depth = depths[resnum[jj]-1]
    noise = noises[resnum[jj]-1]

    if np.abs(depth) > 0.1:
        depth *= noise

    thispd = np.mean(time[tcads[1:]]-time[tcads[:-1]])
    # periodic from the time of first transit
    start = np.arange(mtran) * thispd + time[tcads[0]]
    # how much is this estimate off on average
    offset = np.median(time[tcads] - start)
    # adjust it
    start += offset
    thist0 = start[0]

    ithist0.append(thist0)
    #ithispd.append(thispd)
    itcads.append(tcads)
    imtran.append(mtran)
    idur.append(dur)
    idepth.append(depth)
    icwidth.append(cwidth)
    ipord.append(pord)
    #icwidth.append(0.5)
    #ipord.append(3)
    if anonymous:
        ilabels.append('New Candidate; Period XX.XX' +'d')
    else:
        #ilabels.append('New Candidate; Period ' + str(np.round(thispd,decimals=2))+'d')
        ilabels.append('Microlensing pulses')
    #
    iiskoi.append(False)
    iexactkoi.append(False)
# if we want to add in comparisons to the known planets
if usekoi:
    # load in the KOI list and get the units right
    KICs, t0, pd, koidepths, koidurations, _,KOInum = np.genfromtxt(koilist,delimiter=',',unpack=True,filling_values=0., usecols = (0,1,2,3,4,5,6))
    koidurations /= 24.
    koidepths /= 1e6
    t0 += 54833. - ek.timeoffset
    srch = np.where(KICs == KIC)[0]
    """
    if len(srch) < resnum:
        print "Can't find this planet in the KOI list!"
        sys.exit(1)
    ii = srch[resnum-1]
    """
    for count, ii in enumerate(srch):
        thist0 = t0[ii]
        thispd = pd[ii]
        while thist0 > time[0] + thispd:
            thist0 -= thispd
        while thist0 < time[0]:
            thist0 += thispd

        # number of transits
        ntran = np.ceil((time[-1] - thist0)/thispd)
        tcads = np.zeros(ntran)
        for jj in np.arange(ntran):
            ttran = thist0 + jj * thispd
            # nearest cadence to the center of transit
            tcads[jj] = np.where(np.absolute(time - ttran) == min(np.absolute(time - ttran)))[0][0]
        tcads = tcads[tcads > 0]
        tcads = tcads.astype(np.int)
        mtran = len(tcads)

        #findfile = glob(prefix+'Real'+str(int(count+1))+'.QATSinds1' )
        if usetimes:
            findfile = glob(maskprefix+'.'+str(int(count+1))+'.transit_times.txt' )
            if len(findfile) > 0:
                tcads = np.loadtxt(findfile[0],unpack=True)
                mtran = len(tcads)
                iexactkoi.append(True)
                thispd = np.mean(tcads[1:]-tcads[:-1])
                moffset = np.mean(tcads - (thist0 + thispd * np.arange(len(tcads))))
                thist0 += moffset
            else:
                iexactkoi.append(False)
        else:
            findfile = glob(maskprefix+'.'+str(int(count+1))+'.mask.txt' )
            if len(findfile) > 0:
                _, tcads = np.loadtxt(findfile[0],unpack=True,skiprows=1)
                # make sure it can be used as indexing into arrays
                tcads = tcads[tcads > 0]
                tcads = tcads.astype(np.int)
                mtran = len(tcads)
                iexactkoi.append(True)
                thispd = np.mean(time[tcads[1:]]-time[tcads[:-1]])
                moffset = np.mean(time[tcads] - (thist0 + thispd * np.arange(len(tcads))))
                thist0 += moffset
            else:
                iexactkoi.append(False)


        ithist0.append(thist0)
        #ithispd.append(thispd)
        itcads.append(tcads)
        imtran.append(mtran)
        idur.append(koidurations[ii])
        idepth.append(koidepths[ii])
        if len(ipord) >= 1:
            icwidth.append(np.median(icwidth))
            ipord.append(int(np.median(ipord)))
        else:
            icwidth.append(0.7)
            ipord.append(2)
        if anonymous:
            ilabels.append('KOI-XXXX' + '; Period XX.XX'+'d')
        else:
            ilabels.append('Occultations')
            #ilabels.append('KOI-'+str(KOInum[ii]) + '; Period ' + str(np.round(thispd,decimals=2))+'d')

        iiskoi.append(True)

fig = plt.figure(1)
plt.clf()
ithispd = [88.18052,88.18052]

for jj in np.arange(len(imtran)):
    ax = plt.subplot(1, len(imtran),jj+1)
    #plt.fill_between([-1000,1000],[1000,1000],color='#cccccc',zorder=-50)
    thist0 = ithist0[jj]
    thispd = ithispd[jj]
    tcads = itcads[jj]
    mtran = imtran[jj]
    dur = idur[jj]
    depth = idepth[jj]
    cwidth = icwidth[jj]
    pord = ipord[jj]
    prevl = -dur/2.
    prevr = dur/2.
    avgnoise = np.zeros(mtran)

    previi = 0
    # go through every transit
    for ii in np.arange(len(tcads)):
        # where QATS says the center of transit is
        if usetimes and iiskoi[jj]:
            center = np.abs(tcads[ii] - time).argmin()
            center = [center]
        else:
            center = np.where(cad == tcads[ii])[0]

        if len(center) > 0:
            center = center[0]
            toffset = thist0 + ii * thispd

            contregion = np.where((((time >= time[center] - cwidth - dur/2.) & (time <= time[center] - dur/2.)) | ((time >= time[center] + dur/2.) & (time <= time[center] + cwidth + dur/2.))) & (segnum == segnum[center]) & (np.isfinite(fluxerr)))[0]

            if len(contregion) > pord + 2:
                tfit = time[contregion] - time[center]
                ffit = flux[contregion]
                ferrfit = fluxerr[contregion]

                contfit = np.polyval(np.polyfit(tfit,ffit,pord,w=1./ferrfit),tfit)
                avgnoise[ii] = np.std(ffit - contfit)

    mednoise = np.median(avgnoise[avgnoise > 0])
    if depth > 0.:
        fmins = 1. - depth - mednoise#/2.
        fmaxs = 1. + mednoise
    else:
        fmaxs = 1. - depth + mednoise#/2.
        fmins = 1. - mednoise

    # for KOI-3278
    mednoise = 360/1e6
    depth = depth / np.abs(depth) * 0.0011
    if depth > 0.:
        fmins = 1. - depth #- mednoise#/2.
        fmaxs = 1. + mednoise
    else:
        fmaxs = 1. - depth #+ mednoise#/2.
        fmins = 1. - mednoise
    fmins = 1. - 0.001
    fmaxs = 1. + 0.001

    # go through every transit
    #for ii in np.arange(len(tcads)):
    anonoff = int(fstart*len(tcads))
    for ii in np.arange(int(fstart*len(tcads)),int(fend*len(tcads))):
        # where QATS says the center of transit is
        if usetimes and iiskoi[jj]:
            center = np.abs(tcads[ii] - time).argmin()
            center = [center]
        else:
            center = np.where(cad == tcads[ii])[0]

        if len(center) > 0:
            center = center[0]
            toffset = thist0 + ii * thispd

            #region = np.where((time >= time[center] - cwidth - dur/2.) & (time <= time[center] + cwidth + dur/2.) & (segnum == segnum[center]) & (np.isfinite(fluxerr)))[0]
            # get all the points
            region = np.where((time >= time[center] - cwidth - dur/2.) & (time <= time[center] + cwidth + dur/2.))[0]

            contregion = np.where((((time >= time[center] - cwidth - dur/2.) & (time <= time[center] - dur/2.)) | ((time >= time[center] + dur/2.) & (time <= time[center] + cwidth + dur/2.))) & (segnum == segnum[center]) & (np.isfinite(fluxerr)))[0]

            if len(contregion) > pord + 2:
                tfit = time[contregion] - time[center]
                ffit = flux[contregion]
                ferrfit = fluxerr[contregion]
                modfit = np.polyval(np.polyfit(tfit,ffit,pord,w=1./ferrfit),time[region] - time[center])
                fdetrend = flux[region] / modfit

                # points where we shouldn't trust the fluxes to be right
                badpts = np.where(~np.isfinite(fluxerr[region]) | (segnum[region] != segnum[center]))[0]
                fdetrend[badpts] = np.nan

                tfit = time[region] - toffset
                tfit *= 24.

                if dosmooth:
                    fdetrend = sgnl.medfilt(fdetrend,kernel_size=3)

                im = plt.imshow(np.array([fdetrend]),vmin=fmins,vmax=fmaxs,extent=[np.min(tfit)-ek.kepcad/2. ,np.max(tfit)+ek.kepcad/2.,ii-anonoff,ii+1-anonoff],interpolation='nearest',origin = 'lower',aspect='auto',cmap=cmap,zorder=2)#,cmap='winter'
                #if not usekoi:
                """
                if iiskoi[jj] and not iexactkoi[jj]:
                    plt.plot(np.array([-dur/2.,-dur/2.])*24.,[previi,ii+1-anonoff],'r',lw=2)
                    plt.plot(np.array([+dur/2.,+dur/2.])*24.,[previi,ii+1-anonoff],'r',lw=2)
                else:
                    if iiskoi[jj] and usetimes:
                        plt.plot(np.array([tcads[ii]-toffset-dur/2.,tcads[ii]-toffset-dur/2.])*24.,[ii-anonoff,ii+1-anonoff],'r',lw=2)
                        plt.plot(np.array([tcads[ii]-toffset+dur/2.,tcads[ii]-toffset+dur/2.])*24.,[ii-anonoff,ii+1-anonoff],'r',lw=2)
                        if ii > 0:
                            plt.plot(np.array([prevl,tcads[ii]-toffset-dur/2.])*24.,[previi,ii-anonoff],'r',lw=2)
                            plt.plot(np.array([prevr,tcads[ii]-toffset+dur/2.])*24.,[previi,ii-anonoff],'r',lw=2)
                        prevl = (tcads[ii]-toffset-dur/2.)
                        prevr = (tcads[ii]-toffset+dur/2.)
                    else:
                        plt.plot(np.array([time[center]-toffset-dur/2.,time[center]-toffset-dur/2.])*24.,[ii-anonoff,ii+1-anonoff],'r',lw=2)
                        plt.plot(np.array([time[center]-toffset+dur/2.,time[center]-toffset+dur/2.])*24.,[ii-anonoff,ii+1-anonoff],'r',lw=2)
                        if ii > 0:
                            plt.plot(np.array([prevl,time[center]-toffset-dur/2.])*24.,[previi,ii-anonoff],'r',lw=2)
                            plt.plot(np.array([prevr,time[center]-toffset+dur/2.])*24.,[previi,ii-anonoff],'r',lw=2)
                        prevl = (time[center]-toffset-dur/2.)
                        prevr = (time[center]-toffset+dur/2.)
                """
                previi = ii + 1 - anonoff
            else:
                tmpbad = np.zeros(50)
                tmpbad[:] = np.nan
                plt.imshow(np.array([tmpbad]),vmin=fmins,vmax=fmaxs,extent=[-1e5,1e5,ii-anonoff,ii+1-anonoff],interpolation='nearest',origin = 'lower',aspect='auto',cmap=cmap,zorder=2)
    # true unless you're looking for antitransits
    #if fmins < 1.:

    """
    cbar = plt.colorbar(orientation='horizontal',pad=0.15,ticks=[fmins,1.,fmaxs],shrink=0.8)

    # put the labels in ppm depth rather than raw fluxes
    if depth > 0:
        cbar.ax.set_xticklabels([str(int(np.round((fmins-1.)*-1e6,decimals=1))),'0',str(int(np.round((fmaxs-1.)*-1e6,decimals=1)))])
    else:
        cbar.ax.set_xticklabels([str(int(np.round((fmins-1.)*1e6,decimals=1))),'0',str(int(np.round((fmaxs-1.)*1e6,decimals=1)))])

    if bigfont:
        cbar.ax.tick_params(labelsize=22)
        #cbar.set_label('Detrended Flux',fontsize=30)
        if depth > 0.:
            cbar.set_label('Occultation Depth (ppm)',fontsize=26)
        else:
            cbar.set_label('Pulse Height (ppm)',fontsize=26)
    else:
        #cbar.set_label('Detrended Flux')
        cbar.set_label('Transit Depth (ppm)')
    """

    fig.subplots_adjust(bottom = 0.2)

    if plotrange*dur < cwidth:
        if usetimes and iiskoi[jj]:
            minoff = np.min(tcads - (thist0 + thispd * np.arange(len(tcads)))) * 24.
            maxoff = np.max(tcads - (thist0 + thispd * np.arange(len(tcads)))) * 24.
        else:
            minoff = np.min(time[tcads] - (thist0 + thispd * np.arange(len(tcads)))) * 24.
            maxoff = np.max(time[tcads] - (thist0 + thispd * np.arange(len(tcads)))) * 24.
        plt.axis([-(plotrange + 0.5)*dur*24. + minoff,(plotrange + 0.5)*dur*24. + maxoff,0,int((mtran)*(fend-fstart))])
        #plt.axis([-(plotrange + 0.5)*dur*24.,(plotrange + 0.5)*dur*24.,0,mtran*17/20])
    else:
        plt.axis('tight')

    if bigfont:
        plt.title(ilabels[jj],fontsize=30)
        plt.xlabel('Time from center of event (hrs)',fontsize=26)
        if jj==0:
            plt.ylabel('Event number',fontsize=26)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(22)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(22)
        #ax.set_yticklabels(())
    else:
        plt.title(ilabels[jj])
        plt.xlabel('Time from center of event (hrs)')
        if jj==0:
            plt.ylabel('Transit number')

    if jj == 0:

        cbarax = fig.add_axes([0.2,0.07,0.6,0.05])
        cbar = fig.colorbar(im, cax = cbarax, orientation='horizontal',ticks=[fmins,1.,fmaxs])

        # put the labels in ppm depth rather than raw fluxes
        if depth > 0:
            cbar.ax.set_xticklabels([str(int(np.round((fmins-1.)*-1e6,decimals=1))),'0',str(int(np.round((fmaxs-1.)*-1e6,decimals=1)))])
        else:
            cbar.ax.set_xticklabels([str(int(np.round((fmins-1.)*1e6,decimals=1))),'0',str(int(np.round((fmaxs-1.)*1e6,decimals=1)))])

        if bigfont:
            cbar.ax.tick_params(labelsize=22)
            #cbar.set_label('Detrended Flux',fontsize=30)
            if depth > 0.:
                cbar.set_label('Occultation Depth (ppm)',fontsize=26)
            else:
                cbar.set_label('Relative change in flux (ppm)',fontsize=26)
        else:
            #cbar.set_label('Detrended Flux')
            cbar.set_label('Transit Depth (ppm)')

plt.subplots_adjust(wspace=0.1)
    # attempt at a common x-axis label
    #fig.text(0.5,0.1,'Hours',ha='center',va='center')
if saveresult:
    if anonymous:
        midpart = 'River.Anon'
    else:
        midpart = 'River'
    if dopdf:
        plt.savefig(prefix + '.'+midpart+'.pdf')
    else:
        plt.savefig(prefix + '.'+midpart+'.png')

#plt.savefig('/astro/users/eakruse/Desktop/figure_1.eps')

