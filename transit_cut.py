"""
Take approximate cadences of transit and chop them out into
a transit-only light curve.
"""
import numpy as np
from prepare_lc import preparelc

# list of input files. Must be a list. Assumed to be in the format of QATS
# output, so first line is skipped and all other lines contain transit index
# (ignore this) and cadence columns
infiles = ['./KOI3278cen_tran.txt', './KOI3278cen_occ.txt']

KIC = 3342467
usepdc = False

# total number of cadences to cut out around each event
region = 64

# where to save this chunk of the light curve
outtxt = './KOI3278_events_sap.txt'

# ==========================================================================

cads = np.array([])
for ii in np.arange(len(infiles)):
    # get the cadences of events
    tcads = np.loadtxt(infiles[ii], skiprows=1, unpack=True,
                       usecols=(1,))
    for jj in tcads:
        start = int(jj-region/2)
        # add surrounding regions to the list of cadences to save
        twin = np.arange(start, start+region)
        cads = np.concatenate((cads, twin))

# get the light curve
time, flux, fluxerr, cad, quart, qual = preparelc(KIC, usepdc=usepdc,
                                                  fill=True)

lorig = len(cads)
# get valid cadences
cads = np.unique(cads)
cads = cads[(cads > 0) & (cads < cad[-1])]
cads = cads.astype(int)

# MCMC run requires equal number of cadences per event
# will have to handle errors (e.g. transits near the beginning or end
# of observation) before generalizing this for use with other systems
if lorig != len(cads):
    print 'Warning! No longer equal number of cadences per event!'

# save the selected portion of the light curve
outarr = np.zeros((len(cads), 3))
outarr[:, 0] = time[cads]
outarr[:, 1] = flux[cads]
outarr[:, 2] = fluxerr[cads]

np.savetxt(outtxt, outarr)
