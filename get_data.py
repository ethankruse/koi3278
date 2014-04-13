"""
Download the necessary files if they don't already exist.
"""

import inputs as inp
from glob import glob
import subprocess
import os

dataloc = inp.keplerdata
KIC = 3342467

KICstr = str(int(KIC))
files = glob(dataloc + 'kplr*' + KICstr + '*llc.fits')

# change this if we actually had to download something
dload = 0
# can't find the light curves
if len(files) == 0:
    # move to the download location
    cwd = os.getcwd()
    os.chdir(dataloc)
    # run the wget script to get the light curves from MAST
    subprocess.check_call(['./kepler_wget.sh'])
    os.chdir(cwd)
    dload += 1

# check for the WD models
files = glob(inp.wdfiles)
if len(files) == 0:
    # move to the download location
    cwd = os.getcwd()
    os.chdir('./wdmodels/')
    # run the wget script to get the WD models from Bergeron website
    subprocess.check_call(['./bergeron_wdmodels_wget.sh'])
    os.chdir(cwd)
    dload += 1

if dload:
    print 'Downloaded necessary data.'
else:
    print 'All data already downloaded. Continuing.'
