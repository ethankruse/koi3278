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
from scipy import optimize as opt
import matplotlib.ticker as plticker
import scipy.stats
from model_funcs_public import logprob, loglikeli, initrange, light_curve_model, msage, kepler_problem, isointerp, loadisos


# whether or not to use the adjustments for crowding (3rd light contamination)
usecrowd = True
# crowding value in Kepler for each quarter (1-17)
quartcontam = np.array([0.9619, 0.9184, 0.9245, 0.9381, 0.9505, 0.9187, 0.9246, 0.9384, 0.9598, 0.9187, 0.9248, 0.9259, 0.9591, 0.9186, 0.9244, 0.9383, 0.9578])
# quarter for each event
equarts = np.array([ 1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,  16, 16, 17])

# use scipy.optimize to get an initial fit
findfit = False
# run the full MCMC model
domcmc = False
# where to save the MCMC output
outfile = './chain_final_isochrones.txt'
# number of MCMC iterations
niter = 100000

# fit quadratic limb darkening coefficients as free parameters
# if False, uses the Sing fits based on the stellar parameters
fitlimb = False

# subsample/npert: change sampling of times per cadence to obtain higher accuracy at ingress/egress
subsample  = 10

# time/fluxes/flux errors for all events to fit
# must have equal numbers of points in each cut!
infile = './KOI3278_events_sap.txt'

# multiply the Kepler flux errors by this amount to get a reduced chi-square closer to 1.
expanderror = 1.13

# =========================================================================
# load in the sections of the light curve near transits
t,f,ferr = np.loadtxt(infile,unpack=True)
ferr *= expanderror

# set up the crowding parameters for each event
crowding = np.ones(len(equarts))
if usecrowd:
    for ii in np.arange(len(crowding)):
        crowding[ii] = quartcontam[equarts[ii]-1]


# ==================================================================== #
# isochrone loading section

# this takes forever, so if you've already loaded things once, don't bother again
try:
    loaded
except NameError:
    loaded = 1

    isobundle = loadisos()

    # unpack the model bundle
    magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle
    minfeh, maxfeh, minage, maxage = limits


print 'Done loading isochrones'
# ============================================================================ #

# current parameters for the model and starting values
labels = ['$P$ (days)','$t_{tran}$ (days)','$e\cos\omega$','$e\sin\omega$','$b$','$M_{2,init}$','$M_2$','$M_1$','[Fe/H]','Age (Gyr)','Distance (pc)','$\sigma_{sys}$','h (pc)','$A_\lambda$ scale']
# median solution from the MCMC analysis (without crowding)
p = np.array([  8.81804953e+01,   8.54189557e+01,   1.47075839e-02, 3.88439171e-03,   7.10199165e-01,   2.20962752e+00, 6.11162960e-01,   1.03747625e+00,   3.80623768e-01, 1.91048678e+00,   8.09286733e+02,   2.41907976e-02, 1.17995034e+02,   1.00266715e+00])
# absolute minimum chi-square found in MCMC (without crowding)
p = np.array([  8.81806192e+01,   8.54188445e+01,   1.47156750e-02, 6.15356399e-03,   7.05240224e-01,   2.38417117e+00, 5.96523680e-01,   1.02311930e+00,   3.09126840e-01, 1.71942645e+00,   7.87965211e+02,   1.73709656e-02, 1.13214456e+02,   9.97091028e-01])

# median solution from the MCMC analysis (with crowding)
p = np.array([  8.81805180e+01,   8.54189900e+01,   1.47132293e-02, 4.83767012e-04,   7.05595086e-01,   2.40081224e+00, 6.33573877e-01,   1.04177206e+00,   3.94625983e-01, 1.62016796e+00,   8.08342999e+02,   2.46057348e-02, 1.17068978e+02,   1.00122149e+00])
# absolute minimum chi-square found in MCMC (with crowding)
p = np.array([  8.81805979e+01,   8.54189422e+01,   1.47105950e-02, 5.83059972e-03,   7.02722610e-01,   2.35546161e+00, 6.26868773e-01,   1.03255051e+00,   3.46963869e-01, 1.71307399e+00,   7.99324162e+02,   1.51296591e-02, 1.23274350e+02,   1.00831069e+00])


if fitlimb:
    p = np.concatenate((p,np.array([5.64392567e-02, 5.07460729e-01])))
    labels.append('$u_{S1,1}$')
    labels.append('$u_{S1,2}$')





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
    for result in sampler.sample(startlocs,iterations=niter,storechain=False):
        position = result[0]
        iternum = sampler.iterations
        ofile = open(outfile,'a')
        for k in np.arange(position.shape[0]):
            ofile.write('{0} {1} {2} {3}\n'.format(iternum,k,str(result[1][k])," ".join([str(x) for x in position[k]])))
        ofile.close()
        print iternum


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

magobs, magerr, maglam, magname, interps, limits, fehs, ages, maxmasses, wdmagfunc = isobundle

wdage = np.log10(10.**age - 10.**(msage(M2init,FeH,isobundle)))
wdmag = wdmagfunc(np.array([[M2,wdage]]))[0]



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
