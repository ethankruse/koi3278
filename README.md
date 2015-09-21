# KOI 3278

This contains a stand-alone version of the code used to analyze the
self-lensing binary system KOI-3278, as it appears in Science. It will
reproduce the bulk of our analysis, including the MCMC modeling and
the key figures, notably Figure 2 of the main text.

If you make use of this code please cite our work:
[Kruse & Agol, Science 344, 275 (2014).](http://www.sciencemag.org/content/344/6181/275.short)

We have also included the PARSEC isochrones used (PARSECv1.1/) as well
as wget scripts to download the Kepler light curves (to lightcurve/)
and white dwarf models from Pierre Bergeron (wdmodels/). See the
README files within the subdirectories for more information.

Everything can be run using python assuming the following packages are
installed (most are defaults in every python installation):
* `sys`
* `os`
* `subprocess`
* `numpy`
* `scipy`
* `matplotlib`
* `glob`
* `astropy`
* `emcee`

To reproduce all that we have done, one can simply run the shell
script `./run.sh` and all the figures should appear. This procedure
should take less than a minute to complete. It may take slightly
longer on the first run as the wget scripts download the necessary
data files.

If you would like to generate the table of median model values and
their one-sided error bars, you can change the `texout` variable in
mcmc_analyze.py from None to any file name. This will take a couple
minutes to complete.

Finally, if you would like to perform your own MCMC run, you can
change the `domcmc` variable in mcmc_run.py from False to True. The
length of the run will depend on the number of walkers and iterations
(also chosen as variables in the file), but with the defaults it will
take ~24 hours.
