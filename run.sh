python transit_cut.py
echo 'run.sh : (1/3) Done loading files.'
python mcmc_run.py &
echo 'run.sh : (2/3) Plotted results from the best fit model.'
python mcmc_analyze.py &
echo 'run.sh : (3/3) Done with the analysis.'