python ./get_data.py
echo 'run.sh : (1/4) Downloaded data.'
python ./transit_cut.py
echo 'run.sh : (2/4) Done loading files.'
python ./mcmc_run.py &
echo 'run.sh : (3/4) Plotted results from the best fit model.'
python ./mcmc_analyze.py &
echo 'run.sh : (4/4) Done with the analysis.'