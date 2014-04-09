python ./get_data.py
echo 'run.sh : (1/4) Downloaded data.'
python ./transit_cut.py
echo 'run.sh : (2/4) Done loading files.'
python ./mcmc_run.py &
echo 'run.sh : (3/4) Plotting results from the best fit model in the background.'
python ./mcmc_analyze.py &
echo 'run.sh : (4/4) Running the analysis in the background.'