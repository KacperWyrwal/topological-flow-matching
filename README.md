# Topological Flow Matching 
Repository for the earthquake, traffic flow, ocean current, single-cell differentiation, and brain fMRI experiments in Topological Flow Matching.

Please first set up a new virtual environment and, after activating it, run
```
pip install -r requirements.txt 
pip install -e . 
```

To reproduce a given experiment on GPU run
```
python -m topofm.cli.run -m experiment=experiment_name/model_name 
```
for ```experiment_name``` in ```[earthquakes, traffic, ocean, single_cell, brain]``` and ```model_name``` in ```[icfm, otcfm, itfm, ottfm]```. 

To run an experiment on CPU run 
```
python -m topofm.cli.run -m experiment=experiment_name/model_name run.device=cpu
```

Thank you!
