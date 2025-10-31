# Exoplanet Oblateness Detection Project

Description
This repository provides a framework for detecting the oblateness of exoplanets using MCMC fitting on simulated light curves generated with the Python module Squishyplanet.
The main scripts are designed to:
	•	Manually calculate missing parameters from the NASA Exoplanet Archive,
	•	Simulate realistic JWST-like photometric noise,
	•	Generate synthetic transit light curves, and
	•	Perform MCMC fitting to recover the injected oblateness signals.

Steps
	1.	Download all files in the main directory.
	2.	Run target_selection_final.py → generates selected_targets_v01.csv
	3.	Run Noise_cal_final.py → generates selected_targets_v02.csv
	4.	Run noised_transit_fitting_final.py to perform MCMC fitting and analyze the results.

Target Ranking

Files and scripts in the target_ranking folder contain the methods and calculations used for exoplanet target ranking, which identify the most promising candidates for oblateness detection.
