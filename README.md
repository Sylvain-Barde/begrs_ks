# begrs_ks
This repository contains the replication files for the BEGRS estimation of the financial K+S model of *Dosi, G., Fagiolo, G., Napoletano, M., Roventini, A. and Treibich, T., 2015. Fiscal and monetary policies in complex evolving economies. Journal of Economic Dynamics and Control, 52, pp.166-189.*

## Requirements and Installation

Running replication files require:
- The `begrs` toolbox with dependencies installed for estimation.
- The MIC toolbox, used for the goodness of fit. This can be downloaded from https://github.com/Sylvain-Barde/mic-toolbox (Note, this might require compilation)

Note: the files were run using GPU-enabled and large multi-CPU HPC nodes, therefore any attempt at replication should take into account this computational requirement. This is particularly the case for the SBC analysis, which is time-consuming even on an HPC node. The files are provided for the sake of transparency and replication, and all results are provided in the associated release (see below).

## Release contents

The release provides 4 zipped archives which collectively contain the following folders. These contain all the configuration files for the K+S model as well as all the intermediate results of the scripts, so that the tables and figures in the paper can be generated directly, without requiring a full re-run of the entire analysis.

- `/figures`: contains the output figures that are used in the paper
- `/K+S`: contains the codebase for the K+S model as well as the simulated datasets used for the BEGRS estimation, the SBC analysis, MIC goodness-of-fit and policy experiments
- `/logs` : contains run logs for the MIC analysis
- `/models`: contains the saved trained BEGRS surrogate models and their associated posterior estimates
- `/sbc`: contains the results of the SBC diagnostic of the BEGRS surrogate
- `/scores`: contains the MIC scores on the empirical datasets
- `/tables`: contains the tables used in the paper
- `/us_data`: contains the empirical US data used in the estimation

## Run sequence:

The various scripts should be run in the following order, as the outputs of earlier scripts for the inputs of later ones. To run a later file (e.g. output generation) without running an earlier file (e.g. estimation), use the folders provided in the release as the source of the intermediate inputs.

### 1. Generate simulation training data

- `parallel_ks_train_run.py` - Generate K+S simulation data (used for training and SBC testing). This script requires multi-CPU node.

### 2. Run estimations

 Run the BEGRS estimation and the SBC diagnostic on the training and testing data.

- `begrs_ks_train.py` - Train a BEGRS object on the simulated K+S data.
- `begrs_ks_est_us.py` - Estimate the K+S model parameters using BEGRS from the empirical data.
- `begrs_ks_sbc.py` - Run a SBC diagnostic on the BEGRS surrogate.

### 3. Run MIC score and policy analyses

- `parallel_ks_mc_run.py` - Generate K+S simulation data using the BEGRS posterior estimates. Generates data used for both the MIC analysis and the policy replications. Requires multi-CPU node.
- `quantisation diagnostics.py` - (Optional) Used to verify the quantisation settings used in the MIC analysis.
- `parallel_mic_score.py` - Run MIC analysis on simulated data. Requires multi-CPU node.

### 4. Generate outputs for paper

- `begrs_ks_est_outputs.py` - Generate outputs from the BEGRS estimation for the paper.
- `begrs_ks_sbc_outputs.py` - Generate outputs from SBC diagnostic of BEGRS for the paper.
- `sim_table_output.py` - Generate descriptive statistics of simulated data.
- `mc_run_output.py` - Generate tables for the replication of the policy analysis.
- `mic_score_output.py` - Generate tables for MIC goodness-of-fit analysis.
