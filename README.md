# Picard-LQG-Pantheon
Python analysis and optimization script for Picard Horn Metric with Planckian Cusp Cut-off on Pantheon+ SNe Ia dataset
# Picard-LQG-Pantheon: MCMC Joint Analysis
**Constraining a Phenomenological Picard Horn Topology with Effective Geometric Cut-off**

This repository contains the Python scripts, MCMC chains, and data processing pipelines used for the Bayesian cosmological constraints on the Picard Horn metric ($\mathbb{H}^3/\text{PSL}(2,\mathbb{Z})$). The study investigates the viability of replacing dark energy with hyperbolic spatial topology while resolving the $H_0$ tension.

## 📌 Abstract / Overview
We perform a joint Markov Chain Monte Carlo (MCMC) analysis utilizing:
* **1590 Type Ia Supernovae** from the Pantheon+ sample (including full systematic covariance).
* **32 Cosmic Chronometers** ($H(z)$ measurements).

Our 3-parameter phenomenological model introduces an effective geometric cut-off scale (inspired by Loop Quantum Gravity volume quantization) to avoid physical singularities. The analysis yields a topological deformation amplitude that deviates from zero at $>7\sigma$ statistical significance, successfully alleviating the Hubble tension ($H_0 = 72.51 \pm 0.28 \text{ km/s/Mpc}$).

## 📂 Repository Structure
* `mcmc_3params_picard.py`: Runs the core 3-parameter MCMC analysis ($H_0, \alpha, z_c$) with a fixed theoretical prior $z_{\text{cut}} = 30.0$. Generates the main corner plot (`fig1.png`).
* `mcmc_2params_lcdm.py`: Runs the standard baseline $\Lambda\text{CDM}$ model ($H_0, \Omega_m$) for AIC/BIC statistical comparison. Generates `fig2.png`.
* `mcmc_4params_unconstrained.py`: Runs the full 4-parameter model (leaving $z_{\text{cut}}$ unconstrained) to demonstrate the flat likelihood at low redshifts. Generates `fig3.png`.

## ⚙️ Requirements
To run the scripts and reproduce the posterior distributions, you will need the following Python libraries:
```bash
pip install numpy pandas scipy emcee corner matplotlib
Link: [https://doi.org/10.5281/zenodo.22761303]
Note: A random seed (np.random.seed(20152024)) has been set in the scripts to ensure full reproducibility of the MCMC posterior quantiles.
📄 Citation
If you find this code or theoretical framework useful in your research, please consider citing the associated preprint on Zenodo:

Nguyen Huy Nhat. (2024). Constraining a Phenomenological Picard Horn Topology with Effective Geometric Cut-off: MCMC Joint Analysis of SNe Ia and Cosmic Chronometers. Zenodo. [Insert Your Zenodo DOI Link Here]
