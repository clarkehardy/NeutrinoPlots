# NeutrinoPlots

## About

This code is intended to be used to reproduce some of the typical plots used in neutrino physics for papers, theses, or talks. Currently, it can be used to make the "Lobster Plot" with the probability density in the parameter space shown, in addition to the usual 3&sigma; contours. This is done using a Markov Chain Monte Carlo (MCMC) to sample the marginalized posterior distribution of the effective Majorana mass, as described in Ref. \[1\].

## Installation

The package can be installed by running the following from the top-level directory:
```bash
pip install -e .
```
During this step the dependencies listed in [requirements.txt](requirements.txt) will be automatically installed. For this reason it is recommended to install the package in a virtual p ython environment using `venv`, `virtualenv`, or `conda`. To uninstall, run:
```bash
pip uninstall neutrino-plots
```

## Usage



## References

1. Agostini *et al* PRD **96**, 053001 (2017)

