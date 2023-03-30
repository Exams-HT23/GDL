# Geometric Deep Learning
*Hilary Term 2023*

This repository provides the anonymised codebase for candidate number 1069635.

## Overview
* [encoder.ipynb](/encoder.ipynb) - Conducts experiments to determine whether removal of the encoder in GRAFF significantly diminishes performance or not.
* [scalability.ipynb](/scalability.ipynb) - Conducts experiments to assess the performance of the scalable version of GRAFF.
* [models.py](/models.py) - Provides implementations of MLP, GCN, standard GRAFF (with and without an encoder) and scalable GRAFF models.
* [preprocessing.py](/preprocessing.py) - Provides code to pre-diffuse node features for a number of large-scale datasets.
* [utils.py](/utils.py) - Contains utility functions for training and evaluating models.
