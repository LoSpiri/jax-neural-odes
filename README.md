# GRU-RNN & GRU-ODE-Bayes & Continuous Normalizing Flows

**Machine Learning Project for Life Sciences**  
*Authors: Lorenzo Spiridioni & Maxim Hirschmann*  
*Date: February 10, 2025*

## Overview
This repository contains implementations and experiments exploring **neural ODE-based architectures** for handling continuous-time data and generative modeling. The project focuses on two main methodologies: **GRU-ODE-Bayes** for irregularly sampled time series and **Continuous Normalizing Flows (CNF)** for density estimation and sampling.

## Key Concepts

### GRU-ODE-Bayes
A hybrid architecture combining continuous-time evolution with discrete updates, designed to handle:
- **Irregular timestamps** naturally via continuous modeling.
- **Asynchronous, feature-level missingness** by updating only when data arrives.
- **Uncertainty evolution** which reduces upon observation.

**Mechanism:**
1.  **Predictor (Continuous):** Evolves a latent hidden state in continuous time using an ODE-inspired GRU.
2.  **Corrector (Discrete):** Performs a Bayesian-style update of the hidden state when an observation arrives.

### Continuous Normalizing Flows (CNF)
A generative framework that transforms a simple base distribution (e.g., Gaussian) into a complex target data distribution via an invertible ODE.
- **Flowing to Target:** The probability mass is reshaped over time by integrating an ODE parameterized by a neural network.
- **Applications:** Generative modeling, exact log-likelihood evaluation, and sampling.

## Implementations & Results

### GRU-ODE-Bayes Experiments
- **Lynx and Hare & Covid Datasets:** Benchmarked performance against simple Neural ODEs.
- **Spirals Dataset:** Demonstrated robustness to **irregular and missing data**, maintaining lower loss compared to standard models even with 25% data availability.
- **Sepsis Dataset:** Applied to medical time series for classification tasks.

### CNF Experiments
- **Ethanol Sampling:** successfully learned the distribution of 9-atom structures (5,000 variations) for generative sampling.
- **Likelihood Estimation:** Evaluated log-likelihoods on image data.

## Open Source Contribution
As part of this work, we contributed to the open-source ecosystem:
- **Diffrax Pull Request:** [Link to PR #728](https://github.com/patrick-kidger/diffrax/pull/728)
