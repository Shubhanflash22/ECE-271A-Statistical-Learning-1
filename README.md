# ECE 271A: Statistical Learning & Pattern Recognition 🐆📊
A comprehensive implementation of statistical learning methods for image classification and pattern recognition. This project demonstrates Bayesian classification, Gaussian mixture models, and advanced machine learning techniques applied to cheetah vs. grass segmentation using DCT features.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Models & Methods](#models&methods)
* [Results](#results)
* [Future Work](#future-work)
* [License](#license)

## Project Overview
This repository contains my solutions to four comprehensive homework assignments from ECE 271A (Statistical Learning 1) at UC San Diego. The project focuses on cheetah vs. grass segmentation using DCT (Discrete Cosine Transform) features extracted from 8×8 image blocks.

Key implementations include:
* Histogram-based Bayesian classification with minimum probability of error
* Multivariate Gaussian models with Maximum Likelihood estimation
* Bayesian learning with informative priors (MAP and predictive distributions)
* Gaussian Mixture Models (GMM) trained via Expectation-Maximization (EM)

## Dataset
* Image Data:
  * Target: cheetah.bmp (255×270 grayscale image)
  * Ground truth: cheetah_mask.bmp (binary segmentation mask)
  * Training samples: DCT coefficients from labeled 8×8 blocks
* Training Sets:
  * HW1: 250 cheetah samples, 1053 grass samples (scalar features)
  * HW2: Full 64-dimensional DCT feature vectors
  * HW3: Four progressive datasets (D1–D4) with increasing sample sizes
  * HW4: Same as HW2, used for GMM modeling
* Feature Representation:
  * 8×8 DCT blocks → 64 coefficients per block
  * Zig-zag scanning for consistent ordering
  * Features range from DC (mean) to high-frequency AC components

## Features
**HW1: Histogram-Based Classification**
* Scalar Feature Extraction: Index of 2nd largest DCT coefficient
* Bayesian Decision Rule: Minimum probability of error classifier
* Performance: ~16.81% error rate with single-feature model

**HW2: Multivariate Gaussian Classification**
* 64-Dimensional Modeling: Full covariance Gaussian per class
* Feature Selection: Visual inspection of marginal densities
* Dimensionality Comparison: 64D vs. best 8D features
* Performance: 5.52% error (64D), 3.11% error (8D) — demonstrates curse of dimensionality

**HW3: Bayesian Learning with Priors**
* Three Classification Strategies:
  * ML (Maximum Likelihood): Baseline with no prior
  * MAP (Maximum a Posteriori): Point estimate with Gaussian prior
  * Predictive: Full Bayesian integration over posterior uncertainty
* Prior Strategies:
  * Strategy 1: Informative class-specific priors (μ₀=1 for cheetah, μ₀=3 for grass)
  * Strategy 2: Neutral prior (μ₀=2 for both classes)

Performance: Predictive < MAP < ML for small datasets; all converge with large data

**HW4: Gaussian Mixture Models**
* EM Algorithm: Diagonal-covariance GMMs trained via iterative optimization
* Initialization Study: 25 random initializations (5 per class) reveal local optima sensitivity
* Model Complexity Analysis: C ∈ {1, 2, 4, 8, 16, 32} components
* Performance: Optimal at C=8 with ~4% error; C=1,2 fail (>70% error); C=32 overfits

## Installation

```bash

```

## Usage

1. Prepare your dataset in the required format (CSV with sensor parameters).
2. Run the preprocessing script:

```bash
python preprocess_data.py
```

3. Train the machine learning model:

```bash
python train_model.py
```

4. Predict activities on new smartwatch data:

```bash
python predict_activity.py --input new_data.csv
```

## Models & Methods

* Uses classical machine learning algorithms for activity recognition (e.g., Random Forest, SVM).
* Input features include sensor readings like acceleration, gyroscope data, and heart rate.
* Output: Predicted activity label for each timestamp.

## Results

* Achieved high accuracy when tested on real-world smartwatch data.
* Model can reliably predict activities such as walking, running, sitting, and more.

## Future Work

* Expand dataset with more users for better generalization.
* Explore deep learning models (e.g., LSTM, CNN) for temporal sensor data.
* Deploy as a real-time smartwatch application for fitness tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
