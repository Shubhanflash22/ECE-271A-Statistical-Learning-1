# ECE 271A: Statistical Learning & Pattern Recognition 🐆📊
A comprehensive implementation of statistical learning methods for image classification and pattern recognition. This project demonstrates Bayesian classification, Gaussian mixture models, and advanced machine learning techniques applied to cheetah vs. grass segmentation using DCT features.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Methods](#methods)
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
All MATLAB solutions for the course are provided in this repository. Each homework assignment is in its respective file: "HWx_solution.m where x can be 1,2,3,5"
```bash
# 1.Clone the repository
git clone https://github.com/yourusername/ece271a-statistical-learning.git
cd ece271a-statistical-learning
# 2. Open Matlab
# 3. Run the Solution
    * Navigate to the folder containing the .m files in MATLAB.
    * Open the desired homework file, e.g., HW1_solution.m.
    * Click Run to execute the script.
```

## Usage

1. Download the required homework
2. Run the script as described above

## Methods
**HW1 – Histogram-Based Bayesian Classification**

* Extracted scalar features: index of the 2nd largest DCT coefficient from 8×8 blocks.
* Constructed class-conditional histograms for cheetah and grass.
* Applied minimum probability of error decision rule (Bayesian classifier).

**HW2 – Multivariate Gaussian Classification**

* Represented each 8×8 DCT block as a 64-dimensional feature vector.
* Modeled class distributions with multivariate Gaussian (full covariance).
* Performed dimensionality reduction and feature selection using marginal densities.
* Compared performance of 64D vs. best 8D features to study the curse of dimensionality.

**HW3 – Bayesian Learning with Priors**

* Implemented three classification strategies: Maximum Likelihood (ML), Maximum a Posteriori (MAP), and Predictive (full Bayesian).
* Incorporated Gaussian priors with varying informativeness for class-specific mean estimation.
* Evaluated effect of prior choice on small vs. large datasets.

**HW4 – Gaussian Mixture Models (GMM)**

* Trained diagonal-covariance GMMs using Expectation-Maximization (EM).
* Studied sensitivity to random initialization (multiple runs per class).
* Evaluated models with different numbers of components (C = 1, 2, 4, 8, 16, 32) to identify optimal complexity.
* Compared performance in terms of segmentation error on cheetah_mask.bmp.

## Results

* Dimensionality Paradox: More features ≠ better performance (8D beats 64D without regularization)
* Prior Informativeness: Class-specific priors significantly outperform neutral priors
* Initialization Sensitivity: EM is highly sensitive to starting point (~60% of inits fail)
* Optimal Complexity: C=8 provides best bias-variance tradeoff for GMM
* Bayesian Benefits: Most pronounced for small datasets; diminish as N grows

## Future Work

* Deep Learning: Replace hand-crafted DCT features with learned CNN representations
* Transfer Learning: Extend to other animal/background classification tasks
* Active Learning: Adaptively select most informative training blocks
* Ensemble Methods: Combine multiple GMM initializations for robust predictions

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
