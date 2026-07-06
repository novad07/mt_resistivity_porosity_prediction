# MT2Porosity: Machine Learning for Effective Porosity Estimation from Magnetotelluric Data

## Study Site
FORGE 78B-32, Utah FORGE Geothermal Field, Milford, Utah, United States (38.500171°N, 112.88221°W)

# Description

Reservoir porosity governs fluid storage capacity and mobility in geothermal systems. Direct measurement of porosity requires well data, which is expensive and spatially limited. Magnetotelluric (MT) resistivity offers indirect access to subsurface porosity structure through its sensitivity to fluid content, but the relationship between resistivity and porosity is not universal. It depends on lithology, fracture network geometry, and thermal fluid history.

This repository implements and compares three machine learning architectures for predicting effective neutron porosity (NDPHI) from 2D magnetotelluric resistivity inversion data at well FORGE 78B-32. The three architectures are Artificial Neural Network (ANN), Long Short-Term Memory (LSTM), and Siamese-LSTM. Each architecture is optimized using Bayesian Optimization through Keras Tuner and evaluated on a depth-based train validation test split.

The resistivity model used in this study comes from a single MT profile located approximately 10 meters from the wellhead, inverted in 2D using TE mode polarization. The inversion result is validated against the AT90 resistivity log with a normalized RMSE of 8.9 percent after Gaussian smoothing. Petrophysical diagnostics included in this repository show that MT resistivity at single well scale behaves as a near perfect monotonic function of depth. This finding follows the same reasoning used in prior MT porosity studies at Soultz sous Forêts, where resistivity contribution could only be meaningfully separated from depth at multi well or spatial scale rather than single well scale.

# Installation

## Using Conda

```
conda env create -f environment.yml
conda activate forge-porosity-ml
```

The environment file includes TensorFlow, Keras Tuner, scikit-learn, lasio for LAS file parsing, and MTpy for dimensionality and strike analysis.

# Data

Raw and processed datasets are located under `data/`. Well log data was loaded from LAS format using lasio and cleaned for outliers prior to depth conversion.

```
data/
├── raw/
│   ├── 78B_AT90.csv                  # AT90 resistivity log, well FORGE 78B-32
│   ├── A1_near_78B.csv               # 2D MT inversion resistivity profile
│   ├── trajectory_78B32.csv          # Directional survey for MD to TVD conversion
```

The final dataset consists of 1476 points at 1 meter depth resolution, with MT resistivity and depth as input features and effective neutron porosity as the target.

# Functionality

## Petrophysical Diagnostics
- Cross correlation and Archie Glover model comparison
- MT resistivity to depth identifiability diagnostic
- AT90 partial correlation analysis after depth is accounted for

## Machine Learning Models
- ANN with Dense layers and LeakyReLU activation
- LSTM with sliding window sequence construction
- Siamese-LSTM with shared backbone and dual branch fusion

## Hyperparameter Optimization
- Bayesian Optimization through custom Keras Tuner subclasses
- Collapsed prediction penalty to prevent constant output on flat validation zones
- Depth based 85/5/10 train validation test split

# Key Findings

MT resistivity at single well scale correlates with depth at a Spearman coefficient near negative one, meaning its contribution to porosity prediction cannot be separated from depth alone at this spatial scale. The AT90 log retains an independent partial correlation of negative 0.376 after depth is controlled for, indicating that near wellbore resistivity carries signal beyond what MT alone resolves here. The classical Archie model fails to reproduce the observed resistivity porosity relationship in this fractured granite setting, while a modified Glover formulation recovers a fluid exponent between 1.77 and 1.90 with matrix resistivity between 100 and 183 ohm meters.

On the test set, all three architectures achieve comparable MAPE, with LSTM at 5.10 percent, Siamese-LSTM at 5.62 percent, and ANN at 6.22 percent. Negative R² values on the test set are expected under this depth based splitting scheme and reflect distributional shift rather than architecture failure, since the test interval lies outside the depth range seen during training.
