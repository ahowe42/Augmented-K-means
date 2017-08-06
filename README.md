# Augmented-K-means
Python code to implement the Augmented k-means algorithm as described in my arXiv article #1705.07592 [Improved Clustering with Augmented k-means](http://arxiv.org/abs/1705.07592v1).

This repository includes the custom Python scripts and data files to replicate (up to stochasticity) the results published in the above-mentioned article. Python dependencies related to computation and modeling are `numpy`, `scikit-learn`, and `matplotlib`. Each file is briefly described here.

- `2017ClusteringAugmentedkmeans.pdf` - the above-mentioned article, downloaded in May 2017

- `AugKM_Test_Driver.py` - script to perform Monte Carlo simulation, using either known or simulated data, comparing the clustering performance of Augmented k-means with k-means++

- `AugKmeansPP.py` - the actual Augmented k-means algorithm

- `SupportFuncs.py` - contains a few generic support functions specifically relevant to clustering: `ScatterGrpd` generates either a 2d or 3d scatter plot of grouped data, with each group using a different color & marker; `LoadClustData` prepares either real or simulated data for modeling; `SimClustData` simulates observations of data from a mixture of *Multivariate Power Exponential* distributions; `MVPexpRnd` simulates multivariate PE data; `ClustsAlign` attempts to edit estimated cluster labels so that the label indices align optimally with known labels; `SimandScat` is a helper function which sequentially calls `LoadClustData` and `SimClustData`

- `data_mixnorm_5.m` - data file which codes for a simulation protocol which generates and overlapping mixture of 4 elliptical PE distributions, having varying values of the distribution parameters; see `LoadClustData` for an explanation of the data format

- `data_mixnorm_5.eps` - a grouped bivariate scatterplot of a sample of data generated using the protocol `data_mixnorm_5.m`

- `iris_data.m` - tab-separated file with Fisher's Iris dataset; the first column are the cluster labels

- `wine_data.m` - tab-separated file with the wine recognition dataset of M. Fiorina *et al.* used in "Aeberhard, S., Coomans, D., de vel, O., 1992. Comparison of Classifiers in High Dimensional Settings. Tech. Rep. 92-02, Dept. of Computer Science and Dept. of Mathematics and Statistics, James Cook University of North Queensland."
