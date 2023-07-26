# Visualizing-Eigenvector-Dynamics

## Description
Principal component analysis (PCA) can yield scalar weights that can be analyzed and visualized in various ways. The following are a few examples of the power of PCA: Polar Coordinate EQ Bar Visualization, Correlation Matrices, Time Series Correlations.

## Polar Coordinate EQ Bar Visualization
The input data must be a matrix compiling all of the polar coordinate descriptions of an object over time and across different experimental conditons. This particular code is designed for a csv file.

Type in the file directory leading to your input csv file into the "Calculating principal components.py" script. The script will calculate the eigenvalues and associated eigenvectors that will explain the variance in the data, select the top eight eigenvectors, and save them as a csv file.

The "Generating PC weights.py" file will generate associated principal component (PC) weights with each of the eigenvectors from the previous script by applying them to the actual radii of the contour of interest, along with the average shape of the contour of interest over time. This script will need three inputs: first the principal components themselves, then the actual radii of the contour of interest, and then the average shape all as csv files.

The "Graphing PC weights.py" file will generate two graphs of the top eight principal component weights over time, using the output file from the previous script.

The "Top eight eigenvectors.jpeg" file is an image of the top eight eigenvectors calculated from example data.
![Top eight eigenvectors](https://github.com/John-CU-Ahn/Principal-Component-Analysis-of-Biological-Images/assets/140204157/0ca374ec-fb05-4f05-8730-c6ef5d6af8e3)


## Correlation Matrices


## Time Series Correlations
