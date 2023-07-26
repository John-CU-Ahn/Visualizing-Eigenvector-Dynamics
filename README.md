# Visualizing-Eigenvector-Dynamics

## Description
Principal component analysis (PCA) can yield scalar weights that can be analyzed and visualized in various ways. The following are a few examples of the power of PCA: Polar Coordinate EQ Bar Visualization, Correlation Matrices, Time Series Correlations.

## Polar Coordinate EQ Bar Visualization
This visualization displays the spatiotemporal impact of different principal component (PC) scalar weight values of a cancer cell.
![frame 564](https://github.com/John-CU-Ahn/Visualizing-Eigenvector-Dynamics/assets/140204157/0d89651b-5c4d-4b25-ab90-ae05e381d1b7)

Type in the file directories leading to your input csv files into the "PC EQ Generating small barplots.py" script to generate .png files of the barplots. You will need PC scalar weights from experimental data.

To generate the cartesian representations of the scalar weights, add the indicated directories into the "PC EQ Cartesian scalar weights.py" script.

To stitch the barplots and cartesian plots together, use the "PC EQ Overlaying barplots on the cartesian scalar weights.py" file.

The "Polar Coordinate EQ Bar movie.mp4" video file shows this visualization over time.
## Correlation Matrices
This visualization displays correlations between all pairs of the top eight selected principal components.
![frame_177](https://github.com/John-CU-Ahn/Visualizing-Eigenvector-Dynamics/assets/140204157/9cd8a8e4-ecb6-4034-8d68-c924d3dc858a)

Add the indicated directories into the "Correlation matrices.py" file to generate the heatmaps of the pairwise correlations.

The "Correlation Matrices.mp4" video file shows this visualization over time.
## Time Series Correlations
This visualization shows one pairing of PC correlation over time, highlighting moments of significant positive and negative correlation.
![Frame_goldblue0_Cell0_window30_Positive and Negative correlations of PC1 and PC2 example](https://github.com/John-CU-Ahn/Visualizing-Eigenvector-Dynamics/assets/140204157/728f97f4-33f0-4059-a1fe-db89c89b5a2d)

Add the indicated directories into the "Time series correlations.py" file.

The "Time Series Correlations.mp4" video file shows this visualization over time.
