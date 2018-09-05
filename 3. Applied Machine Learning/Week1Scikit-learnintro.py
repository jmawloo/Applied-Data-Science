# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:37:52 2018

@author: Jeff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from IPython import get_ipython as ipy
ipy().magic("matplotlib qt5") # Figures pop out in separate windows in Spyder.

fruits = pd.read_table("FruitData.txt") # Reads in text files in table format.
fruits.head()

"""Breaking down Table Components:
    - Label numbers for machine supplied by human. Name labels not necessary.
    - Actual feature data contains mass, dimensions and *color*.
    - Color_scale based on cmap and contains score ranging from 0 to 1.
    
2 sets for machine: Training set (labelled set) and test set; split dataframe in half.
    -train_test_split function randomly shuffles dataset and sorts samples into training and test set.
    **Standard split: 75% training, 25% test.
"""

X = fruits[["mass", "width", "height","color_score"]] # Used for features of labels (2-D array)
y = fruits["fruit_label"] # Used for labels themselves. (1-D array)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state provides seed value for function's internal rng.

# Visualizing small dataset with pairplot:
from matplotlib import cm
cmap = cm.get_cmap("gnuplot")
scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={"bins":15}, figsize=(12, 12), cmap=cmap)

# Using a 3D graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c=y_train, marker='o', s=100, cmap=cmap) # 3D scatter plot.
ax.set_xlabel("Width")
ax.set_ylabel("Height")
ax.set_zlabel("color_score")

# Create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) # Number of neighbors to compare with.

# Train classifier (i.e. fit estimator) using training data
knn.fit(X_train, y_train) # subclass inheriting more general estimator class. Implements proper prediction mechanism
# Minkowski Metric has p = 2 which makes it Euclidean Metric.

# Estimate accuracy of classifier on future data using test data.
print(knn.score(X_test, y_test)) # Fraction of test set items whose true label was computed correctly

# Use trained k-NN classifier model to classify new objects.
lookup = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique())) # Label #-name pair
fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.45]]) # Predict function, need all 4 parameters to generate.
print(lookup[fruit_prediction[0]])

fruit_prediction = knn.predict([[100, 10.2, 12.3, 0.63]])
print(lookup[fruit_prediction[0]])

# Sensitivity of k-NN classifier accuracy to choice of 'k' parameter
k_range = range(1,20)
scores=[]
for k in k_range:# k=1 to k=19
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)    
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.scatter(k_range, scores)
plt.xlabel('k')
plt.ylabel("accuracy")
plt.xticks([0,5,10,15,20])

#   Graph displaying decision boundaries.
from adspy_shared_utilities import plot_fruit_knn
plot_fruit_knn(X_train, y_train, 5, "uniform")

