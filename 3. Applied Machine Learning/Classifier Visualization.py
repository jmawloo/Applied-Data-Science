# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:35:43 2018

@author: Jeff

Classifier Visualization Playground
The purpose of this notebook is to let you visualize various classsifiers' decision boundaries.

The data used in this notebook is based on the UCI Mushroom Data Set stored in mushrooms.csv.

In order to better vizualize the decision boundaries, we'll perform Principal Component Analysis (PCA) on the data to reduce the dimensionality to 2 dimensions. Dimensionality reduction will be covered in a later module of this course.

Play around with different models and parameters to see how they affect the classifier's decision boundary and accuracy!
"""

from IPython import get_ipython as ipy
ipy().magic("matplotlib qt5")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

df = pd.read_csv("mushrooms.csv")
df2 = pd.get_dummies(df) # Convert categorical vars into dummy/indicator vars.

df3 = df2.sample(frac=0.08, random_state=0) # return random sample of items from axis/object. Frac = fraction of items to return.

X = df3.iloc[:, 2:]
y = df3.iloc[:, 1]

pca = PCA(n_components=2).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label="Edible", s=2) # Don't put 0,1 as (0,1)
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label="Poisonous", s=2)
plt.legend()
plt.title("Mushroom Data Set\nFirst Two Principal Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.gca().set_aspect("equal")

def plot_mushroom_boundary(X, y, fitted_model):
    plt.figure(figsize=(9.8, 5), dpi=100)
    
    for i, plot_type in enumerate(["Decision Boundary", "Decision Probabilities"]):
        plt.subplot(1, 2, i+1)
        
        mesh_step_size = 0.01 # step size for meshgrid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + .1  # Offset min_max values by .1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        
        if i == 0:
            Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()]) #np.c_ = Slice object translation to concatenation along second axis. Ravel() returns 1D contiguous array
        else:
            try:
                Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            except:
                plt.text(0.4, 0.5, "Probabilities Unavailable", horizontalalignment="center", verticalalignment="center", transform=plt.gca().transAxes, fontsize=12)
                plt.axis("off")
                break
            
        Z = Z.reshape(xx.shape)
        plt.scatter(X[y.values==0,0], X[y.values==0,1], alpha=0.4, label="Edible", s=5) 
        plt.scatter(X[y.values==1,0], X[y.values==1,1], alpha=0.4, label="Poisonous", s=5)
        plt.imshow(Z, interpolation="nearest", cmap="RdYlBu_r", alpha=0.15, extent=(x_min, x_max, y_min, y_max), origin="lower") # display img on axes. Extent is lower-left and upper-right corners of image. Origin=place [0,0] index in lower left corner of axis.
        plt.title(plot_type + '\n' + str(fitted_model).split('(')[0] + " Test Accuracy: " + str(np.round(fitted_model.score(X, y), 5)))
        plt.gca().set_aspect("equal"); # Suppress output
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.02)
    
#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=100.0, max_iter=300).fit(X_train, y_train) #C=1.0 by default. Apparently doesn't affect accuracy?

plot_mushroom_boundary(X_test, y_test, model)
        
#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train) # 5 neighbors by default? Also preset to 20.

plot_mushroom_boundary(X_test, y_test, model)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, max_leaf_nodes=100) # Presets: max_depth=3, others=None.
model.fit(X_train, y_train)

plot_mushroom_boundary(X_test, y_test, model)

# **Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier # Performs multiple Decision Tree fitting on dataset.
model = RandomForestClassifier(max_depth=10).fit(X_train, y_train) # Default=None

plot_mushroom_boundary(X_test, y_test, model)

# Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=100).fit(X_train, y_train) # Choices include linear, rbf
#model = SVC(kernel="linear", C=0.1) Score of 0.83
#model = SVC(kernel="rbf", C=1) Score of 0.8344
#model = SVC(kernel="rbf", C=10) Score of 0.87117
plot_mushroom_boundary(X_test, y_test, model)

# **Gaussian Naive_Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train) #perform online updates to model parameters via `partial_fit` method.

plot_mushroom_boundary(X_test, y_test, model)

# **Neural Network MLPClassifier
from sklearn.neural_network import MLPClassifier
model = MLPClassifier().fit(X_train, y_train)

plot_mushroom_boundary(X_test, y_test, model)