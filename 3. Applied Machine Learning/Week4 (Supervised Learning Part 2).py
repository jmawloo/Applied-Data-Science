# -*- coding: utf-8 -*-
#%%
"""
Created on Fri Aug  3 13:53:38 2018

@author: Jeff
"""

"""===   PREAMBLE + DATASETS   ==="""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset
from IPython import get_ipython as ipy
ipy().magic("matplotlib qt5")

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
sn.set()
                            
# Fruits dataset
fruits = pd.read_table("fruit_data_with_colors.txt")

feature_names_fruits = ["height", "width", "mass", "color_score"]
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits["fruit_label"]
target_names_fruits = ["apple", "mandarin", "orange", "lemon"]

X_fruits_2d = fruits[["height", "width"]]
y_fruits_2d = fruits["fruit_label"] # If =y_fruits then just a pointer that refers to y_fruits

# Synthetic dataset for simple regression
from sklearn.datasets import make_regression
plt.figure()
plt.title("(1) Simple regression problem with one input variable")
X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0, noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker='o', s=50)

# SynData for complex regression
from sklearn.datasets import make_friedman1
plt.figure()
plt.title("(2) Complex regression problem with one input variable")
X_F1, y_F1 = make_friedman1(n_samples=100, n_features=7, random_state=0)

plt.scatter(X_F1[:, 2], y_F1, marker='o', s=50)

# SynData for binary classification
plt.figure()
plt.title("(3) Sample binary classification problem with two informative features")
X_C2, y_C2 = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=0)

plt.scatter(X_C2[:, 0], X_C2[:, 1], marker='o', c=y_C2, s=50, cmap=cmap_bold)

# More difficult synthetic dataset for classification (binary)
# with classes NOT linearly separable
X_D2, y_D2 = make_blobs(n_samples=100, n_features=2, centers=8, cluster_std=1.3, random_state=4) # Random state=4? Hmmmm
y_D2 %= 2

plt.figure()
plt.title("(4) Sample binary classification problem with non-linearly separable classes")
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2, marker='o', s=50, cmap=cmap_bold)

# Breast cancer dataset for classification
cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)

# Communities and Crime dataset
X_crime, y_crime = load_crime_dataset()

#%%
"""===   NAIVE BAYES CLASSIFIERS   ==="""
from sklearn.naive_bayes import GaussianNB
from adspy_shared_utilities import plot_class_regions_for_classifier as pcr

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
pcr(nbclf, X_train, y_train, X_test, y_test, "(5) Gaussian Naive Bayes classifier: Dataset 1")

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
pcr(nbclf, X_train, y_train, X_test, y_test, "(6) Gaussian Naive Bayes classifier: Dataset 2")

# Application to real-world dataset
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
print("Breast cancer dataset")
print("Accuracy of GaussianNB classifier on training set: {:.2f})".format(nbclf.score(X_train, y_train)))
print("Accuracy of GaussianNB classifier on test set: {:.2f})".format(nbclf.score(X_test, y_test)))

#%%
"""===   ENSEMBLES OF DECISION TREES   ==="""
"""Random forests"""
from sklearn.ensemble import RandomForestClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot as pcrs

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = RandomForestClassifier().fit(X_train, y_train)
title = "(7) Random Forest Classifier, complex binary dataset, default settings"
pcrs(clf, X_train, y_train, X_test, y_test, title, subaxes)

# Random forest: Fruit Dataset
fig, subaxes = plt.subplots(1, 6, figsize=(32, 6))

X_train, X_test, y_train, y_test = train_test_split(X_fruits.values, y_fruits.values, random_state=0) # .as_matrix() deprecated

title = "(8) Random Forest, fruits dataset, default settings"
pair_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] # Non-repeating Permutations

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = RandomForestClassifier().fit(X, y)
    pcrs(clf, X, y, None, None, title, axis, target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()

clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
print("\nRandom Forest, Fruit dataset, default settings")
print("Accuracy of RF classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of RF classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

# Random Forests on real-world dataset
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = RandomForestClassifier(max_features=8, random_state=0).fit(X_train, y_train)

print("\nBreast cancer dataset")
print("Accuracy of RF classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of RF classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

#%%
"""Gradient Boosted Decision Trees"""
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))
clf = GradientBoostingClassifier().fit(X_train, y_train) # Defaults: learning_rate=0.1, n_estimators=100, max_depth=3
title = "(9) GBDT, complex binary dataset, default settings"
pcrs(clf, X_train, y_train, X_test, y_test, title, subaxes)

# Gradient-boosted decision trees on fruit dataset
X_train, X_test, y_train, y_test = train_test_split(X_fruits.values, y_fruits.values, random_state=0)

fig, subaxes = plt.subplots(1, 6, figsize=(32, 6))
pair_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
title = "(10) GBDT, fruit dataset, default settings"

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = GradientBoostingClassifier().fit(X, y)
    pcrs(clf, X, y, None, None, title, axis, target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])

plt.tight_layout()

clf = GradientBoostingClassifier().fit(X_train, y_train)

print("\nGBDT, Fruit dataset, default settings")
print("Accuracy of GBDT classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of GBDT classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

# Gradient boosted DT on real-world dataset
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = GradientBoostingClassifier(random_state=0).fit(X_train, y_train) # Why seed now?
print("\nBreast cancer dataset (learning_rate=0.1, max_depth=3)")
print("Accuracy of GBDT classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of GBDT classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=2, random_state=0).fit(X_train, y_train) # Lower values reduce overfitting
print("\nBreast cancer dataset (learning_rate=0.01, max_depth=2)")
print("Accuracy of GBDT classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of GBDT classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

#%%
"""===   NEURAL NETWORKS   ==="""
# Activation Functions
xrange = np.linspace(-2, 2, 200)

plt.figure(figsize=(7, 6))

plt.plot(xrange, np.maximum(xrange, 0), label="relu") # Compares and returns maxima of 2 arrays, or broadcastable arrays
plt.plot(xrange, np.tanh(xrange), label="tanh") 
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label="logistic")
plt.legend()
plt.title("(11) Neural network activation functions")
plt.xlabel("Input value (x)")
plt.ylabel("Activation function output")

#%%
"""Neural Networks: Classification"""
# Synthetic dataset 1: single hidden layer
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(1, 3, figsize=(18, 6))

for units, axis in zip([1, 10, 100], subaxes): # units = hidden units to specify when training
    nnclf = MLPClassifier(hidden_layer_sizes=[units], solver="lbfgs", random_state=0).fit(X_train, y_train) # [units] means that the arg can take multiple values for different layers (default is just 100)
    # Solver specifies which algorithm to use for learning weights of network. lbfgs is optimizer in family of quasi_newton methods
    # seed must be set for rand b/c weights initialized randomly.
    
    title = "(12) Dataset 1: Neural net classifier, 1 layer, {} units".format(units)
    pcrs(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()

# Synthetic dataset 1: two hidden layers
nnclf = MLPClassifier(hidden_layer_sizes=[10, 10], solver="lbfgs", random_state=0).fit(X_train, y_train)

pcr(nnclf, X_train, y_train, X_test, y_test, "(13) Dataset 1: Neural net classifier, 2 layers, [10, 10] units")

# Regularization parameter: Alpha
fig, subaxes = plt.subplots(1, 4, figsize=(24, 6))

for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver="lbfgs", activation="tanh", alpha=this_alpha, hidden_layer_sizes=[100, 100], random_state=0).fit(X_train, y_train)
    # With increasing alpha, test score improves and model no longer overfits.
    
    title = "(14) Dataset 2: NN classifier, alpha = {:.3f}".format(this_alpha)
    pcrs(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()

# Effect of different choices of activation function.
fig, subaxes = plt.subplots(1, 3, figsize=(6, 18))

for this_activation, axis in zip(["logistic", "tanh", "relu"], subaxes):
    nnclf = MLPClassifier(solver="lbfgs", activation=this_activation, alpha=0.1, hidden_layer_sizes=[10, 10], random_state=0).fit(X_train, y_train)
    
    title = "(15) Dataset 2: NN classifier, 2 layers [10, 10], {} \
    activation function".format(this_activation)
    
    pcrs(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()

# Application to real-world dataset:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=[100]*2, alpha=5.0, random_state=0).fit(X_train_scaled, y_train)

print("\nBreast cancer dataset")
print("Accuracy of NN classifier on training set: {:.2f}".format(clf.score(X_train_scaled, y_train)))
print("Accuracy of NN classifier on test set: {:.2f}".format(clf.score(X_test_scaled, y_test)))
# 2nd highest score to Random Forest one.

#%%
"""Neural Networks: Regression"""
from sklearn.neural_network import MLPRegressor

fig, subaxes = plt.subplots(2, 3, figsize=(11, 8), dpi=70)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1, 1) # One column, inferred rows

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state=0) # Sample R1 dataset

for thisaxisrow, thisactivation in zip(subaxes, ["tanh", "relu"]):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpreg = MLPRegressor(hidden_layer_sizes=[100, 100], activation=thisactivation, alpha=thisalpha, solver="lbfgs").fit(X_train, y_train)
        y_predict_output = mlpreg.predict(X_predict_input)
        
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output, '^', markersize=10)
        thisaxis.plot(X_train, y_train, 'o')
        
        thisaxis.set_xlabel("Input feature")
        thisaxis.set_ylabel("Target value")
        thisaxis.set_title("(16) MLP regression\n(alpha={}, activation={})".format(thisalpha, thisactivation))
        
        plt.tight_layout()
# Uses 2 hidden layers with 100 hidden units each. Loops thru cycles of different settings
