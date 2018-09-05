# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 16:32:43 2018

@author: Jeff
"""
"""   ====Review====   """
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from IPython import get_ipython as ipy # Pops out figures
ipy().magic("matplotlib qt5")
sn.set()

np.set_printoptions(precision=2) # Output display of floats. Sets decimal places to 2.

fruits = pd.read_table("FruitData.txt")

features = ["height", "width", "mass", "color_score"]
X_fruits = fruits[features]
y_fruits = fruits["fruit_label"]
target_names = ["apple", "mandarin", "orange", "lemon"]

X_fruits_2d = fruits[["height", "width"]]
y_fruits_2d = fruits["fruit_label"]

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler # Transform features by scaling them to specified range. feature_range=(0,1) by default.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit transformer object to data & datasize, then transform (more efficient method)
X_test_scaled = scaler.transform(X_test) # Using fitted object to transform test data as well.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("Accuracy of K-NN classifier on training set: {:.2f}".format(knn.score(X_train_scaled, y_train)))
print("Accuracy of K-NN classifier on test set: {:.2f}".format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print("Predicted fruit type for " + str(example_fruit) + " is " + str(target_names[knn.predict(example_fruit_scaled)[0]-1]) + '\n')

"""  ====Datasets====   """

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
                            
# Synthetic dataset for SIMPLE regression: (Dw this will be covered later. Purpose is just to display visualization of this.)
from sklearn.datasets import make_regression
plt.figure()
plt.title("(1) Simple regression problem with one input variable")
X_R1, y_R1 = make_regression(n_samples=100, n_features=1,# X represents feature value
                             n_informative=1, bias=150.0, # y represents regression target value.
                             noise=30, random_state=0) 
plt.scatter(X_R1, y_R1, marker='o', s=50)

# Synthetic dataset for more complex regression
from sklearn.datasets import make_friedman1
plt.figure()
plt.title("(2) Complex regression problem with one input variable")
X_F1, y_F1 = make_friedman1(n_samples=100, n_features=7, random_state=0)
plt.scatter(X_F1[:, 2], y_F1, marker='o', s=50)

# Synthetic dataset for binary classification
from sklearn.datasets import make_classification
plt.figure()
plt.title("(3) Sample binary classification problem with 2 informative features")
X_C2, y_C2 = make_classification(n_samples=100, n_features=2,# 100 n_samples
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0.1, # 10% chance of flipping correct label; pose challenge to classifier.
                                 class_sep=0.5, random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2, marker='o', s=50, cmap=cmap_bold)

# More difficult syndata for binary classifying with non-linearly-separable classes.
from sklearn.datasets import make_blobs
X_D2, y_D2 = make_blobs(n_samples=100, n_features=2, centers=8, cluster_std=1.3, random_state=4) # 100 samples grouped into 8 clusters.
y_D2 %= 2 # To make cluster blobs binary
plt.figure()
plt.title("(4) Sample binary classification problem with non-linearly-separable classes")
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2, marker='o', s=50, cmap=cmap_bold)

# Breast cancer dataset for classification
cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)

# Communities and Crime dataset
X_crime, y_crime =  load_crime_dataset()
# Target valuje to predict: per capita violent crime rate.

"""   ====K-Nearest Neighbors====    """

# Classification
from adspy_shared_utilities import plot_two_class_knn
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

# Figures 5-7
plot_two_class_knn(X_train, y_train, 1, "uniform", X_test, y_test) # Overfitting for complex model b/c too much variance.
plot_two_class_knn(X_train, y_train, 3, "uniform", X_test, y_test) # General trend more properly captured. Less accuracy in training set,
plot_two_class_knn(X_train, y_train, 11, "uniform", X_test, y_test) # But more accuracy in test set.

#    Regression
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
knnreg = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print("R-squared test score:{:.3f}\n".format(knnreg.score(X_test, y_test)))

fig, subaxes = plt.subplots(1, 2, figsize=(8,4))
X_predict_input = np.linspace(-3, 3, 50).reshape(-1, 1) # Linspace uses # of samples instead of step size, then reshapes it to (inferred, 1 column)
X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state=0) # Only take a sample amount of the data.

for thisaxis, K in zip(subaxes, [1, 3]): # K-values up to 55.
    knnreg = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    
# Regression model Complexity as a function of K. (Figure 8)
for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]): # K-values up to 55.
    knnreg = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
   
    thisaxis.set_xlim([-2.5, 0.75])
    thisaxis.plot(X_predict_input, y_predict_output,'^', markersize=10, label="Predicted", alpha=0.8)
    thisaxis.plot(X_train, y_train, 'o', label="True Value", alpha=0.8)
    thisaxis.set_xlabel("Input feature")
    thisaxis.set_ylabel("Target value")
    thisaxis.set_title("KNN reguression (K={})".format(K))
    thisaxis.legend()
    plt.tight_layout() # Prevents clipping of labels.

# Regression model complexity as a function of K
#(plot k-NN regression on sample dataset for different values of K.
fig, subaxes = plt.subplots(1, 5, figsize=(20, 5))
X_predict_input = np.linspace(-3, 3, 500).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0) # Take all the data.

for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]): # (Figure 9)
    knnreg = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_predict_input)
    train_score = knnreg.score(X_train, y_train)
    test_score = knnreg.score(X_test, y_test)
    
    thisaxis.plot(X_predict_input, y_predict_output)
    thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label="Train")
    thisaxis.plot(X_test, y_test, '^', alpha=0.9, label="Test")
    thisaxis.set_xlabel("Input feature")
    thisaxis.set_ylabel("Target value")
    thisaxis.set_title("KNN Regression (K={})\n\\Train $R^2 = {:.3f}$, Test $R^2 = {:.3f}$".format(K, train_score, test_score))
    thisaxis.legend()
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
    # Larger values of K result in simpler models with lower complexity.
    
"""    ====Linear Models for Regression====    """
# Linear regression
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)

print("linear model coeff (w): {}".format(linreg.coef_)) # Underscores denote values derived from training data.
print("linear model intercept (b): {:.4f}".format(linreg.intercept_))
print("R-squared score (training): {:10.5f}".format(linreg.score(X_train, y_train)))
print("R-squared score (test): {:5.3f}\n".format(linreg.score(X_test, y_test)))    

# Example plot
plt.figure(figsize=(5, 4))
plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_*X_R1 + linreg.intercept_, "r-")
plt.title("(10) Least-squares linear regression")
plt.xlabel("Feature value (x)")
plt.ylabel("Target value (y)")

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)

print("Crime dataset")
print("linear model coeff:\n{}".format(linreg.coef_))
print("linear model intercept: {:.4f}".format(linreg.intercept_))
print("R-squared score (training): {:10.5f}".format(linreg.score(X_train, y_train)))
print("R-squared score (test): {:5.3f}\n".format(linreg.score(X_test, y_test)))    

# Ridge regression
from sklearn.linear_model import Ridge # It's just called "Ridge" lol
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0)

linridge = Ridge(alpha=20.0).fit(X_train, y_train)
print("Crime dataset")
print("ridge regression linear model intercept: {}".format(linridge.intercept_))
print("ridge regression linear model coeff:\n{}".format(linridge.coef_))
print("R-squared score (training): {:.3f}".format(linridge.score(X_train, y_train)))
print("R-squared score (test): {:.3f}".format(linridge.score(X_test, y_test)))
print("Number of non-zero features: {}\n".format(np.sum(linridge.coef_ != 0)))

# Ridge regression with feature normalization (dependent on above)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
print("Crime dataset")
print("ridge regression linear model intercept: {}".format(linridge.intercept_))
print("ridge regression linear model coeff:\n{}".format(linridge.coef_))
print("R-squared score (training): {:.3f}".format(linridge.score(X_train_scaled, y_train)))
print("R-squared score (test): {:.3f}".format(linridge.score(X_test_scaled, y_test)))
print("Number of non-zero features: {}\n".format(np.sum(linridge.coef_ != 0)))

# With regularization parameter: alpha
print("Ridge regression: effect of alpha regularization parameter\n")
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha=this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print("Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \nr-squared training: {:.2f}, r-squared test: {:.2f}\n".format(this_alpha, num_coeff_bigger, r2_train, r2_test))

# Lasso Regression
from sklearn.linear_model import Lasso
# from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter=10000).fit(X_train_scaled, y_train) # max_iter is to resolve a convergence warning; increasing increases computation time.
print("Crime dataset")
print("lasso regression linear model intercept: {}".format(linlasso.intercept_))
print("lasso regression linear model coeff:\n{}".format(linlasso.coef_))
print("Non-zero features: {}".format(np.sum(linlasso.coef_ != 0)))
print("R-squared score (training): {:.3f}".format(linlasso.score(X_train_scaled, y_train)))
print("R-squared score (test): {:.3f}\n".format(linlasso.score(X_test_scaled, y_test)))

print("Features with non-zero weight (sorted by abs magnitude:")
for e in sorted(list(zip(list(X_crime), linlasso.coef_)), key=lambda e: -abs(e[1])): # Sort by magnitude in descending order.
    if e[1] != 0:
        print("\t{}, {:.3f}".format(e[0], e[1]))
# In this example positives and negatives represent correlations.
        
# Lasso regression with regualrization parameter: alpha
print("\nLasso regression: effect of alpha regularization\n\\parameter on number of features kept in final model")
for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter=10000).fit(X_train_scaled, y_train)
    r1_train = linlasso.score(X_train_scaled, y_train)
    r1_test = linlasso.score(X_test_scaled, y_test)
    print("Alpha = {:.2f}\nFeatures kept: {}\nr-squared training: {:.2f}, r-squared test: {:.2f}\n".format(alpha, np.sum(linlasso.coef_ != 0), r1_train, r1_test))
    
# "Polynomial regression" (More like polyfeatures)
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)

print("linear model coeff (w): {}".format(linreg.coef_))
print("linear model intercept (b): {:.3f}".format(linreg.intercept_))
print("R-squared score (training): {:.3f}".format(linreg.score(X_train, y_train)))
print("R-squared score (test): {:.3f}".format(linreg.score(X_test, y_test)))

print('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n') # Single \ is a display format for print statement.
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)

print("(poly deg 2) linear model coeff (w):\n{}".format(linreg.coef_))
print("(poly deg 2) linear model intercept (b): {:.3f}".format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

print('\nAddition of many polynomial features often leads to\n\
overfitting, so we often use polynomial features in combination\n\
with regression that has a regularization penalty, like ridge\n\
regression.\n')
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state=0)
linreg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'.format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}\n'.format(linreg.score(X_test, y_test)))

"""   ====Linear models for Classification====   """
#       Logistic regression:

# For binary classification on fruits dataset using height, width features (positive class: apple, negative: others)
from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import ( # Write import statement on multiple lines.
    plot_class_regions_for_classifier_subplot as pcr)

fig, subaxes = plt.subplots(1, 1, figsize=(7,5))
y_fruits_apple = y_fruits_2d == 1 # Transforms to binary (apples vs other objects)
X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d.values, y_fruits_apple.values, random_state=0) # as_matrix() is deprecated

clf = LogisticRegression(C=100).fit(X_train, y_train) # C is amount of regularization.
pcr(clf, X_train, y_train, None, None, "(11) Logistic regression for binary\
 classification\nFruit dataset: Apple vs others", subaxes)

h, w = 6, 8
print("A fruit with height {} and width {} is predicted to be: {}"
      .format(h, w, ["not an apple", "an apple"][int(clf.predict([[h, w]])[0])])) # np.bool_ index interpretation will no longer be supported.

h, w = 10, 7
print("A fruit with height {} and width {} is predicted to be: {}"
      .format(h, w, ["not an apple", "an apple"][int(clf.predict([[h, w]])[0])]))

print("Accuracy of Logistic regression classifier on training set: {:.2f}"
      .format(clf.score(X_train, y_train)))
print("Accuracy of Logistic regression classifier on test set: {:.2f} \n"
      .format(clf.score(X_test, y_test)))

# Logistic regression on simple synthetic dataset
#from sklearn.linear_model import LogisticRegression
#from adspy_shared_utilities import ("pcr")

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

fig, subaxes = plt.subplots(1, 1, figsize=(7,5))
title = "(12) Logistic regression, simple synthetic dataset C = {:.3f}".format(1.0)
pcr(clf, X_train, y_train, None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f} \n'
     .format(clf.score(X_test, y_test)))
     
# Logistic regression regularization: C parameter
X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d.values, y_fruits_apple.values, random_state=0) # as_matrix() is deprecated

fig, subaxes = plt.subplots(3, 1, figsize=(4,10))

for this_C, subplot in zip([0.1, 1, 100], subaxes): # Pair different C values with created subaxes
    clf = LogisticRegression(C=this_C).fit(X_train, y_train) # (Figure 13)
    title = "Logistic regression (apple vs rest), C = {:.3f} \n".format(this_C)
    
    pcr(clf, X_train, y_train, X_test, y_test, title, subplot)
    
    plt.tight_layout()

# Applications to real dataset:
#from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = LogisticRegression().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f} \n'
     .format(clf.score(X_test, y_test))) # Very accurate!

#     Support Vector Machines
# Linear support vector machines
from sklearn.svm import SVC
#from adspy_shared_utilities import "pcr"

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

fig, subaxes = plt.subplots(1, 1, figsize=(7,5))
this_C = 1.0
clf = SVC(kernel="linear", C=this_C).fit(X_train, y_train) # Also has regularization.
title = "(14) Linear SVC, C = {:.3f}".format(this_C)
pcr(clf, X_train, y_train, None, None, title, subaxes)

# LSVM: C parameter
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
fig, subaxes = plt.subplots(1, 2, figsize=(8,4))

for this_C, subplot in zip([1e-5, 100], subaxes): # (Figure 15)
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = "Linear SVC, C = {:.5f}".format(this_C)
    pcr(clf, X_train, y_train, None, None, title, subplot)
    
# Application to real dataset
#from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = LinearSVC().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#     Multi-class classification with Linear Models:
# LinearSVC with M classes generates M one vs rest classifiers.
#from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state=0)

clf = LinearSVC(C=5, random_state=67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

# Multi-class results on fruit dataset
plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

y_fruits_2d = pd.DataFrame(y_fruits_2d) # To satiate the ValueError thrown by c parameter in plt.scatter (Yeah, don't know why either)
#"c of shape (59,) not acceptable as a color sequence for x with size 59, y with size 59", but the sizes are all the same?

plt.scatter(X_fruits_2d[["height"]], X_fruits_2d[["width"]],
           c=y_fruits_2d, cmap=cmap_fruits, edgecolor="black", alpha=0.7) # np.array else shape is not correct
plt.title("(16) LinearSVC with 4 fruit classes")
x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, colors):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=0.8)
    
plt.legend(target_names)
plt.xlabel("height")
plt.ylabel("width")
plt.xlim(-2, 12)
plt.ylim(-2, 15)

"""   ====Kernelized Support Vector Machines====   """
#     Classification
#from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier as pcr2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

# Default SVC kernel is Radial Basis Function (RBF)
pcr2(SVC().fit(X_train, y_train), X_train, y_train, None, None, "(17) Support Vector Classifier: RBF kernel")

# Compare decision boundaries with poly kernel, degree = 3
pcr2(SVC(kernel="poly", degree=3).fit(X_train, y_train), X_train, y_train, None, None, "(18) Support Vector Classifier: Polynomial kernel, degree = 3")

# Support vector machine with RBF kernel: gamma parameter
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot as pcr
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
fig, subaxes = plt.subplots(3, 1, figsize=(4,11))

for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes): # (Figure 19)
    clf = SVC(kernel="rbf", gamma=this_gamma).fit(X_train, y_train)
    title = "Support Vector Classifier: \nRBF kernel, gamma = {:.2f}".format(this_gamma)
    pcr(clf, X_train, y_train, None, None, title, subplot)
    plt.tight_layout()
    
# SVM with RBF Kernel: using both C and gamma parameter:
from sklearn.svm import SVC
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot as pcr
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
fig, subaxes = plt.subplots(3, 4, figsize=(15,10), dpi=50) # Dots per inch

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes): # (Figure 20)
    for this_C, subplot in zip([0.1, 1, 15,250], this_axis):
        title = "gamma = {:.2f}, C = {:.2f}".format(this_gamma, this_C)
        clf = SVC(kernel="rbf", gamma=this_gamma, C=this_C).fit(X_train, y_train)
        pcr(clf, X_train, y_train, X_test, y_test, title, subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
#    SVM application to real dataset: unnormalized data
#from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = SVC(C=10).fit(X_train, y_train)
print("\nBreast cancer dataset (unnormalized features)")
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))

#    SVM application to real dataset: normalized data with feature preprocessing using minmax scaling
#from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}\n'
     .format(clf.score(X_test_scaled, y_test)))

"""   ====Cross-validation====   """
#    Example based on k-NN classifier with fruit dataset (2 features)
from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(n_neighbors=5)
X = X_fruits_2d.values
y = np.ravel(y_fruits_2d.values) # Return contiguous flattened array
cv_scores = cross_val_score(clf, X, y) # 1st argument is training model, 2nd is the 2 features, 3rd is the labels. (3-fold by default)

print("Cross-validation scores (3-fold):", cv_scores) # change cv parameter above for different folds.
print("Mean cross-validation score (3-fold): {:.3f}\n".format(np.mean(cv_scores)))

"""NOTE ON PERFORMING Cross-validation on more advanced scenarios:
    - Certain cases require normalizing data first before introducing it to classifier
    - DON'T scale entire dataset with single transform -> leads to indirect data leakage
    - Must apply scaling to each fold separately. 
    - Easiest way = use "pipelines". For further info:
        http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        or the Pipeline section in the recommended textbook: Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido (O'Reilly Media).
"""

"""   ====Validation Curve====   """
#from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 4) # 1st 2 parameters specifies min and max orders of magnitude, respectively. 3rd specifies count.
train_scores, test_scores = validation_curve(SVC(), X, y, # SVC = radial based feature support vector machine
                                             param_name="gamma",
                                             param_range=param_range,
                                             cv=3)

print("Training scores [row=param value, column=fold number]\n", train_scores)
print("Test scores:\n", test_scores)

# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

plt.figure()

train_scores_mean = np.mean(train_scores, axis=1) # Take mean aggregate of each gamma value result.
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("(21) Validation Curve with SVM")
plt.xlabel("$\gamma$ (gamma)")
plt.ylabel("Score")
plt.ylim(0.0, 1.1) # Set min and max bounds of y-axis
lw = 2 # Linewidth or thickness
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw) #x, y
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw) # X, y_min, y_max
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)

plt.legend(loc="best")

"""   ====Decision Trees====   """
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree # Graphviz function.
#from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))

# Setting max decision tree depth to help avoid overfitting:
clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print('Accuracy of Decision Tree classifier (max decisions = 3) on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))

# Visualizing decision trees (TODO: make it display):
plt.figure()
plot_decision_tree(clf, iris.feature_names, iris.target_names) # (Figure 22)
# Color intensity represents which majority class is present in each node.
# values section corresponds to how many training instances belong in each class.

# Visualizing (pre-pruned version max_depth = 3)
plt.figure()
plot_decision_tree(clf2, iris.feature_names, iris.target_names) # (Figure 23)

# Feature importance
from adspy_shared_utilities import plot_feature_importances

plt.figure(figsize=(10,4), dpi=80) # Dots per inch
plot_feature_importances(clf, iris.feature_names) # (Figure 24)
print("Feature importances: {}\n".format(clf.feature_importances_)) # Inherent property of classifier, not user-defined property

#from sklearn.tree import DecisionTreeClassifier
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot as pcr
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
fig, subaxes = plt.subplots(1, 6, figsize=(32, 6))

pair_list = [[0, 1], [0, 2], [0, 3], [1, 2] ,[1, 3], [2, 3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = "(25) Decision Tree, max_depth = {:d}".format(tree_max_depth)
    pcr(clf, X, y, None, None, title, axis, iris.target_names)
    
    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])
    
    plt.tight_layout()
    
# Decision Trees on real-world dataset:
#from sklearn.tree import DecisionTreeClassifier
#from adspy_shared_utilities import plot_decision_tree
#from adspy_shared_utilities import plot_feature_importances
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state=0)

clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=0).fit(X_train, y_train)
#TODO: Exercise is to remove 2 parameters & see the effect of overfitting on accuracy.

plt.figure()
plot_decision_tree(clf, cancer.feature_names, cancer.target_names) # (Figure 26)

print('Breast cancer dataset: decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}\n'
     .format(clf.score(X_test, y_test)))

plt.figure(figsize=(10,6), dpi=80)
plot_feature_importances(clf, cancer.feature_names) # (Figure 27)
plt.tight_layout()
