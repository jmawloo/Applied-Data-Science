# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:35:16 2018

@author: Jeff
"""

"""   ===Evaluation for Classification===   """
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from IPython import get_ipython as ipy
ipy().magic("matplotlib qt5")

print(plt.style.available) # Prints all the styles out
plt.style.use("seaborn-darkgrid")

dataset = load_digits() # Handwritten digits with 10 classes (representing digits 0 through 9)
X, y = dataset.data, dataset.target 

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)): # Count of instances in each class (roughly balanced) using np.bincount(Frequency of each successive integer.)
    print(class_name, class_count)
print("")
# Now simulate imbalanced dataset by labelling 1 as positive class 1 and all other classes negative (0)
y_binary_imbalanced = y.copy() # pd Dataframe
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print("Original labels:\n    ", y[1:30])
print("New binary labels:    \n", y_binary_imbalanced[1:30])
print(np.bincount(y_binary_imbalanced)) # 9:1 ratio

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
svm = SVC(kernel="rbf", C=1).fit(X_train, y_train)
print("Radial basis score: ", svm.score(X_test, y_test)) # 90% seems pretty good. Is it really though?

"""Dummy Classifiers; useful as baseline to compare against actual classifiers esp. with imbalanced classes."""
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train) # Uses a builtin strategy. Looks for the most frequent y_train label occurrences.

y_dummy_predictions = dummy_majority.predict(X_test)
print('\n', y_dummy_predictions)
print("Dummy score: ", dummy_majority.score(X_test, y_test), '\n') # Almost same accuracy too.

# Change parameters to improve accuracy of SVC
svm = SVC(kernel="linear", C=1).fit(X_train, y_train)
print("Linear Kernel score: ", svm.score(X_test, y_test), '\n')

"""Confusion Matrices"""
# Binary (2-class) confusion matrix
from sklearn.metrics import confusion_matrix

dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print("Most frequent class (dummy classifier)\n", confusion) # True + false neg in 1st column, and False + True pos in 2nd col. All negatives b/c what dummy assigned to do.

# Different dummy classifier type:
dummy_classprop = DummyClassifier(strategy="stratified").fit(X_train, y_train) # Will give slightly different results due to randomness.

y_classprop_predicted = dummy_classprop.predict(X_test)
confusion = confusion_matrix(y_test, y_classprop_predicted) # Parameters = true and predicted labels
print("Random class-proportional prediction (dummy classifier)\n", confusion)

# Linear Kernelized SVM for comparison:
svm = SVC(kernel="linear", C=1).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print("Support vector machine classifier (linear kernel, C=1)\n", confusion)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print("Logistic regression classifier (default settings)\n", confusion)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print("Decision tree classifier (max_depth = 2)\n", confusion) # Unlike Previous classifiers, has noticeably more false negatives than positives.

"""Evaluation Metrics for Binary Classifications"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = (TP + TN) / (Total)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * Precision * Recall / (Precision + Recall) 
print("Accuracy: {:.2f}".format(accuracy_score(y_test, tree_predicted))) # Parameters = true labels vs predicted. 
print("Precision: {:.2f}".format(precision_score(y_test, tree_predicted)))
print("Recall: {:.2f}".format(recall_score(y_test, tree_predicted)))
print("F1: {:.2f}\n".format(f1_score(y_test, tree_predicted)))

# Often useful to look at all scores at once (report function.)
from sklearn.metrics import classification_report

print(classification_report(y_test, tree_predicted, target_names=["not 1", '1']))
# Last column "support" shows # of instances in test set that have label.

# Other classifiers (Dummy, SVM, Logistic Regression, Decision Tree)
print("\nRandom class-proportional (dummy)\n", classification_report(y_test, y_classprop_predicted, target_names=["not 1", '1']))
print("SVM\n", classification_report(y_test, svm_predicted, target_names=["not 1", '1']))
print("Logistic Regression\n", classification_report(y_test, lr_predicted, target_names=["not 1", '1']))
print("Decision tree\n", classification_report(y_test, tree_predicted, target_names=["not 1", '1']))

"""Decision Functions"""
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test) # Calculates probabilities of instances belonging in certain classes.
y_score_list = list(zip(y_test[:20], y_scores_lr[:20])) # Only display 1st 20 results
print("Decision Function Scoring:\t",y_score_list)

# Probability of positive class given instance:
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[:20], y_proba_lr[:20, 1])) # Show probability of positive class (,1) for 1st 20 instances
print("\nProbability Scores:\t", y_proba_list, '\n') # Scores are from 0-1

"""Precision-recall curves"""
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr) # Using logistic regression predictions
closest_zero = np.argmin(np.abs(thresholds)) # Argmin returns INDICES of min value(s). Only returns one value by default.
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero] # Pinpoints where classifier score threshold of zero was selected.

# Create figure to display PRC curve
plt.figure()
ax = plt.axes()

plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label="Precision-Recall Curve")
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle="none", c='r', mew=3) # mew=MarkerEdgeWidth
plt.xlabel("Precision", fontsize=16)
plt.ylabel("Recall", fontsize=16)
ax.set_aspect("equal")

"""ROC (Receiver Operating Characteristics) & Acre-Under curves"""
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

y_score_lr = lr.fit(X_train, y_train).decision_function(X_test) # Using same old instance of LogisticRegression classifier.
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr) # returns false_positive rate, true_positive rate, and decision thresholds
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label="LogRegr ROC curve (area = {:.2f})".format(roc_auc_lr))
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("ROC curve (1-of-10 digits classifier)", fontsize=16)
plt.legend(loc="lower right", fontsize=13)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--") # Baseline
ax.set_aspect("equal")

# SVM under different gamma settings.
from matplotlib import cm # colormap

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
for g in [0.01, 0.1, 0.20, 1]: # Different gamma values
    svm = SVC(gamma=g).fit(X_train, y_train) # train support vector classifier.
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test) # As comparison
    print("gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(g, accuracy_svm, roc_auc_svm))
    
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, label="SVM (gamma = {:.2f}, area = {:.2f})".format(g, roc_auc_svm))
print("")    
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate (Recall)", fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle="--") # Baseline. Also color=k?
plt.legend(loc="lower right", fontsize=11)
plt.title("ROC curve: (1-of-10 digits classifier)", fontsize=16)
ax.set_aspect("equal")

"""Evaluation measures for multi-class classification"""
# Multi-class confusion matrix, using nonbinary version of dataset
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

for kern in ["Linear", "RBF"]: # Compare linear against rbf    
    svm = SVC(kernel=kern.lower()).fit(X_train_mc, y_train_mc)
    svm_predicted_mc = svm.predict(X_test_mc)
    confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
    df_cm = pd.DataFrame(confusion_mc, index=[i for i in range(0, 10)], columns=[i for i in range(0, 10)]) # Index and columns represent predicted and true digit values
    
    plt.figure(figsize=(5.5, 4))
    sn.heatmap(df_cm, annot=True) # Plot confusion matrix as heatmap
    plt.title("SVM {0} Kernel \nAccuracy:{1:0.3f}".format(kern, accuracy_score(y_test_mc, svm_predicted_mc)))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

# Multi-class classification report
print(classification_report(y_test_mc, svm_predicted_mc))

# Micro- vs. macro-averaged metrics
print("\nMicro-averaged precision = {:.2f} (treats instances equally)".format(precision_score(y_test_mc, svm_predicted_mc, average="micro")))
print("Macro-averaged precision = {:.2f} (treats classes equally)".format(precision_score(y_test_mc, svm_predicted_mc, average="macro")))

print("Micro-averaged recall = {:.2f}".format(recall_score(y_test_mc, svm_predicted_mc, average="micro")))
print("Macro-averaged recall = {:.2f}".format(recall_score(y_test_mc, svm_predicted_mc, average="macro")))

print("Micro-averaged f1 = {:.2f}".format(f1_score(y_test_mc, svm_predicted_mc, average="micro")))
print("Macro-averaged f1 = {:.2f}\n".format(f1_score(y_test_mc, svm_predicted_mc, average="macro")))

"""Regression evaluation metrics"""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

diabetes = load_diabetes()

X = diabetes.data[:, None, 6]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy="mean").fit(X_train, y_train)

y_predict = lm.predict(X_test)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

print("Linear model, coefficients: ", lm.coef_)
print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test, y_predict_dummy_mean)))
print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
print("r2_score (Linear Model): {:.2f}\n".format(r2_score(y_test, y_predict)))

# Plot outputs
plt.figure()
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_predict, color="green", linewidth=2)
plt.plot(X_test, y_predict_dummy_mean, color="red", linestyle="dashed", linewidth=2, label="dummy")

"""Model selection using evaluation metrics"""
# Cross-validation example (no parameter tuning, just simply evaluating using different metrics across multiple folds)
from sklearn.model_selection import cross_val_score

dataset = load_digits()
# Make into binary problem with "digit 1" as + class and all else as negative class.
X, y = dataset.data, dataset.target == 1
clf = SVC(kernel="linear", C=1) # Remember, for cross-validation, don't need to split dataset.

# Accuracy = default scoring metric
print("Cross-validation (Accuracy)", cross_val_score(clf, X, y, cv=5))
# Use AUC as scoring metric
print("Cross-validation (AUC)", cross_val_score(clf, X, y, cv=5, scoring="roc_auc"))
# Use recall as scoring metric
print("Cross-validation (Recall)", cross_val_score(clf, X, y, cv=5, scoring="recall"), '\n')

# Grid Search e.g.
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = SVC(kernel="rbf")
grid_values = {"gamma": [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# Default metric to optimize over grid parameters: Accuracy
grid_clf_acc = GridSearchCV(clf, param_grid=grid_values).fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print("Test set Accuracy: ", accuracy_score(y_test, grid_clf_acc.predict(X_test)))
print("Grid best parameter (max. Accuracy): ", grid_clf_acc.best_params_)
print("Grid best score (Accuracy): ", grid_clf_acc.best_score_, '\n')

# Default metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid=grid_values, scoring="roc_auc").fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)

print("Test set AUC: ", roc_auc_score(y_test, y_decision_fn_scores_auc))
print("Grid best parameter (max. AUC): ", grid_clf_auc.best_params_)
print("Grid best score (AUC): ", grid_clf_auc.best_score_, '\n')

# Eval metrics supported for Model Selection:
from sklearn.metrics.scorer import SCORERS
print(sorted(list(SCORERS.keys())))

"""Two-feature classification example using digits dataset"""
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot as pcrs

"""
Create 2-feat. input vector matching example plot above
Jitter points (Add small random noise) in case there're areas in feature space where many instances have same features
"""
jitter_delta = 0.25
X_twovar_train = X_train[:, [20, 59]] + np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twovar_test = X_test[:, [20, 59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel="linear").fit(X_twovar_train, y_train)
grid_values = {"class_weight":["balanced", {1:2}, {1:3}, {1:4}, {1:5}, {1:10}, {1:20}, {1:50}]}

plt.figure(figsize=(9, 6))
for i, eval_metric in enumerate(("precision", "recall", "f1", "roc_auc")):
    grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric).fit(X_twovar_train, y_train)
    print("\nGrid best parameter (max. {0}): {1}".format(eval_metric, grid_clf_custom.best_params_))
    print("Grid best score (max. {0}): {1}".format(eval_metric, grid_clf_custom.best_score_))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    pcrs(grid_clf_custom, X_twovar_test, y_test, None, None, None, plt.subplot(2, 2, i+1)) # Inserts 4 subplots into figure
    plt.title(eval_metric + "-oriented SVC")
plt.tight_layout()
print("")

"""
Observations: Precision favours reducing false positives thus top shifts to right, recall is opposite thus shift to left.
f1 is harmonic mean of previous 2, so it represents that. Plus the optimal weight between previous 2 as well.
AUC appears to slightly favour precision more in this case.
"""

# Precision-recall curve for default SVC classifier (with balanced class weights)
from adspy_shared_utilities import plot_class_regions_for_classifier as pcr

# Create 2-feature input vector matching example plot above
jitter_delta = 0.25
X_twovar_train = X_train[:, [20, 59]] + np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twovar_test = X_test[:, [20, 59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel="linear", class_weight="balanced").fit(X_twovar_train, y_train)

y_scores = clf.decision_function(X_twovar_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

pcr(clf, X_twovar_test, y_test)
plt.title("SVC, class_weight = \"balanced\", optimized for accuracy")

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.title("Precision-recall curve: SVC, class_weight = \"balanced\"")
plt.plot(precision, recall, label="Precision-Recall Curve")
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle="none", c='r', mew=3)
plt.xlabel("Precision", fontsize=16)
plt.ylabel("Recall", fontsize=16)
ax.set_aspect("equal")
print("At zero threshold, precision: {:.2f}, recall: {:.2f}".format(closest_zero_p, closest_zero_r))

