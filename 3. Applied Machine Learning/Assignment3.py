# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 23:44:06 2018

@author: Jeff
"""

"""
In this assignment you will train several models and evaluate how effectively they predict
instances of fraud using data based on this dataset from Kaggle.

Each row in fraud_data.csv corresponds to a credit card transaction. 
Features include confidential variables V1 through V28 as well as Amount 
which is the amount of the transaction.

The target is stored in the class column, where a value of 1 corresponds to an 
instance of fraud and 0 corresponds to an instance of not fraud.
"""

import numpy as np
import pandas as pd

"""Question 1:
Import the data from fraud_data.csv. What percentage of the observations in the dataset are 
instances of fraud?

This function should return a float between 0 and 1.
"""
def answer_1():
    df = pd.read_csv("fraud_data.csv")
    answer = len(df[df["Class"] == 1]) / len(df)
    return answer


ans1 = answer_1()
print(ans1, '\n')

"""For rest of questions"""
df = pd.read_csv("fraud_data.csv")
X = df.drop("Class", axis=1) # Or df.iloc[:, :-1]
y = df["Class"] # Or df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""Question 2:
Using X_train, X_test, y_train, and y_test (as defined above), 
train a dummy classifier that classifies everything as the majority class of the training data.
What is the accuracy of this classifier? What is the recall?

This function should a return a tuple with two floats, i.e. (accuracy score, recall score).
"""
def answer_2():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, recall_score
    
    dum = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_predict = dum.predict(X_test)
    
    return (accuracy_score(y_test, y_predict), recall_score(y_test, y_predict))


ans2 = answer_2()
print(ans2, '\n')

"""Question 3
Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters.
What is the accuracy, recall, and precision of this classifier?

This function should a return a tuple with three floats, i.e. (accuracy score, 
recall score, precision score).
"""
def answer_3():
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    
    svc = SVC().fit(X_train, y_train)
    y_predict = svc.predict(X_test)
    
    return (accuracy_score(y_test, y_predict), recall_score(y_test, y_predict), precision_score(y_test, y_predict))


ans3 = answer_3()
print(ans3, '\n')

"""Question 4:
Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is the 
confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.

This function should return a confusion matrix, a 2x2 numpy array with 4 integers.
"""

def answer_4():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    svc = SVC(C=1e9, gamma=1e-7).fit(X_train, y_train)
    decision = svc.decision_function(X_test)
    
    y_predict = svc.predict(X_test[decision >= -220]) # Setting -220 as decision threshold
    y_test2 = y_test[decision >= -220] # Do same for y_test
    confusion = confusion_matrix(y_test2, y_predict)
    
    return confusion


ans4 = answer_4()
print(ans4, '\n')

"""Question 5:
Train a logisitic regression classifier with default parameters using X_train and y_train.

For the logisitic regression classifier, create a precision recall curve and a roc curve 
using y_test and the probability estimates for X_test (probability it is fraud).

Looking at the precision recall curve, what is the recall when the precision is 0.75?

Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?

This function should return a tuple with two floats, i.e. (recall, true positive rate).
"""

def answer_5():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    
    lr = LogisticRegression().fit(X_train, y_train) # predict_proba is classifier method.
    y_proba = lr.predict_proba(X_test)[:, 1] # [:, 1]To get probability that instance is fraud
    precision, recall, _threshold = precision_recall_curve(y_test, y_proba)
    fpr_lr, tpr_lr, _thresholds = roc_curve(y_test, y_proba)
    
    rec, tpr = recall[precision == 0.75][0], np.average(tpr_lr[np.around(fpr_lr, 2) == 0.16]) # only approximate matches, average multiple results
    return rec, tpr

ans5 = answer_5()
print(ans5, '\n')

"""Question 6:
Perform a grid search over the parameters listed below for a Logisitic Regression classifier, 
using recall for scoring and the default 3-fold cross validation.

'penalty': ['l1', 'l2']

'C':[0.01, 0.1, 1, 10, 100]

From .cv_results_, create an array of the mean test scores of each parameter combination.

This function should return a 5 by 2 numpy array with 10 floats.

Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. 
You might need to reshape your raw result to meet the format we are looking for.
"""

def answer_6():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    grid_vals = {"penalty":["l1", "l2"], 'C':np.logspace(-2, 2, 5)}
    lr = LogisticRegression()
    grid_lr_rec = GridSearchCV(lr, grid_vals, scoring="recall", cv=None).fit(X_train, y_train) #cv=None uses default 3-fold
    result = np.array(grid_lr_rec.cv_results_["mean_test_score"]).reshape(5, 2)
    
    return result


ans6 = answer_6()
print(ans6, '\n')

# Following function helps visualize results from gridsearch
def GridSearch_Heatmap(scores):
    from IPython import get_ipython as ipy
    ipy().magic("matplotlib qt5")
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure()
    sns.heatmap(scores, xticklabels=["l1", "l2"], yticklabels=np.logspace(-2, 2, 5)) # Assuming scores is in its proper shape.
    plt.yticks(rotation=0)


GridSearch_Heatmap(ans6)