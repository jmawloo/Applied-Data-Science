# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:42:41 2018

@author: Jeff
"""

"""
In this assignment you'll explore the relationship between model complexity and generalization performance, 
by adjusting key parameters of various supervised learning models. 
Part 1 of this assignment will look at regression and Part 2 will look at classification.
"""

"""   ===PART 1: REGRESSION===   """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython import get_ipython as ipy
import matplotlib.pyplot as plt
ipy().magic("matplotlib qt5")

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label="training data")
    plt.scatter(X_test, y_test, label="test data")
    plt.legend(loc=4);
    
    
part1_scatter()


"""Question 1:
Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees
1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and 
then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 
(e.g. np.linspace(0,10,100)) and store this in a numpy array. The first row of this array should correspond 
to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.    
"""
def answer_1():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures    
    
    answer = np.empty((4,100))
    for i, deg in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))# Reshaped to account for 1 feature (Need 2D array).
        # X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
        linpol = LinearRegression().fit(X_train_poly, y_train) # Need poly transformed version of X_train, else it won't work
        
        X_test_poly = poly.transform(np.linspace(0, 10, 100).reshape(-1, 1)) # Do the same poly fitting and transformations for prediction set.
        y_test = linpol.predict(X_test_poly)
        
        answer[i] = y_test # y_test.flatten ensures that the test set is 1D.
        
    return answer


ans1 = answer_1()
print(ans1)

# Plot prediction results for each degree:
def plot_one(degree_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(X_train, y_train, 'o', label="training data", markersize=10) # Plot GLOBAL vars X_train and y_train.
    plt.plot(X_test, y_test, 'o', label="test data", markersize=10)
    
    for i, deg in enumerate([1, 3, 6, 9]):
        plt.plot(np.linspace(0, 10, 100), degree_predictions[i], alpha=0.8, lw=2, label="degree={}".format(deg))
    plt.ylim(-1, 2.5)
    plt.legend(loc=4)

    
plot_one(ans1)

"""Question 2
Write a function that fits a polynomial LinearRegression model on the training data
X_train for degrees 0 through 9. For each model compute the  R**2  (coefficient of 
determination) regression score on the training data as well as the the test data,
and return both of these arrays in a tuple.

This function should return one tuple of numpy arrays (r2_train, r2_test). 
Both arrays should have shape (10,)
"""
def answer_2():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score # Deprecated: "from sklearn.metrics.regression import r2_score"
    
    r2_train, r2_test = np.empty((10,)), np.empty((10,))
    for deg in range(10): # Degrees 0 thru 9 inclusive
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        
        linpol = LinearRegression().fit(X_train_poly, y_train)
        r2_train[deg] = r2_score(y_true=y_train, y_pred=linpol.predict(X_train_poly)) # R2 score compares true labels "y_train" with predicted labels. Score can also be negative if ur model gey
        r2_test[deg] = r2_score(y_true=y_test, y_pred=linpol.predict(X_test_poly))
    
    return (r2_train, r2_test)


ans2 = answer_2()
print(ans2)


def plot_two(R2_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(10), R2_scores[0], alpha=0.5, lw=2, label="training data")
    plt.plot(np.arange(10), R2_scores[1], alpha=0.5, lw=2, label="test data")
    
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("$ R^2 $ Score")
    
    
plot_two(ans2)

"""Question 3
Based on the R**2 scores from question 2 (degree levels 0 through 9), 
what degree level corresponds to a model that is underfitting? 
What degree level corresponds to a model that is overfitting? 
What choice of degree level would provide a model with good generalization performance on this dataset?

Hint: Try plotting the R**2 scores from question 2 to visualize the relationship between 
degree level and R**2. Remember to comment out the import matplotlib line before submission.

This function should return one tuple with the degree values in this order: 
(Underfitting, Overfitting, Good_Generalization). There might be multiple correct solutions,
however, you only need to return one possible solution, for example, (1,2,3).
"""
def answer_3():
    """
    R2_test <= 0.7 & R2_train <=0.7: Underfitting [1]
    R2_test <=0.7 & R2_train > 0.7: Overfitting [2]
    R2 > 0.7: Good Generalization [3]
    
    In order of dimension #.
    """
    
    result = np.zeros((10,), dtype=np.uint8)# Gotta save that memory >u<
    for i in np.arange(10):
        if ans2[1][i] > 0.7: # Good
            result[i] = 3
        elif ans2[1][i] <= 0.7 and ans2[0][i] > 0.7: # Overfit
            result[i] = 2
        else: # Underfit
            result[i] = 1
    
    return tuple(result)


ans3 = answer_3()
print(ans3)

"""Question 4
Training models on high degree polynomial features can result in overly complex models that overfit, 
so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

For this question, train two models: a non-regularized LinearRegression model 
(default parameters) and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) 
both on polynomial features of degree 12. Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.

This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)
"""
def answer_4():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    
    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    
    linpoly = LinearRegression().fit(X_train_poly, y_train)
    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train) # Still gives convergencewarning
    
    LinReg_R2_test_score = r2_score(y_test, linpoly.predict(X_test_poly))
    Lasso_R2_test_score = r2_score(y_test, linlasso.predict(X_test_poly))
    
    return LinReg_R2_test_score, Lasso_R2_test_score


ans4 = answer_4()
print(ans4)

"""   ===PART 2: CLASSIFICATION===
Here's an application of machine learning that could save your life! 
For this section of the assignment we will be working with the UCI Mushroom Data Set stored in mushrooms.csv.
The data will be used to train a model to predict whether or not a mushroom is poisonous. 
The following attributes are provided:

Attribute Information:

cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
bruises?: bruises=t, no=f
odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
gill-attachment: attached=a, descending=d, free=f, notched=n
gill-spacing: close=c, crowded=w, distant=d
gill-size: broad=b, narrow=n
gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
stalk-shape: enlarging=e, tapering=t
stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
veil-type: partial=p, universal=u
veil-color: brown=n, orange=o, white=w, yellow=y
ring-number: none=n, one=o, two=t
ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d


The data in the mushrooms dataset is currently encoded with strings. 
These values will need to be encoded to numeric to work with sklearn. 
We'll use pd.get_dummies to convert the categorical variables into indicator variables.
"""

mush_df = pd.read_csv("mushrooms.csv")
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:, 2:]
y_mush = mush_df2.iloc[:, 1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

"""Question 5
Using X_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier 
with default parameters and random_state=0. What are the 5 most important features 
found by the decision tree?

As a reminder, the feature names are available in the X_train2.columns property, 
and the order of the features in X_train2.columns matches the order of the feature 
importance values in the classifier's feature_importances_ property.

This function should return a list of length 5 containing the feature names in 
descending order of importance.

Note: remember that you also need to set random_state in the DecisionTreeClassifier.

"""
def answer_5():
    from sklearn.tree import DecisionTreeClassifier
    
    tree = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    features = pd.DataFrame([tree.feature_importances_, X_train2.columns], index=["Importance", "Labels"]).T
    
    important_features = features.sort_values("Importance", ascending=False, kind="mergesort").iloc[:5]
    return np.array(important_features["Labels"])


ans5 = answer_5()
print(ans5)


"""Question 6
For this question, we're going to use the validation_curve function in sklearn.model_selection 
to determine training and test scores for a Support Vector Classifier (SVC) with varying 
parameter values. Recall that the validation_curve function, in addition to taking 
an initialized unfitted classifier object, takes a dataset as input and does its own 
internal train-test splits to compute results.

Because creating a validation curve requires fitting multiple models, for performance 
reasons this question will use just a subset of the original mushroom dataset: 
please use the variables X_subset and y_subset as input to the validation curve function 
(instead of X_mush and y_mush) to reduce computation time.

The initialized unfitted classifier object we'll be using is a Support Vector Classifier 
with radial basis kernel. So your first step is to create an SVC object with default parameters 
(i.e. kernel='rbf', C=1) and random_state=0. Recall that the kernel width of the RBF kernel is 
controlled using the gamma parameter.

With this classifier, and the dataset in X_subset, y_subset, explore the effect of gamma on 
classifier accuracy by using the validation_curve function to find the training and test scores 
for 6 values of gamma from 0.0001 to 10 (i.e. np.logspace(-4,1,6)). Recall that you can 
specify what scoring metric you want validation_curve to use by setting the "scoring" 
parameter. In this case, we want to use "accuracy" as the scoring metric.

For each level of gamma, validation_curve will fit 3 models on different subsets of the data,
returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training 
and test sets.

Find the mean score across the three models for each level of gamma for both arrays, creating 
two arrays of length 6, and return a tuple with the two arrays.

e.g.

if one of your array of scores is

array([[ 0.5,  0.4,  0.6],
       [ 0.7,  0.8,  0.7],
       [ 0.9,  0.8,  0.8],
       [ 0.8,  0.7,  0.8],
       [ 0.7,  0.6,  0.6],
       [ 0.4,  0.6,  0.5]])
it should then become

array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
This function should return one tuple of numpy arrays (training_scores, test_scores) where each array in the tuple has shape (6,).
"""
def answer_6():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    svc = SVC(kernel="rbf", C=1, random_state=0) # Validation curve will fit automatically.
    
    traintemp, testtemp = validation_curve(svc, X_subset, y_subset, scoring="accuracy", param_name="gamma", param_range=np.logspace(-4, 1, 6)) # Can conveniently specify parameter to vary in model here.
    
    train_scores = np.array(list(map(lambda x: np.mean(x), traintemp))) # To get the mean values of each set of 3 scores.
    test_scores = np.array(list(map(lambda x: np.mean(x), testtemp))) 

    return train_scores, test_scores


ans6 = answer_6()
print(ans6)


def plot_6(dataset):
    plt.figure(figsize=(10, 5))
    plt.semilogx(np.logspace(-4, 1, 6), dataset[0], alpha=0.5, lw=2, label="training set") # Make logarithmic plot
    plt.semilogx(np.logspace(-4, 1, 6), dataset[1], alpha=0.5, lw=2, label="test set")
    
    plt.xlabel("Gamma")
    plt.ylabel("Validation Curve Score")
    plt.legend()
    
    
plot_6(ans6)

"""Question 7
Based on the scores from question 6, what gamma value corresponds to a model that is underfitting 
(and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting 
(and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with 
good generalization performance on this dataset (high accuracy on both training and test set)?

Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. 
Remember to comment out the import matplotlib line before submission.

This function should return one tuple with the degree values in this order: 
(Underfitting, Overfitting, Good_Generalization) Please note there is only one correct solution.
"""
def answer_7(data): # Relies on ans6
    result = np.zeros((6, ))
    for i in range(6):
        if data[1][i] > 0.95: # Perfect
            result[i] = 3
        elif data[1][i] <= 0.95 and data[0][i] > 0.95: # Overfit
            result[i] = 2
        else:
            result[i] = 1
            
    return result


ans7 = answer_7(ans6)
