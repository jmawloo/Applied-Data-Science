# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:13:09 2018

@author: Jeff
"""

"""PANDAS VISUALIZATION
Pandas uses Matplotlib "under the hood" to display visualizations.
    - Matplotlib has different styles that can be used for plotting things.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from IPython import get_ipython as get
get().magic("matplotlib qt5")


# Changing Pandas' style
print(plt.style.available) # Prints all the styles out
plt.style.use("seaborn-colorblind")


# Creating and plotting DataFrame
np.random.seed(123)

df = pd.DataFrame({'A':np.random.randn(365).cumsum(0),# Cumulative sum accumulates the random results.
                   'B':np.random.randn(365).cumsum(0)+20,
                   'C':np.random.randn(365).cumsum(0)},
                   index=pd.date_range('1/1/2017', periods=365)) #Set days of every year in 2017 as index 
print(df.head())
df.plot();  # Simple wrapper around plt.plot. Semicolon supresses unwannted matplotlib output.

"""
for i in plt.style.available: # iterate through all available styles.
    plt.style.use(i)
    df.plot();
"""
df.plot('A','B', kind="scatter") # Can also specify kind of graph to plot.
ax = df.plot.scatter('A','C',c='B',s=df['B'], colormap="ocean") # Specify graph type directly. Also make dots vary in size. Column B determines color and sizes of dots.
# Since object is returned, can perform additional method operations to scatter plot.

ax.set_aspect("equal") #Setting equal aspect ratio allows viewer to see range of A as much smaller than range of C.

# Other plots:
df.plot.box();
df.plot.hist(alpha=0.7);

"""KERNEL DENSITY ESTIMATE PLOT [kde()]:
    Useful for visualizing estimate of variable's density function.
    Use when deriving smooth continuous function from given sample.
"""
df.plot.kde();

# Pandas.rools.plotting
"""
Can do many things with these tools, including 3D visualizations.

SCATTER Matrix: Way of comparing each column in dataframe to other columns in a pair-wise fashion.
Parallel Coordinates Plots: Common way of  visualizing high-dimensional multivariate data
    - Each variable in dataset corresponds to equally-spaced parallel line.
    - Values of each variable then connected by line for each observation made.
    - Coloring by class allows viewer to more easily see patterns or clustering.
"""

iris = pd.read_csv("iris.csv")
print(iris.head())

# pd.tools.plotting.scatter_matrix(iris); Deprecated
pd.plotting.scatter_matrix(iris);
plt.figure() # Don't forget this if plotting function is not method of class
pd.plotting.parallel_coordinates(iris, 'Name');



# SEABORN
"""
Wraparound matplotlib. Basically makes process of visualization and creating certain matplotlib plots much easier.
    -kdeplot: Plots density function for given dataset.
    -distplot: Plots histogram AND density function.
    -jointplot: Creates scatterplot along histograms for each individual variable on each axis.
        - Allows for easy comparison of density of 2 variables.
"""

import seaborn as sns # Just importing this by itself will change the style if no other style has been set.

np.random.seed(1234)
v1 = pd.Series(np.random.normal(0,10,1000), name="v1") # (mean = 0, std = 10, samples = 1000)
v2 = pd.Series(2*v1 + np.random.normal(60, 15, 1000), name="v2")

plt.figure()
plt.hist(v1, alpha=0.7, bins=np.arange(-50, 150, 5), label="v1")
plt.hist(v2, alpha=0.7, bins=np.arange(-50, 150, 5), label="v2") # Fixed bins useful for plotting graphs in same figure.
plt.legend();

plt.figure()
plt.hist([v1, v2], histtype="barstacked", density=True) # Normalize distribution curves and stack overlapping bars
v3 = np.concatenate([v1, v2]) # can also use immutable tuple. Joins the two series datasets.
sns.kdeplot(v3) # Combined density functions of V1 and V2.

#Do it all in one command:
plt.figure()
sns.distplot(v3, hist_kws={"color":"Teal"}, kde_kws={"color":"Navy"})

# Jointplot (from week 2: )
plt.figure()
grid = sns.jointplot(v1, v2, alpha=0.4) # Normally dist. variables with positive correlation.

"""
Can tweak plots generated using matplotlib tools.
    - Some plotting functions return matplotlib object, while others are figures interacting with multiple panels and return Seaborn grid object.
"""
grid.ax_joint.set_aspect("equal") # get axis subplot, then set aspect ratio.

"""
Hexbin plots: "bivariate counterpart" to histograms.
    -Number of observations falling into hexagonal bins. Works well with relatively larger datasets.
    
Kernel Density Estimation plots are another good way of visualizing data.
    -2-dimensional KDE plots are continuous versions of hexbin jointplots.
"""
# Don't need to manually create new figure for Seaborn :)
sns.jointplot(v1, v2, alpha=0.4, kind="hex") # use hexplots.

# Turn off gray grids in following charts.
sns.set_style("white")
sns.jointplot(v1, v2, alpha=0.4, kind="kde", space=0) # Space plots marginal dist histograms directly on borders of scatterplot.

# Seaborn also has builtin function that creates scatterplot matrix
sns.pairplot(iris, hue="Name", diag_kind="kde") # pass in dataframe, map name categories to different colors, use kde along diagonals instead of default histograms.
#Pairplot can be very useful in exploratory data analysis.

"""
Violin plot; more informative version of boxplot; can convey info such as multimodality.
Swarm plot: scatter plot for categorical data.
"""

plt.figure(figsize=(12,8))
plt.subplot(121) #1 row occupied, 2 columns occupied, start at index 1(left)
sns.swarmplot("Name", "PetalLength", data=iris); # Labels, data
plt.subplot(122)#1 row, 2 columns, start at index 2 (right)
sns.violinplot("Name", "PetalLength", data=iris); # NSFW lol
# Wider areas indicate more common values, like histogram.
