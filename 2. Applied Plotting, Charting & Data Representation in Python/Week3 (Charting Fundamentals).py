# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:24:21 2018

@author: Jeff
"""

import matplotlib.pyplot as plt
import numpy as np


# Subplots: Display different graphs on specified number of rows and columns.
plt.figure()
plt.subplot(1, 2, 1) # Number of rows, number of columns, axis to start in (left-hand side).
linear_data = np.array([i for i in range(1,9)])
plt.plot(linear_data, '-o')

quad_data = linear_data**2 # Broadcast urself.
plt.subplot(1, 2, 2) # Now display it on the right side.
plt.plot(quad_data, '-o', c='r')

# You can call back previous graphs anywhere in your code.
plt.subplot(1, 2, 1) # Oh damn, deprecation warning (future = new instance always created and returned.)
plt.plot(linear_data**3, '-x', c='g') # This will overlap a cubic function on the 1st graph

"""
Problem: putting more data on the same graph also changes the y-axes values, potentially skewing the
graph and misleading the reader. 

Solution: "Share" the x and y-axes.
"""

plt.figure()
ax1 = plt.subplot(1, 2, 1) # The last parameter starts its COUNT at 1, and NOT 0.
plt.plot(linear_data, '-o')
plt.subplot(1, 2, 2, sharey=ax1)
# Unlike previous subplot, don't actually have to store in another variable. Auto gca()'s and stores it as last object for future plot function calls. 
plt.plot(quad_data, '-x') #y-axis is now shared.

# Remember:
plt.figure() # Creates new figure.
print(plt.subplot(111) == plt.subplot(1, 1, 1)) # But still use commas since the 1st one doesn't allow double digits.

# Plt.subplots allows you to create multple figure axes at once, and can unpack those axes to different variables to modify each one of them.
fig, ((a1,a2,a3),  (a4,a5,a6), (a7,a8,a9)) = plt.subplots(3, 3, sharex=True, sharey=True)
a5.plot(linear_data, '-') # Method of plotting only has labels for very left OR very bottom figures
a4.plot(quad_data, '-', c='r') # The inner (a4,a5,a6) parantheses describe the column element, and the tuple as a whole describes the row it's on.
plt.gca().axis([-0.5, 8.5, 0, 20]) # You can rescale everything to not skew data this way.


# To enable labels, iterate thru axes:
for ax in plt.gcf().get_axes():
    """
    ax.tick_params(reset=True) # Need to reset them for it to work.
    for label in ax.get_xticklabels() + ax.get_yticklabels(): # Joining 2 lists and iterating thru joined list.
        label.set_visible(True)
    """
    # Alternatively:
    ax.tick_params(labelbottom=True, labelleft=True)
    
plt.gcf().canvas.draw() # Force redraw of figure canvas to see the newly displayed labels, if they don't show up.

"""HISTOGRAM: Bar chart showing frequency of data.
e.g. probability dist. 
 - Probability Density Function: y-axis is probability that event occurs under value, and x-axis is value itself. Y between 0 and 1 (impossibility and certainty)
Normal dist = standard deviations.
Sampling = pulling out value from distribution.
"""

plt.figure()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True) # Intentionally let graphs have different y-axis parameters. Also need outer brackets.
axes = [ax1, ax2, ax3, ax4]
for n in range(0, len(axes)):
    samp_size = 10**(n+1) # Increase size by powers of 10 thru each iteration.
    sample = np.random.normal(loc=0.0, scale=1.0, size=samp_size) # Location on x-axes of mean, standard deviation from mean, sample size.
    # axes[n].hist(sample) # plots a histogram. 10 bins to sort values in by default, so bins get wider as sample size increases due to outliers.
    axes[n].hist(sample, bins=100) # Plots much smoother histograms.
    axes[n].set_title('n={}'.format(samp_size))  #Use set_title, don't just use title!
    
"""
Question: How many bins to use? 
    -For a more coarse and general understanding of data, use amount of bins corresponding to size of data needed for proper decision making.
    -Too little bins provides no information (coarse grain granularity), whereas too much also provides no info (fine grain granularity)
    - e.g. when using 10000 bins no longer displays trends between samples; displays size of sample themselves.
    Similar to using aggregate statistics to describe population sample
"""
# Scatter plot. Y-values come from normal dist, x-values come from random dist.
plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.scatter(X, Y) # Not clear what distributions are.


"""
GRIDSPEC: Maps axes over multiple cells in grid.
"""
plt.figure()
import matplotlib.gridspec as gridspec
gspec = gridspec.GridSpec(3, 3) # pass in elements of object directly instead of specifying row, column, and which axis for subplot. Mind where "GridSpec" is capitalized.

top_hist = plt.subplot(gspec[0, 1:]) # use slicing to specify places to cover. Also list index starts at 0.
side_hist = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])

lower_right.scatter(X, Y)
top_hist.hist(X, bins=100)
s = side_hist.hist(Y, bins=100, orientation='horizontal') # Orientation rotates the graph.

# Only care about relative values for histograms, so not iterested in how large the bins are exactly.
top_hist.clear() # Clear graph occupying that space.
top_hist.hist(X, bins=100, normed=True) # normed = Scale frequency data between 0 and 1. Deprecated tho, use 'density' keyword.
side_hist.clear()
side_hist.hist(Y, bins=100, density=True, orientation='horizontal')
side_hist.invert_xaxis() # Flip the histogram.

# To clear things up a bit, set axes ranges.
for ax in [top_hist, lower_right]:
    ax.set_xlim(0, 1) # xmin value, xmax value.
for ax in [side_hist, lower_right]:
    ax.set_ylim(-5, 5)
    
"""
BOX-AND-WHISKER PLOT: Method of showing aggregate statistics of various samples in concise manner.
    - Shows median, min and max values (The ends of the whisker), and "interquartile range" (The box)
pd.df.describe() shows the total elements in a dataset, as well as the mean, std, min and max values. It also shows "25%, 50%, 75%" or interquartile values, where most of the values  fall under.
    - For an actual plot, mean/median plotted as straight line.
    - 2 boxes are then plotted on either side of the line, representing the 25% to 75% range.
    - Capped thin lines are then drawn to the min and max values.
"""

import pandas as pd

norm_dist = np.random.normal(loc=0.0, scale=1.0, size=10000)
rand_dist = np.random.random(size=10000)
gamma_dist = np.random.gamma(shape=2, size=10000)

df = pd.DataFrame({"normal":norm_dist,
                   "random":rand_dist,
                   "gamma":gamma_dist})
    
print(df.describe()) # See summary statistics of dataframe created.

plt.figure()
_ = plt.boxplot(df["normal"], whis="range") # whis tells boxplot what to set whiskers to represent, in this case 'range' means all the way up to max/min.
# Use dummy variables to throw away unwanted return values. Need this so that the output of a million artists is not printed either, at least in the Jupyter notebook.

plt.clf() # Clears the CURRENT figure.
plt.boxplot([df["normal"], df["random"], df["gamma"]], whis="range") # Pass in multiple datasets as list of lists.

# Looking at gamma using histogram￼￼
plt.figure() 
plt.hist(df["gamma"], bins=100)

# To get overlay of graphs on top of each other, use import mpl_toolkits.axes_grid1.inset_locator as mpl_il
import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([ df["normal"], df["random"], df["gamma"] ]) # If we don't specify whis, only goes out halfway from interquartile range. Can be used to detect outliers (shown as points)
ax2 = mpl_il.inset_axes(plt.gca(), width="60%", height="40%", loc=2) # Put smaller axis within larger one, specify its dimensions, then specify its location.
ax2.hist(df["gamma"], bins=100)
ax2.margins(x=0.5) # Set autoscaling margin so that the xrange covers a larger area
# Not as flexible as gridspec; location of smaller graph limited by where the larger graph displays its data.
ax2.yaxis.tick_right() # Will display ticks on right side of smaller graph.

"""
HEATMAPS: Used to visualize 3-Dimensional data, and takes advantage of spatial proximity.
    -good heatmap e.g. = weather maps (lattitude, longitude, temp/rainfall amounts [use color to indicate intensity])
    - e.g. Australian heatmap of Malaysian Airlines missing flight.
    
* Don't use heatmaps for categorical data. Misleads viewer to look for patterns & ordering thru spatial proximity.

In Matplotlib, heatmaps = 2-D histogram where x & y indicate potential points, and color indicates frequency off observation.
"""

plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.hist2d(X, Y, bins=25) # Lighter colors generally indicate higher observational frequencies.

# Increasing the bin size
plt.figure()
plt.hist2d(X, Y, bins=100) # You get a clearer picture, but eventually everything becomes one color since  every datapoint becomes unique.
plt.colorbar() # Displays legend for heatmap
# A lot of accounting and bookkeeping happens under the hood for function to be called.

