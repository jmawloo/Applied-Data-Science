# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:48:40 2018

@author: Jeff
"""
from IPython import get_ipython # Used to call magic commands for IPython.
ipython = get_ipython()
ipython.magic("matplotlib inline") # Run this to change the way graphics are processed in the script. This enables animations, but forces all figures to be displayed in  new windows


import matplotlib as mpl
print(mpl.get_backend()) # How they do it in Java.

import matplotlib.pyplot as plt
plt.plot(4,3,'.') # Creates a plot with point on it.

# Creating a plot without the scripting layer:
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure() # Creates new figure
canvas = FigureCanvasAgg(fig) # Assoc. fig with backend.
ax = fig.add_subplot(111) # Add subplot to figure (create only one of them, 111)
ax.plot(3, 2, '.') # Plot the point.

canvas.print_png("test.png") # Save the plot to a file. 

# Can print using html magic by: %%HTML; <img src='test.png' />

plt.figure() # Create new figure
plt.plot(2, 3, 'o') # There's also a circle marker
ax = plt.gca() # Gets the current axes of the plot
# plt.gcf() gets the current figure and does things with that.
ax.axis([0, 6, 0, 10]) # Set axis properties. [xmin, xmax, ymin, ymax]

plt.figure()
plt.plot(2.5, 2.5, 'o')
plt.plot(3, 3, 'o')
plt.plot(1, 1, 'o')

ax = plt.gca()
print(ax.get_children()) # Gets the different child objects that make up the figure. Prints out first for some reason, hmm...
"""
Line2D objects represent the points.
Spine objects used to represent actual renderings of frame border (tic markers, 2 axis objects, and text)
Rectangle draws the background of it.
"""

# Creating Scatter plots:
import numpy as np
plt.figure() # Don't forget this else previous data isn't replaced
a = np.array([i for i in range(1,100,6)])
b = a

# How to change the colors:
colors = ["green"] * (len(a) - 2)
colors.append('red')
colors.append('yellow')

plt.scatter(a, b, s=100, c=colors) # Set size of datapoints to 100 and set colors to correspond to the array.

# Using python3's built-in zip function: 
zip_gen = zip([1,2,3,4,6], [5,7,8,9,0])
print(list(zip_gen)) # Storing data points in tuples is common.

#You can also unpack the generator:
# x,y = zip(*zip_gen) # Need an asterisk to indicate list of extra args for the zipper. Doesn't work tho. 

zip_gen = zip([1,2,3,4,6], [5,7,8,9,0])
x,y = zip(*zip_gen) # You can only iterate thru iterators ONCE.
print(x, y) # This converts tuples back into raw x and y data.

# Labelling Data Series.
plt.figure()
plt.scatter(x[:2], y[:2], s=100, c='orange', label='non-midgets')
plt.scatter(x[2:], y[2:], s=100, c='black', label='midgets')

#Can also give titles to axes and entire graph.

plt.xlabel("Efficiency")
plt.ylabel("Number of hours wasted debugging")
plt.title("Efficiency vs. number of wasted debugging hours xdddd")
plt.legend(loc=3, frameon=False, title='LEGEND FAM') # Shows legend. the loc keyword specifies location relative to graph (quadrant #.)

print(plt.gca().get_children()) # get children of every figure element, including legend.
legend = plt.gca().get_children()[-2]
print(legend.get_children()[0].get_children()[1].get_children()[0].get_children())  # HPacker objects.

from matplotlib.artist import Artist

def rec_gc(art, depth=0): # REcursively check if object is artist or not by testing artist membership of child oobjects.
    if isinstance(art, Artist): # If 'art' is an instance of "Artist"
        print(" " * depth + str(art)) # Depth just handles formatting in this case
        
        for child in art.get_children():
            rec_gc(child, depth+2)
            
rec_gc(legend)

#Line plots:
plt.figure()
lin = np.array([i for i in range(10)])
quad = lin**2 # This is called broadcasting.
plt.plot(lin, '-o', quad, '-o') # Both use dots and draw lines
plt.plot([22,33,44], '--r') # Dashes, with no points.
plt.plot([20,21,22,23,24], '-s') # This one uses square points
plt.xlabel("AYYYYYY")
plt.ylabel("LMAOOOOOO")
plt.legend(["Linear", "Quad", "Dashes", "Squares"], title="Legend") 

plt.gca().fill_between(range(len(lin)), # get current axes, then fill a color within certain xrange that has to match the length of the next set of data values.
                       lin, quad, # I'm guessing lower and upper bounds.
                       facecolor="blue", # Color of fill
                       alpha=0.25) # Remember this deals with transparency
# fill_between often used in error analysis and standard deviation.

# For datetime plotting:
plt.figure()
date = np.arange('2017-04-12', '2017-04-22', dtype="datetime64[D]") # Create list of important dates in datetime64 format (in terms of days)

plt.plot(date, lin, '-o', date, quad, '-o') # need them to be the same dimensions in order to graph properly.
# In the course it said matplotlib doesn't understand numpy, but it does currently.

"""
# Can use pandas to convert datetimes from numpy so they can be used by matplotlib.
import pandas as pd

date = list(map(pd.to_datetime, date)) # Need to convert an iterator into list first, then use it.
rec_gc(plt.gca().get_children()[4]) will get children of the x-axes object which is formatted weirdly.

"""

x = plt.gca().xaxis # This is an attribute that gives the x-axis.

for item in x.get_ticklabels(): # Get the date labels
    item.set_rotation(45) # And rotate them 45 degrees. Iterations help with formatting.
    
plt.subplots_adjust(bottom=0.25) # If you can't see the graph labels.

ax = plt.gca() # Another way of referring to the same figure.
ax.set_xlabel("Gay")
ax.set_ylabel("How long can the text of an axis title even be? I mean it's sometimes not practical to include only a bit of information, so I'm just including as much info as I can.")
ax.set_title("Quadratic ($x^2$) vs. Linear ($\\frac{x}{y}$)") # LaTeX formats math equations that are between $$.

# Bar Charts
plt.figure()
xvals = range(len(lin)) # Converts it back to a regular list I guess/
plt.bar(xvals, lin, width = 0.3, color='red') # Fire bars

new_xvals = []
for item in xvals:
    new_xvals.append(item + 0.3) # This just shifts the x-values by 0.3, which is the width of the bars.
    
plt.bar(new_xvals, quad, width=0.3, alpha=0.5)

# Creating list of error values:
from random import randint
lin_err = [randint(0, 15) for i in range(len(lin))]
plt.bar(xvals, lin, width=0.3, yerr=lin_err) # To display the error bars.

# Stacked Bar Charts

plt.figure()
xvals = range(len(lin))
plt.bar(xvals, lin, width=0.3, color='b') # b in this case means blue.
plt.bar(xvals, quad, width=0.3, color='r', bottom=lin) # r in this case means red. Also bottom keyword is what allows the stacking.

# Pivot bar to run horizontally instad of vertically. Just use plt.barh()
plt.figure()
plt.barh(xvals, lin, height=0.3, color="black") # Change width to height.
plt.barh(xvals, quad, height=0.3, color="brown", left=lin) # change bottom to left


# This is how to practice dejunkifying

plt.figure()
languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# TODO: change the bar colors to be less bright blue
# TODO: make one bar, the python bar, a contrasting color
bar = plt.bar(pos, popularity, align='center', color='lightslategrey', linewidth=0) # This gives cyan color for other bar colors
bar[0].set_color('g') # This gives grey shade for Python, and is a NoneType

# To directly label the bars:

for i in bar:
    plt.gca().text(i.get_x() + i.get_width()/2, i.get_height()-5, str(int(i.get_height())) + '%', ha='center', color='white', fontsize=11) # This is how u insert labels.

# TODO: soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8) # can use alpha transparency to cheat
# plt.ylabel('% Popularity', alpha=0.8) # don't need this anymore
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# removes all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(which="both", # Apply to both major and minor axes (default="major")
                 bottom=False, # Remove bottom ticks
                 left=False, 
                 labelleft=False) # Remove left labels.

# TODO: remove the frame of the chart
for spine in plt.gca().spines.values(): # This is how to access the spine objects
    spine.set_visible(False)
plt.show()

