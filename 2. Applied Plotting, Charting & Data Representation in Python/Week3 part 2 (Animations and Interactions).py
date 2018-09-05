# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:38:41 2018

@author: Jeff
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython import get_ipython # Used to call magic commands for IPython.
ipython = get_ipython()
# ipython.magic("matplotlib inline") # Run this to change the way graphics are processed in the script. This enables animations, but forces all figures to be displayed in  new windows
ipython.magic("matplotlib qt5") # Run this to change the way graphics are processed in the script.

"""
INTERACTIVITY: Similar to animation, but need to reference canvas object of current figure.
    - Canvas obj handles all drawing events and is tightly connected with back end.
    - Event-driven programming: many programmers base software  off of this concept.
"""

fig, ax = plt.subplots()
data = np.random.rand(10)
ax.plot(data)

def onclick(event): # User interaction mouseclick event.
    plt.cla() # Clears the axes.
    ax.plot(data)
    ax.set_title("Event at pixels ({} , {})\n and data [{}, {}]".format(event.x, event.y, event.xdata, event.ydata)) # Set a title describing variance of mouse.
    
# This part is called "Wiring it up"/
fig.canvas.mpl_connect("button_press_event", onclick) # This will call the onclick function properly.    
# Need to use the drag/zoom buttons to actually interact with figure.
"""
Pick event: Lets you do things with graph when user clicks on specific chart element.
    - With Spyder, only works if u click a part of the diagram, then use one of the tools to click on diagram again.
"""
from random import shuffle
origins = ["Earth", "Tatooine", "Gallifrey", "Middle of Nowhere", "Asgardia", "Mars", "Minecraftia", "Afterlife", "Planet X", "Dimension C-657"]
shuffle(origins)

df = pd.DataFrame({"height":np.random.rand(10), "weight":np.random.rand(10), "origin":origins})

plt.figure()
plt.scatter(df["height"], df["weight"], picker=5)# informs backend that click can be up to 5 pixels away from the closest object.
plt.gca().set_xlabel("Height")
plt.gca().set_ylabel("Weight")
# Now defining the pick event (different data from mouse event:)
def onpick(event):
    origin = df.iloc[event.ind[0]]["origin"] # Has index value corresponding to "our" index and dataframe.
    plt.gca().set_title("Selected item came from {}.".format(origin))

plt.gcf().canvas.mpl_connect("pick_event", onpick) # To call the function.



"""
ANIMATIONS AND INTERACTIVITY:
    - Require lots of backend interaction.
    -Matplotlib notebook magic function provides some interactivity.

matplotlib.animation module has important helpers to build animations.
    - Call FuncAnimation(), where function is user-defined.
"""
# For some reason can't run this animation block first, else it screws things up. Just wait until animation's done before interacting with other graphs.
import matplotlib.animation as animation

n = 1000 # Cutoff for animation
x = np.random.randn(n) # Returns n samples from standard normal dist.

def update(curr):
    if curr == n: # Tells the animation to stop at n.
        a.event_source.stop() # Will call anumation "a". Tells the animation's event source to stop.
        
    plt.cla() # Clears current axes.
    bins_ = np.arange(-4, 4, 0.5) # Ensures that the bins will not change size, and assume even spacing
    plt.hist(x[:curr], bins=bins_)
    plt.axis([-4, 4, 0, 30]) # To prevent autoscaling thru each animation.
    
    # add titles.
    plt.gca().set_title("Sampling Normal Distribution")
    plt.gca().set_ylabel("Frequency")
    plt.gca().set_xlabel("Value")
    plt.annotate("n = {}".format(curr), [3, 27]) # Displays text in a certain area of graph (right side)
    
    
fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=50) # Figure we're working with, function to use, milliseconds between each iteration.
# Can export animations to file using third parties like FFM page

