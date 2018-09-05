# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:34:09 2018

@author: Jeff
"""

"""
To complete this assignment, create a code cell that:

Creates a number of subplots using the pyplot subplots or matplotlib gridspec functionality.
Creates an animation, pulling between 100 and 1000 samples from each of the random variables (x1, x2, x3, x4) for each plot and plotting this as we did in the lecture on animation.
Bonus: Go above and beyond and "wow" your classmates (and me!) by looking into matplotlib widgets and adding a widget which allows for parameterization of the distributions behind the sampling animations.
Tips:

Before you start, think about the different ways you can create this visualization to be as interesting and effective as possible.
Take a look at the histograms below to get an idea of what the random variables look like, as well as their positioning with respect to one another. This is just a guide, so be creative in how you lay things out!
Try to keep the length of your animation reasonable (roughly between 10 and 30 seconds).
"""

from IPython import get_ipython
ipy = get_ipython()
ipy.magic("matplotlib qt5")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani
import math as m
from matplotlib.widgets import Button

# samp = np.random.randint(100, 1001)
samp = 1000
norm, gamm, expo, unif = (np.random.normal(loc=5, scale=1, size=samp),
                         np.random.gamma(shape=2, scale=1.5, size=samp),
                         np.random.exponential(2, samp),
                         np.random.uniform(0, 10, samp))

fnorm = lambda x: (2*m.pi)**(-0.5) * m.e**(-(x-5)**2 / 2) # Standard Normal Distribution function (a=1).
fgamm= lambda x: ((1/1.5)**2 * x * m.e ** (-x/1.5)) / m.gamma(2)
fexpo = lambda x: (1/2)*m.e**(-0.5*x)
funif = lambda x: 1/10

# Defined list of variable elements
var = [norm, gamm, expo, unif]
col = ["red", "green", "cyan", "black"]
title = ["Normal", "Gamma", "Exponential", "Uniform"]
bins_ = [np.arange(1, 9, 0.2), np.arange(0, 18, 0.4), np.arange(0, 18, 0.4), np.arange(0, 10, 0.2)]
axlen = [[0.75, 9.25, 0, 0.6],[-0.5, 18.5, 0, 0.6],[-0.5, 18.5, 0, 0.6],[-0.25, 10.25, 0, 0.6]]
funcs = [fnorm, fgamm, fexpo, funif]

fig, ((x1, x2), (x3, x4)) = plt.subplots(2, 2)
axs = [x1, x2, x3, x4]

"""
# Instantly Plot all graphs.
for i in range(len(axs)): # Easy way to plot all graphs.
    axs[i].hist(var[i], bins=bins_[i], color=col[i])   
    axs[i].set_title(title[i])
"""

# TODO: Make animations faster, and add widgets.

clicked = False

def click(event):
    global clicked
    clicked = False if clicked else True

def update(curr):
    if curr == samp:
         a.event_source.stop() 
         
    for i in range(len(axs)):
        axs[i].cla()
        axs[i].hist(var[i][:curr], bins=bins_[i], color=col[i], alpha=0.5, density=True)
        axs[i].axis(axlen[i])
        
        if clicked: # This interacts with the button widget
            axs[i].plot(bins_[i], list(map(funcs[i], bins_[i])))
        
        axs[i].set_title(title[i], fontsize=20)
        axs[i].tick_params(which="both", # Apply to both major and minor axes (default="major")
                           labelbottom=False, bottom=False) # Remove tick labels.
        
        for spine in axs[i].spines.values(): # Remove borderlines along graphs.
            spine.set_visible(False)
            
    axs[0].annotate("Frame = {}".format(curr), [2, 0.55]) # Displays current iteration on first plot axis
    axs[0].set_ylabel("Frequency", fontsize=14) # This is set on the top-left graph.
    
a = ani.FuncAnimation(fig, update, interval=200, frames=range(0,1001,20))

# If you add this after animation it won't be overwritten by the animation.
baxis = plt.axes([0.5, 0, 0.25, 0.075]) # Position the button.
button = Button(baxis, "Parameterizations", color="grey", hovercolor="white") # Instantiate the button GUI object
button.on_clicked(click) # Connect to the button.

"""
# Save to file:
import matplotlib as mpl
mpl.use("Agg")
write = ani.FFMpegWriter(fps=1, metadata=dict(artist="Jeff Ma"), bitrate=1800)
a.save("graph.mp4", writer=write)
"""
