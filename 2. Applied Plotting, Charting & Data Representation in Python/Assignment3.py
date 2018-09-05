# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:21:26 2018

@author: Jeff
"""

# Use the following data for this assignment:

import pandas as pd
import numpy as np
import scipy.stats as st # For confidence interval calculation.
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For the color gradient.
import matplotlib.colors as col # For normalizing.
from matplotlib.widgets import Slider


from IPython import get_ipython
ipy=get_ipython()
ipy.magic("matplotlib qt5")

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                   index=[1992,1993,1994,1995])

df2 = pd.DataFrame(df.apply(np.mean, axis=1), columns=["Mean"]) # Apply mean aggregate function to all column elements of rows, and store in new dataframe.

# Confidence calculated for normal dist. 
temp = df.apply(lambda a: st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a)), axis=1) # 95% confidence interval calculation.

# Assign color bar values.
colors = []
value0 = 40000

""" Easiest method
for i in range(4):
    if value <= temp.values[i][0]:
        colors.append("red")
    elif value >= temp.values[i][1]:
        colors.append("blue")
    else:
        colors.append("white")        
        #colors.append("white")
"""
cmap = cm.get_cmap("seismic").reversed()

for i in range(4):
    norm = col.Normalize(vmin=temp.values[i][0], vmax=temp.values[i][1], clip=False)
    colors.append(cmap(norm(value0)))

df2["-Conf"], df2["+Conf"] = ([np.abs(temp.values[i][0]-df2.iloc[i]).values for i in range(4)],# List comprehend the minimum values of resultant tuple. 
                             [np.abs(temp.values[i][1]-df2.iloc[i]).values for i in range(4)]) # Converting to relative confidence intervals requires subtracting mean from them.

bars = plt.bar(df2.index.astype(str), df2["Mean"], yerr=[df2["-Conf"], df2["+Conf"]], capsize=20, width = 0.75, color=colors, edgecolor="white", alpha=0.7)
#Yerr takes in error values and displays them as error bars.
plt.subplots_adjust(bottom=0.25)

# Slider Widget.
yaxis = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor="#ffd1df") #Specify position [xpos, ypos, xlen, ylen] and color of slider (light pink).
saxis = Slider(yaxis, 'Value', 0, 52000, valinit=value0)


def update(val): # Change colors of bars.
    value = saxis.val 
    for i in range(4):
        norm = col.Normalize(vmin=temp.values[i][0], vmax=temp.values[i][1], clip=False)
        bars[i].set_facecolor(cmap(norm(value)))
        
saxis.on_changed(update)