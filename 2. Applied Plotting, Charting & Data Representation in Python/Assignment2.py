# -*- coding: utf-8 -*-
"""
Created on Sat May 26 21:03:46 2018

@author: Jeff
"""

"""An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d200/c5a5dc784c0e2b77accc1a6f594b2ff9418908154063a8391ffd53b8.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.

Each row in the assignment datafile corresponds to a single observation.

The following variables are provided to you:

* **id** : station identification code
* **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
* **element** : indicator of element type
    * TMAX : Maximum temperature (tenths of degrees C)
    * TMIN : Minimum temperature (tenths of degrees C)
* **value** : data value for element (tenths of degrees C)

For this assignment, you must:

1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.

The data you have been given is near **Scarborough, Ontario, Canada**, and the stations the data comes from are shown on the map below.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import get_ipython # Can pass in-line magic commands to this.
ipy = get_ipython()

ipy.magic("matplotlib qt5")

""" # Implementing this functionality will likely cause conflicts with the anaconda modules installed.
import mplleaflet
def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(200,'c5a5dc784c0e2b77accc1a6f594b2ff9418908154063a8391ffd53b8')
"""
ipy.magic("matplotlib inline") # Resets the graphics processing

# Part 1: Record Temp differnces, colorized:

data = pd.read_csv("c5a5dc784c0e2b77accc1a6f594b2ff9418908154063a8391ffd53b8.csv")

data = data[~(data["Date"].str.endswith(r"02-29"))] # Drop leap year dates, prevent inconsistencies. Also use raw byte strings when working with ~. & is the bitwise and.
data["Data_Value"] = data["Data_Value"].apply(lambda x: x/10) # Converts the datavalues into degrees Celsius (from tenths of degrees Celsius)
data["Year"], data["Month-Day"] = zip(*data["Date"].apply(lambda x: (x[:4], x[5:])))

data2 = data[~(data["Date"].str.startswith(r"2015"))]  # Only need 2005-2014 inclusive. Store the 2005-2--14 data in separate variable.
# The ~ symbol is like a negator; it inverts the bytes present in an integer. I used it b/c pandas won't shut up about series being ambiguous cases.

"""
data2["Year"], data2["Month-Date"] =  zip(*data2["Date"].apply(lambda x: (x[:4], x[5:]))) # Another way of separating the dates.
time = pd.DatetimeIndex(data2["Date"]) # Optional: interpret date column as Datetime format.  DayofYear returns ordinal day, based on year.
data2["Month"], data2["Day"] = time.month, time.day # Tuple assignment to isolate the month from the day.
data2.sort_values(by=["Month", "Day"], inplace=True) # Sort month and day.
"""

# Create data for 2005-2014 high and low
low = data2[data2["Element"] == "TMIN"].groupby(["Month-Day"]).agg({"Data_Value":np.min}) # remember Agg takes in a list of columns to map and their respective functions to be used during the map. Groupby groups special function returns with selected column.
high = data2[data2["Element"] == "TMAX"].groupby(["Month-Day"]).agg({"Data_Value":np.max}) # The groupby function also only considers Month_Day as its index, and uses Data_Value from agg as its only values.
# time2 = [i for i in range(1, 366)] # This describes the x-values (ordinal days)

# Create data for 2015 high and low:
data3 = data[data["Year"] == "2015"]
low2015 = data3[data3["Element"] == "TMIN"].groupby("Month-Day").agg({"Data_Value":np.min})
high2015 = data3[data3["Element"] == "TMAX"].groupby("Month-Day").agg({"Data_Value":np.max})

""" Originally wanted to keep the index value, but there's a better method.
low2015 = low2015.where(low2015["Data_Value"] < low["Data_Value"]).dropna() # There's also an np.where, but use pd.df.where since it keeps the index values, and drop the nAn values.
high2015 = high2015.where(high2015["Data_Value"] > high["Data_Value"]).dropna() # Only keep record-breaking temps for both 2015 sets.
"""


# Finding 2015 record-breakers
lowrec = np.where(low2015["Data_Value"] < low["Data_Value"])[0] # We can find out where these values are located later on.
highrec = np.where(high2015["Data_Value"] > high["Data_Value"])[0]# Returns index position of those values. Also need to index [0] because of weird formatting.

# Plot 2005-2014 highs and lows, and fill the space in-between:

plt.plot(low.values, 'b-', alpha=0.7, label="2005-2014 record low") # The .values portion ensures that the xticks count natually from 0.
plt.plot(high.values, 'r-', alpha=0.7, label="2005-2014 record high")
plt.gca().fill_between(range(len(low)), #x-index value.
                       low["Data_Value"], high["Data_Value"], # Lower, then upper bound (values only). Need to do this so plt doesn't complain about multidimensionality issues
                       facecolor="orange",
                       alpha=0.5) 

# For the record-breaking scatter plot.
plt.scatter(lowrec, low2015.iloc[lowrec], s=50, color="purple", label="2015 record-breaking low")# s to scale the marker point size itself
plt.scatter(highrec, high2015.iloc[highrec], s=50, color="brown", label="2015 record-breaking high") 

# Prettify and labeling the graph.
plt.xticks(range(0, len(low), 30), high.index[range(0, len(high), 30)], rotation=45) # Skip the tick labelling by 1 month (30 days). 
plt.gca().axis([-5, 370, -50.0, 50.0]) # Adjust the axes ranges, adjusting the labels as well.
plt.subplots_adjust(top=1.25, right=1.2) #Enlarge the graph

plt.title("Record-breaking Temperatures near Scarborough, ON, Canada")
plt.xlabel("Day of the year [Month-Date]")
plt.ylabel("Temperature [degrees Celsius]")

plt.legend(frameon=False, loc=8, fontsize=10) # Centers it at the bottom
plt.gca().spines["top"].set_visible(False)# Removes lines from top and right side of graph.
plt.gca().spines["right"].set_visible(False)
plt.show()
