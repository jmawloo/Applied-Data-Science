# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:48:52 2018

@author: Jeff
"""
"""
Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
​
This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Scarborough, Ontario, Canada**, or **Canada** more broadly.
​
You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Scarborough, Ontario, Canada** to Ann Arbor, USA. In that case at least one source file must be about **Scarborough, Ontario, Canada**.
​
You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
​
Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
​
As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
​
Here are the assignment instructions:
​
 * State the region and the domain category that your data sets are about (e.g., **Scarborough, Ontario, Canada** and **economic activity or measures**).
 * You must state a question about the domain category and region that you identified as being interesting.
 * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
 * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
 * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
​
What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
​
## Tips
* Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
* Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
* Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
* This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
​
## Example
Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)
"""

"""
1. Region and Domain: Scarborough (Agincourt), ON, Canada; Economic Activity or Measures.
2. Research Question: How does the income distribution vary between the different sections of Scarborough? (2015 data total income group) [Regions: Agincourt, Centre, Guildwood, North, Rouge Park, Southwest]
    - Download: http://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/search-recherche/results-resultats.cfm?Lang=E&TABID=1&G=1&Geo1=FED&Code1=35096&Geo2=PR&Code2=35&SearchText=Scarborough&SearchType=Begins&wb-srch-place=search
3.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import get_ipython as ipy 
ipy().magic("matplotlib qt5")     # Pop out instead of inline

# Load datafiles into dataframe.
df = pd.DataFrame()
names = ["Agincourt","Centre","Guildwood","North","RougePark","Southwest"]
for i in names:
    df[i] = pd.read_csv("IncomeScarborough"+i+".csv", header=None, index_col=0, usecols=[1, 3], squeeze=True, skiprows=lambda x: x in [i for i in range(51)] + [61], nrows=13) # Nrows specifies # of rows to read after the skipped rows.
# Usecols is actually called first before index cols, so index_cols refers to the column elements in usecols.
# Squeeze turns the dataframe into a series if possible, so that all files can be stored in one dataframe.

# Changing the index labels:
df.reset_index(inplace=True)    

df.loc[2,1]= "< \$10k"# (row, then column.)
for i in range(3,12):
    df.loc[i,1] = "[\$" + str(i-2) + "0k - \$" + str(i-1) +"0k)" # Formatting x-labels.
    
df.loc[12,1] = "> \$100k"

df.set_index(1, inplace=True)
# Plotting all the points using matplotlib's hist function.
colors = ["red", "green", "blue", "yellow", "grey", "black"]
for i in range(6):
    plt.figure()
    plt.bar(df.index[2:], df.iloc[2:][names[i]], color=colors[i], alpha=0.8)