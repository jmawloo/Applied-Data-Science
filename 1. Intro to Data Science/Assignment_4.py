# -*- coding: utf-8 -*- 
"""
Created on Thu May 10 12:17:49 2018

@author: Jeff
"""

'''
This assignment requires more individual learning than previous assignments - you are encouraged to check out the pandas documentation to find functions or methods you might not have used yet, or ask questions on Stack Overflow and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

Definitions:

A quarter is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
A recession bottom is the quarter within a recession which had the lowest GDP.
A university town is a city which has a high percentage of university students compared to the total population of the city.
Hypothesis: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (price_ratio=quarter_before_recession/recession_bottom)

The following data files are available for this assignment:

From the Zillow research data site there is housing data for the United States. In particular the datafile for all homes at a city level, City_Zhvi_AllHomes.csv, has median home sale prices at a fine grained level.
From the Wikipedia page on college towns is a list of university towns in the United States which has been copy and pasted into the file university_towns.txt.
From Bureau of Economic Analysis, US Department of Commerce, the GDP over time of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file gdplev.xls. For this assignment, only look at GDP data from the first quarter of 2000 onward.
Each function in this assignment below is worth 10%, with the exception of run_ttest(), which is worth 50%.
'''

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map two letter acronyms to state names
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

f = open("university_towns.txt","r") #Gotta do it the old python way now eh.
uni_town = pd.DataFrame(columns=["State","RegionName"]) #New empty dataframe ready to be used.


# PART 1: def get_list_of_university_towns
'''
Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. 
'''
state = "" # This makes the variable global.
while 1:
    line = f.readline().strip('\n')
    if line == "": #EOF
        break
    elif line.endswith(":"):
        continue #Ignore the lines that say "The five colleges of region:".
    elif "edit" in line:
        state = line[:line.find("[")] #Remove the [edit] part of the line.
        continue # So that the states themselves aren't counted as regionnames
    index = line.find(" (")
    line = line[:(index if index > -1 else 100)] # This accounts for the regions without parantheses.
    uni_town = uni_town.append({"State":state, "RegionName":line},ignore_index=True) #Set this to true to prevent alt 1 & 0's from being index
print(uni_town, '\n')


#Part 2: def get_recession_start
'''
Returns the year and quarter of the recession start time as a 
string value in a format such as 2005q3
'''

GDP = pd.read_excel("gdplev.xlsx", header=None, skiprows=220, usecols="E,G") #Just indicate usecols parameter ALL in string.
GDP[1] = GDP[1].astype(np.float64)
count = 0
recess_start = 0
for i in range(len(GDP[1])-1): #The minus one is there because we're comparing the element with the next element in the list, so it prevents IndexError.
    if count == 2:
        recess_start = GDP[0][i-1] #startpoint is BEFORE those 2 consecutive periods.
        break # Don't forget this, unless you want to print the entirety of the recession period lol.
    elif GDP[1][i] > GDP[1][i+1]: #If the next quarter's value is less than the previous quarter, count that as a consecutive.
        count += 1
    else: #Otherwise, reset the counter to zero.
        count = 0
        
print(recess_start, '\n')

""" Alternative:
for i in range(len(GDP[1])-2) #MINUS 2 
if GDP[1][i] > GDP[1][i+1] and GDP[1][i+1] > GDP[1][i+2]:
    recess_start = GDP[0][i]
    break
"""

#Part 3: def get_recession_end
'''
Returns the year and quarter of the recession end time as a 
string value in a format such as 2005q3
'''

start = GDP[0][GDP[0] == recess_start].index[0] # This is the pd. way of finding the index of our recession_start date.
for i in range(start,len(GDP[1])-2): #Definitely need the -2 here for indexing purposes.
    if GDP[1][i] < GDP[1][i+1] and GDP[1][i+1] < GDP[1][i+2]:
        recess_end = GDP[0][i+2] #endpoint is AFTER those 2 consecutive periods.
        break

print(recess_end, '\n')

#Part 4: def get_recession_bottom
'''
Returns the year and quarter of the recession bottom time as a 
string value in a format such as 2005q3
'''
end = GDP[0][GDP[0]==recess_end].index[0]
min1 = GDP[1].iloc[start:end+1].idxmin() # Include the last element, thus +1. Also use idxmin method to find where the min is.
recess_bottom = GDP[0][min1]
print(recess_bottom, '\n')

#Part 5: def convert_housing_data_to_quarters()
'''
Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows. (But it doesn't since the data was updated rip)
'''
house = pd.read_csv("city_Zhvi_AllHomes.csv") #Only need RegionName, State, and indices with start = Jan-00 and end = Sep-1

"""
liststart = list(house.T.index).index("2000-01") # Where we want to slice the list for our dates.
listend = list(house.T.index).index("2016-09") # Where to end.
house2 = house.T[liststart:listend + 1] #listend+1 includes the third quarter ending period. Leave as-is for resampling purposes. Use T to move the indices to columns.
# Convert timeframes to quarters:
house2.index = pd.to_datetime(house2.index) # Converts column indices to the desired DateTime type.
house2 = (house2.reset_index()
                .rename({'index':"Date"}, axis=1))#Don't need to set index. Resample doesn't take indices nicely.

house2[house2.columns[1:]] = house2[house2.columns[1:]].astype(np.float64) #This is how to typecast all the columns other than date.
house2 = (house2.resample("Q", on="Date") # use "on" keyword to specify row or column to perform resampling on.
                .mean()# The .mean is necessary in order to handle merged data in DatetimeIndexResampler object.
                .T) # The T attribute is so that the dates get stored in columns again.
house2[["RegionName","State"]] = house[["RegionName","State"]]
house2 = house2.set_index(["State","RegionName"]).sort_index(level=0)
for i in house2.columns:
    i = "{0}q{1}".format(str(i)[:4],int(str(i)[5:7])/3)

print(house2)
"""

"""THIS IS AN ALTERNATIVE METHOD (IT'S FASTER)"""

house2 = house[["RegionName","State"]] #ALWAYS use double square braackets to indicate column elements.

def name_states(x):
    return states[x["State"]]

house2["State"] = house2.apply(name_states, axis=1) # This will convert the acronyms to the names, useful for later on.

def remove_whitespace(x): # Some of the data contains a whitespace at the beginning of the name, so this function removes that.
    if x["RegionName"][0] == " ":
        return x["RegionName"].replace(" ", "", 1)
    else:
        return x["RegionName"]

house2["RegionName"] = house2.apply(remove_whitespace, axis=1)

lim = 5
for i in range(2000,2017):
    if i == 2016: #Only want it up to the third quarter of the dataset
        lim = 4 
    for j in range(1, lim): # This takes care of the special case easily.
        house2[str(i) + 'Q' + str(j)] = house[[str(i) + '-%.2i' % (j + k) for k in range(3)]].mean(axis=1) #The formatting of the placeholder requires C knowledge; it basically puts in zeros as placeholders if the integer passed is a single digit.
        # K in this case represents the number of months within a quarter.
        #Also, compute mean with axis set to 1 since the resulting dataframe has its dates stored in the rows.

house2 = house2.set_index(["State","RegionName"]).sort_index(level=0) #Prettifies indices.
print(house2.head())

#Part 6: def run_ttest
'''
First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).
'''
house2 = house2.T.iloc[house2.columns.get_loc(recess_start):house2.columns.get_loc(recess_end)+1] # Slice for recession bits.
house2 = (house2.T.reset_index())

# I guess you need to find the ratio?
def ratio(x): #refers to the rows in dataframe.
    """price_ratio = quarter_before_recession/recession_bottom"""
    # return x[recess_start] / x[recess_bottom] # <-Should be this according to formula.
    return (x[recess_start]-x[recess_bottom]) / x[recess_start]

house2["trend"] = house2.apply(ratio, axis=1)

# Now we can manually do membership testing:
uindex = 0
uindex2 = 0

def isunitown(x): 
    """
    Assume that both lists we're dealing with are in sort order.
    The function also has to consider the possibility that the uni_town isn't in the houses list."""
    
    global uindex, uindex2 # You have to declare them as global FIRST like u declare types in C
    try:
        if x["State"] != uni_town.iloc[uindex]["State"]: # In cases where last uni RegionName of state in sorted list is NOT in the house2 data.
            idx, uindex = uindex2, uindex2 # Separate index comes into play and starts iteration at next state.
            
        else:
            idx = uindex
        
        while x["State"] == uni_town.iloc[idx]["State"]:# iloc is an attribute not a method.            
            if x["RegionName"] == uni_town.iloc[idx]["RegionName"]: #Basically the state AND RegionName have to both match to satisfy this statement..
                uindex = idx + 1 # So that we start at the element AFTER the last successful comparison and start comparing from there.
                return 1 # This EXITS the loop.
            
            idx += 1 # Increment and advance the loop.
        uindex2 = idx # Index2 keeps track of when the next state is.
    except IndexError: # This is guaranteed to happen.
        pass #Don't do anything.
        
    return 0 # If nothing else is satisfied.

house2["isunitown"] = house2.apply(isunitown, axis=1)
"""
#THE FOLLOWING IS NOT A GOOD METHOD FOR FINDING UNI TOWNS (states are neglected so duplicate town entries fail.)
uni_town = set(uni_town["RegionName"]) # So that we can perform set membership testing and the type is consistent. Actually works

def isunitown(x): #x is the row series element in the dataframe.
    if x["RegionName"] in uni_town:
        return 1
    else:
        return 0
    
house2["isunitown"] = house2.apply(isunitown, axis=1) #Apply along rows in a column.
"""
    
non_uni = house2[house2["isunitown"] == 0].loc[:, "trend"].dropna() # So that we can actually work with the data/
uni = house2[house2["isunitown"] == 1].loc[:, "trend"].dropna()

def better():
    if non_uni.mean() < uni.mean(): #Whichever one has a lower mean price ratio is the better one.
        return "non-university town"
    else:
        return "university town"

p_val = list(ttest_ind(non_uni, uni))[1] # The p-value is in the second element.    

def result(): 
    if p_val < 0.01:
        return True
    else:
        return False
    
print("({}, {}, {})".format(result(), p_val, better()))
