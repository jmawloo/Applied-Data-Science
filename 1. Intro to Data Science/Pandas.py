import pandas as pd

#SERIES DATA STRUCTURE
#pd.Series? Works in iPython to display help. Type help(pd.Series) for regular Python.

people = ["Bill","Chris","Martha", None] # The null type is cast into object.
print(pd.Series(people)) # Autosets datatype.

numbers = [1,2,3,None] # none becomes NaN, as well as all floating point.
print(pd.Series(numbers))

import numpy as np
print(np.nan == None, np.nan == np.nan) # Outputs false, can't compare NaN to itself.
print(np.isnan(np.nan)) # Use this special method.

games = {'Half Life':'Freeman', "Garry's Mod": "Garry", "Minecraft":"Steve", "Fortnite":"Battle Royale"}
g = pd.Series(games) # Autosets dictionary as object with respective key and value.
print(g)
print(g.index) #Gets index values.
h = pd.Series([1,2,3], index=["Basic","Tings","fam"]) # Datatype is int64 for this; proof it only depends on value.
print(h)
h = pd.Series(games, index=['Half Life', "Garry's Mod", "Angry Birds"]) # Only gets 2 key:value pairs from dict. Other key returns nonetype value
print(h, '\n')

# QUERYING SERIES
print(g.iloc[3]) # Prints out value for numeric version of key entry (Yes, it's in order).
print(g.loc['Half Life']) # NOTE: These are attributes.
print(g["Garry's Mod"]) # same as g.loc. For numeric values hard to discern between iloc and loc, so use those attributes instead.

# things = pd.Series({1:'a',2:'b',3:'c'})
# things[0] gives an error, so use things.iloc[0]
val = pd.Series([132.2,43.1,123.4,543.3,123.4324,5435.23])
#instead of doing
total = 0
for item in val:
    total += item
print(total)
# Try for faster speed
total = np.sum(val) # Vectorization; parallel computing.
print(total)

val = pd.Series(np.random.randint(0,1000,10000)) # arguments describes min, max and size.
print(val.head) # prints out length and dtype if it's an attribute.

# Using %%timeit on iPython will tell us how fast the sum method is for Numpy compared to a regular for loop.
# Broadcasting: Apply value to every element of the series.
val *= 7
print(val.head()) # prints out only first 5 elements + datatype.
#Slower Alternative:
for label, value in val.iteritems():
    # val.set_value(label, value + 2) <- deprecated.
    val.at[label] = value + 2 #or .iat[] for index counting.
print(val.head())

#.loc and .iloc can be used to set new values too.
# Pandas can also change series datatype based on entry received:
s = pd.Series([1,2,3]) # Remember this is already a dictionary with indices as key values/
s.loc["Pandas"] = "Python"
print(s)

#Non-unique data entries:
life = pd.Series(["woah","sampletext","y","foo"],index=[42,42,42,42])
app = s.append(life) # doesn't change values of original 2 series, unlike the normal append method for lists.
print(app) # Query = request for data or info from a database table.

"""THE DATAFRAME : Primary way to work with data analysis and frames"""
#Selecting + Projecting databases.
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1,purchase_2,purchase_3], index=['Store 1','Store 1','Store 3']) # still works with same entries
print(df.head()) # Organizes everything into an ACTUAL table...
print(df.loc["Store 3"])
print(df.loc["Store 1"]) # This works.
print(type(df.iloc[0]))

"""
for i in range(3):
  print(df.iloc[i].loc["Item Purchased"])
"""
# Better way:
print(df["Item Purchased"]) # No need for df.loc. I guess u use this to find all values for column entries?
#Can also do this:
print(df.T.loc["Item Purchased"])

print(df.loc["Store 1","Cost"]) # Prints out the cost for Store 1
print(df.loc["Store 1"]["Cost"]) # Alternative. But causes Python to return a copy instead of a view of DataFrame.
print(df.loc[:,["Name","Cost"]]) # Supports slicing too.

df_copy = df.drop("Store 3") # returns a copy where the dataset is actually dropped. inplace allows for df itself to be changed, and axis = 1 is for dropping a column (0 for row)
print(df)
print(df_copy)

del df_copy["Cost"] # Will directly remove cost
print(df_copy)

df["Location"] = None
print(df)

#Dataframe indexing and loading:
costs = df['Cost'] # Series creation.
costs += 2 # Broadcasting
print(df) #NOTE: Cost of original dataframe increases as well.

# Load: Use !cat olympics.csv in iPython.
df = pd.read_csv('olympics.csv') # Comma-separated values [CSV]
print(df.head())
df = pd.read_csv('olympics.csv',index_col=0,skiprows=1) #These just remove the indices present outside the table. Index-col also grabs the respective column and sets that as key.
print(df.head())
print(df.columns) # This will print the column headers.
for col in df.columns: #This will change the names of the headers.
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True) # Inplace operator needed to update original dataframe as well.
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True) # rename the original column to a new name.
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True)
print(df.head())

"""Querying DataFrame using Boolean masking (fast and efficient numpy method).
Can be either one dimension like an array or 2 dimensions like a dataframe. 
Each value of array is either true or false.
"""

print(df["Gold"] > 0) # Will tell us which datasets meet this goal. Binding Boolean values
only_silver = df.where(df["Silver"] > 0) # Creates copy of data frame containing values only for those who got silver medals
print(only_silver.head()) # If condition not met returns NaN value
print(only_silver["Silver"].count()) # counts amount of countries with silver medals awarded. Ignores NaN values.
print(df["Silver"].count())

only_silver = only_silver.dropna() # Drops NaN values. can optionally specify axis to drop (row or column of NaN's)
print(only_silver.head())

only_silver = df[df["Silver"] > 0] # can also query data this way.
print(only_silver.head()) #Notice that NaN's do not appear; this is b/c of the shortcut.

print(len(df[(df["Gold"] > 0) | (df["Gold.1"] > 0)])) # Tells us how many countries won gold at some time. Chain conditionals using (x) | (y) or (x) & (y).
print(df[(df["Gold.1"] > 0) & (df["Gold"] == 0)]) # Poor Liechtenstein. Won in winter but not summer
#Also, make sure boolean expressions are kept in parantheses b/c of order of operations.

#INDEXING DATAFRAMES
# set_index turns the selected column into the first index column, while also destroying the old index.
df["country"] = df.index # This preserves the index by setting it as a new column.
df = df.set_index("Gold") # Summer gold medal wins.
print(df.head())
df = df.reset_index() # Resets index to count from 0 for each row.
print(df.head())

df = pd.read_csv("census.csv")
print(df.head())
print(df["SUMLEV"].unique()) # This will print out all the different values, with no repeats.
df = df[df["SUMLEV"] == 50] # gets rid of all the state level data
print(df.head())

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
print(df.head()) # only prints out the columns we want.
df = df.set_index(["STNAME","CTYNAME"]) # State name and county name. Surround indices in square brackets to use them. Also creates empty row below row headings.
print(df.head()) # Merges similar entries for STNAME

print(df.loc[[("Alabama","Autauga County"),("Alabama","Blount County")]]) # To just see the data for 2 counties in same state. Note that index groups are in tuples.

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1','Store 1', 'Store 2'])

df["Location"] = df.index
df = df.set_index(["Location","Name"])
df = df.append(pd.Series(data={"Item Purchased": 'Kitty Food', "Cost": 3.00},name=("Store 2","Kevyn"))) # To add entries, name is for the index keys (in order and in tuple) whereas data is for the other data entries. Use .append method as well.
print(df)

#MISSING VALUES: ALready seen example with Nonetype and NaN
df = pd.read_csv("log.csv") # Pretty common in parallel computation: Timestamps aren't sorted as expected. Lots of values in paused and volume colummns.
print(df.head())
df["volume"] = df["volume"].fillna(42)
print(df.head()) # Now the last column contains default value of 42

"""Other common filling methods:
ffill -> Forward filling: updates value of particular cell with value from previous row.
bfill -> Back filling: same, but next row.
"""

df = df.set_index("time").sort_index()
print(df.head()) # Now the index is set to and sorted by timestamp.
df = df.reset_index()
df = df.set_index(['time','user']).fillna(method='ffill')
print(df.head()) # Notice that the "paused" section was forward filled as well as the "volume" section.

# When performing calculations with Numpy, often ignores cells with no values.
