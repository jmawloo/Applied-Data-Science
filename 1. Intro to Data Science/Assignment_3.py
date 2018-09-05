import pandas as pd
import numpy as np

"""PART 1"""
##Load UN Energy Indicators.
energy = pd.read_excel("Energy Indicators.xls",header=None,skiprows=18,skip_footer=38,usecols="C:F")
# read_excel file basically has no headers, and skips a few unneccessary rows and columns to store the desired dataset.
# Index_col is used to set the countries as columns.
energy.rename(columns={0:"Country",1:"Energy Supply",2:"Energy Supply per Capita",3:"% Renewable"}, inplace=True)
energy["Energy Supply"].where(energy["Energy Supply"] != "...", other=np.NaN, inplace=True) #Replaces all the "..."'s with the NaN's.


energy["Energy Supply"] = energy["Energy Supply"].apply(lambda x: (x * 10**6))
energy.set_index("Country", drop=True, inplace=True)

for i in energy.index: #  string[:string.find("(")-1] will remove the unneccesary parentheses after country name that has one.
    energy.rename({i:i.strip("0123456789")}, inplace=True)
    if i.find("(") == -1:
        continue
    energy.rename({i:i[:i.find("(")-1]}, inplace=True)


energy.rename({"Republic of Korea":"South Korea", "United States of America":"United States", "United Kingdom of Great Britain and Northern Ireland":"United Kingdom", "China, Hong Kong Special Administrative RegionChina, Hong Kong Special Administrative Region":"Hong Kong"}, inplace=True)
energy.reset_index(inplace=True)

##Now setting up GDP
GDP = (pd.read_csv("world_bank.csv",header=4)
        .rename(columns={"Country Name":"Country"})
        .set_index("Country")) # To use the rename functionality.
GDP.drop(GDP.T.index[0:49],axis=1,inplace=True) # Transpose and then drop data for the year columns 1960 to 2005, inclusive, as well as the country codes and descriptors.
GDP.rename({"Korea, Rep.":"South Korea", "Iran, Islamic Rep.":"Iran", "Hong Kong SAR, China":"Hong Kong"},inplace=True)
GDP.reset_index(inplace=True)

##Setting up country document ranks
ScimEn = pd.read_excel("scimagojr.xlsx")
#ScimEn.drop(index=[i for i in range(15,192)],inplace=True)

#QUESTION 1: Now intersection merging ScimEn with energy, then the merged database with GDP.

temp = pd.merge(ScimEn, energy, how='inner', on="Country")
merged_df = pd.merge(temp, GDP, how='inner', on="Country")# 
# print(merged_df.info())  # To get information, very useful.
new_len = len(merged_df)
merged_df_copy = merged_df.drop(index=[i for i in range(15,new_len)]).set_index("Country") # Only keep top 15 countries and set countries as the index
print(merged_df_copy, '\n')

#Question 2: Union(outer) of all three datasets minus the intersection of all three.
temp = pd.merge(ScimEn, energy, how='outer', on="Country")
unioncount = len(pd.merge(temp, GDP, how='outer', on="Country"))
print(unioncount-new_len, '\n')

#Question 3:
Top15 = merged_df_copy
# print(Top15.columns) #.columns returns the column labels

avgGDP = (Top15.loc[:,"2006":"2015"].mean(axis=1,skipna=True) # finds where the gdp years start and slices desired columns *[:,...] use comma.
               .sort_values(ascending=False))
print(avgGDP, '\n')

#Question 4:
country6 = avgGDP.index[5]
diff = Top15.loc[country6,"2015"]-Top15.loc[country6,"2006"]
print(diff, '\n')

#Question 5:
meanESPC = Top15.loc[:,"Energy Supply per Capita"].mean(skipna=True) #Always skip NaN values
print(meanESPC, '\n')

#Question 6:
maxpcr = Top15.loc[:,"% Renewable"].sort_values(ascending=False) #Grabs the max value in series WITH the country name
print((maxpcr.index[0],maxpcr[0]), '\n')

#Question 7:
Top15["Self to Total Ratio"] = Top15["Self-citations"] / Top15["Citations"]
Ratio = Top15["Self to Total Ratio"].sort_values(ascending=False)
print((Ratio.index[0],Ratio[0]), '\n')

#Question 8:
Top15["PopEst"] = Top15["Energy Supply"] / Top15["Energy Supply per Capita"]
country3 = Top15["PopEst"].sort_values(ascending=False).index[2] # Grabs the 3rd largest populated country on list.
print(country3, '\n')

#Question 9:
Top15["Citable Documents per Capita"] = np.float64(Top15["Citable documents"] / Top15["PopEst"]) # Cast it to double so python doesn't ignore it?
Top15["Energy Supply per Capita"] = Top15["Energy Supply per Capita"].astype(np.float64)

#Correlating columns have to be of same datatype (np.float64)
corr = Top15["Citable Documents per Capita"].corr(Top15["Energy Supply per Capita"],method='pearson') # linear correlation between -1 and 1 is called Pearson method (used in psychology)
print(corr, '\n')
#Now graphing. Enter these values in manually.
"""
import matplotlib as plt
%matplotlib inline # line call for iPython
Top15.plot(x='Citable Documennts per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])

Can conclude that the amount of resources a country has shares a relation with the number of scholarly documents it contributes."""

#Question 10:
median = Top15["% Renewable"].median()
Top15["% renew above median"] = Top15["% Renewable"].apply(lambda x: 1 if x >= median else 0) #Ternary lambda operator.
HighRenew = Top15.sort_values("Rank", axis=0, ascending=True)["% renew above median"] 
print(HighRenew, '\n')

#Question 11:
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}

Top15["PopEst"] = Top15["PopEst"].astype(np.float64) # Need to cast dataset to float type first in order to work
Population = Top15.groupby(ContinentDict)["PopEst"].agg(['size','sum','mean','std']) #These are all builtin numpy functions. 
Population = (Population.reset_index()
                        .rename(columns={"index":"Continent"})
                        .set_index("Continent"))
print(Population, '\n')


'''
How to print the elements of a groupby object:
for group, frame in Population:
    print(group, "Group contains: \n", frame)
'''

#Question 12:
Top15["Bins"] = pd.cut(Top15["% Renewable"],5) #Automatically sorts those bins for you.
Top15.reset_index(inplace=True)
Top15["Continent"] = Top15["Country"].apply(lambda x,y: y[x], y=ContinentDict) #Another way of doing lambdas.
# Top15 = Top15.set_index(["Continent","Country"]).sort_index(level=0)This would be how you set and sort multi-level indices for merging purposes.

temp = Top15.groupby(["Continent","Bins"]).size() #This is a series object, thus column naming is impossible
print(temp, '\n') #This is the final answer
#temp = Top15.groupby([ContinentDict,renewbins]).size().unstack() # Unstack = pivot level of necessarily hierarchal table.

#Question 13:
Top15.set_index("Country",inplace=True)
PopEst = Top15["PopEst"].apply(lambda x: '{:,}'.format(x)) # This will help with the formattiing to the thousands place. Yes, a built-iin python feature I never knew.
print(PopEst, '\n')

#OPTIONAL: SEEING WHAT THE GRAPH LOOKS LIKE
""" RUN THIS LINE-BY-LINE
import matplotlib as plt
    %matplotlib inline
    Top15 = answer_one()

    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. \
This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
2014 GDP, and the color corresponds to the continent.")


"""