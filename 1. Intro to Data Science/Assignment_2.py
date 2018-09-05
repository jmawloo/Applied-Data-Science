import pandas as pd
import numpy as np

#PART 1
df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by ' (' #Don't forget the space
df.index = names_ids.str[0] # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that) (i.e. don't take ')'

df = df.drop('Totals')
print(df.head())

#Question 1
print(df.where(df["Gold"] == df["Gold"].max()).dropna().index[0]) #United States

#Question 2
df["Diff"] = abs(df["Gold"]-df["Gold.1"])
print(df.where(df["Diff"] == df["Diff"].max()).dropna().index[0]) #United States

#Question 3
df_copy = df[(df["Gold"] > 0) & (df["Gold.1"] > 0)] #Ignore the warning as we're using the copy.
df_copy["Avg"] = df_copy["Diff"]/df_copy["Gold.2"]
print(df_copy[df_copy["Avg"] == df_copy["Avg"].max()].index[0]) # Bulgaria

#Question 4
Points = pd.Series(data=(df["Gold.2"]*3 + df["Silver.2"]*2 + df["Bronze.2"]),index=df.index)
print(Points)


print("\n")


#PART 2:
#Question 5
census_df = pd.read_csv('census.csv')
census_df = census_df[census_df["SUMLEV"] == 50].set_index("STNAME")
census_df_index = census_df.index.unique()

county_totals = []
for i in range(len(census_df_index)):
    count = 0
    for j in range(len(census_df)):
        if census_df.index[j] == census_df_index[i]: # Comparing the repeats of the original dataframe index entries with the unique values of our proposed series.
            count += 1
    county_totals.append(count)

Count = pd.Series(data=county_totals, index=census_df_index)
print(Count[Count == Count.max()].index[0])

#Question 6.

Sum = []
for i in census_df_index: # So that we don't have any state repeats.
    try:
        Sum.append(sum(census_df.loc[i]
                       .sort_values("CENSUS2010POP", ascending=False)
                       ["CENSUS2010POP"].iloc[:3] # Slicing will work even for states with less than 3 counties.
                       )
                    ) # Sorting + slicing is the best way to obtain 3 largest values.
    except ValueError: # This happens when the values for "CENSUS2010POP" are exactly 1. (District of Columbia...)
        print(i + ": ", census_df["CENSUS2010POP"].loc[i]) #DEBUG
        Sum.append(census_df["CENSUS2010POP"].loc[i])
series = pd.Series(data=Sum, index=census_df_index).sort_values(ascending=False)[:3]
print(list(series.index))

# Question 7
census_df.reset_index(inplace=True)
census_df = census_df.set_index(census_df["CTYNAME"])
est = census_df[["POPESTIMATE2010","POPESTIMATE2011","POPESTIMATE2012","POPESTIMATE2013","POPESTIMATE2014","POPESTIMATE2015"]]

Diff = est.max(axis=1) - est.min(axis=1)
print(Diff[Diff == Diff.max()].index[0])

#Question 8
census_df.reset_index(inplace=True,drop=True)
census_df_copy = census_df[(census_df["REGION"] == 1) | (census_df["REGION"] == 2)]
census_df_copy = census_df_copy[census_df_copy["CTYNAME"].str.startswith("Washington")] # Str.startswith finds whether the string starts with a particular word.
census_df_copy = census_df_copy[(census_df_copy["POPESTIMATE2015"] > census_df_copy["POPESTIMATE2014"])]
print(census_df_copy[["STNAME","CTYNAME"]])