import pandas as pd
import numpy as np

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df["Date"] = ["December 1", "January 1", "mid-May"]
print(df)
df["Delivered"] = True # Scalar value is default for all row values.
print(df)
df["Feedback"] = ["Positive", None, "Negative"] # Can also set list value.
print(df)

adf = df.reset_index() # Now has index values from 0-2
adf["Date"] = pd.Series({0: "december 1", 2: "Mid-may"}) # Pandas will put in missing values for us by saying it's NaN.
print(adf, '\n')

"""Notations for Dataframe  merging:
UNION: Called "full outer join".
INTERSECTION: Called "inner join".

left_index and right_index just take the first column element after the index, with left and right describing the different datasets to be merged.
left_on and right_on need the index keys [0-2 e.g.] (reset_index()) and can run off of any common column.
"""

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}]).set_index("Name")
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}]).set_index("Name")
print(pd.merge(staff_df,student_df, how='outer', left_index=True,right_index=True)) # This will merge both sets into a union.
print(pd.merge(staff_df,student_df, how='inner', left_index=True,right_index=True)) # intersection. Default value of how as well.
print(pd.merge(staff_df,student_df, how='left', left_index=True,right_index=True)) # Left-merges staff data members with student data, not including unique student data members.
print(pd.merge(staff_df,student_df, how='right', left_index=True,right_index=True)) # Right-merges, with student-focussed members.

staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
print(pd.merge(staff_df,student_df, how='outer', left_on="Role", right_on="Name"), "blablalblablablalbalbalblablalb") # left_on and right_on both tell pandas where to join the left and right columns. If they don't match, creates Name_x and Name_y.

staff_df["Location"] = pd.Series(["State Street","Washington Avenue","Washington Avenue"])
student_df["Location"] = pd.Series(["25 Spooner St.","Cathedral #32","1924 Legit Rd."]) # conflicting locations in the different dataframes.


print(pd.merge(staff_df,student_df,how='left',left_on='Name',right_on='Name')) # Upon merging, displays location_x and location_y as separate datasets. _x is always left, and _y is always right.

staff_df.rename(columns={"Name":"First Name"}, inplace=True) # renames heading of column.
student_df.rename(columns={"Name":"First Name"}, inplace=True)
staff_df["Last Name"] = pd.Series(["Wilkins","Nguyen","Rajawar"])
student_df["Last Name"] = pd.Series(["Mosley","Tammourin","Nguyen"])
print(staff_df)
print(student_df)
print(pd.merge(staff_df,student_df,left_on=["First Name","Last Name"],right_on=["First Name","Last Name"])) # only Sally Nguyen shows up.

# PANDAS IDIOMS (Pandorable)
"""Often has high performance and readability, but not always the case.

[some_index][some_other_index] <-Miss me with that fake copy shit. -Pandas
    -i.e. do u really wanna do chain indexing? Try [some_index,some_other_index] slicing.
"""

df = pd.read_csv("census.csv")
df = (df.where(df["SUMLEV"] == 50)
    .dropna()
    .set_index(["STNAME", "CTYNAME"])
    .rename(columns={'ESTIMATESBASE2010':"Est. Base 10"}) # Pandorable way of doing things.
) # Brackets are used to make things more readable.
print(df.head())

"""
pd.applymap: similar to map, it maps a DataFrame into a function.
pd.apply: allows for mapping over a specific row.

"""

def min_max(row): # returns an entirely new DataFrame object.
    data = row[['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']]
    return pd.Series({"max": np.max(data),"min": np.min(data)}) # The order of max and min doesn't matter.

print(df.apply(min_max, axis=1)) # Maps the function, axis = apply function to each row.

def min_max2(row): # adds on to existing Df.
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row["max"] = np.max(data)
    row["min"] = np.min(data)
    return row

print(df.apply(min_max, axis=1), '\n')

rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']

df.apply(lambda x: np.max(x[rows]),axis=1)
df.apply(lambda x: np.min(x[rows]),axis=1)


"""GROUP BY
"""
df = pd.read_csv('census.csv')
df = df[df["SUMLEV"]==50]

# use %%timeit -n 10
for state in df["STNAME"].unique(): # slow way of finding a total population of a member group.
    avg = np.average(df.where(df["STNAME"] == state).dropna()["CENSUS2010POP"])
    print("countries in state " + state + " have an average pop of " + str(avg))

for group, frame in df.groupby("STNAME"): #Gives all subframes in group STNAME, faster method for multilevel indices.
    avg = np.average(frame["CENSUS2010POP"])
    print("counties in state " + group + " have an average pop of " + str(avg), '\n')


df = df.set_index("STNAME")

def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')
# This will only run with the complete data set. Basically it categorizes groups by the state's first alphabet.



df = pd.read_csv('census.csv')
df = df[df["SUMLEV"]==50]

#The agg method; takes a column of information and maps it to a function on its right. Also combines similar sets of data.
print(df.groupby("STNAME").agg({"CENSUS2010POP": np.average}), "THIS IS WHAT I'M LOOKING FOR ")

"""
print(df.groupby("Category").apply(lambda df,a,b: sum(df[a] * df[b]),"Weight (oz.)","Quantity"))
in the apply method within lambda the sum() gives the value for the larger "category" instead of every single item in the category.
"""
print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011'])) #DataFrameGroupBy. Level refers to the 1st level of multilevel index.
print(type(df.groupby(level=0)['POPESTIMATE2010'])) #SeriesGroupBy. Differing types respond to aggregate method differently.
print(df.set_index("STNAME").groupby(level=0)["CENSUS2010POP"].agg({'avg': np.average, 'sum': np.sum})) # Dictionary use is gonna be deprecated? D:
print(df.set_index("STNAME").groupby(level=0)["POPESTIMATE2010","POPESTIMATE2011"].agg({'avg': np.average, 'sum': np.sum})) # Larger section groups smaller one.
print(df.set_index("STNAME").groupby(level=0)["POPESTIMATE2010","POPESTIMATE2011"].agg({"POPESTIMATE2010": np.average, "POPESTIMATE2011": np.sum}), '\n') #This will just have one level of column headings.

# DATA TYPES AND SCALES
"""(a,b),(c,d) scales:
Ratio Scale: Units equally spaced, math operations of +-*/ all valid (e.g. height & weight).
Interval Scale: Units equally spaced, but no TRUE ZERO -> Multiply and divide are invalid.
    -e.g. Temp measured in Celsius, Farenheit (0 is actually a meaningful value). & compass still indicates something at 0 degrees.
Ordinal Scale: Order of units important, but not evenly spaced. (e.g. letter grading of A, A+).
    -Can be challenge to work with.
Nominal Scale (often called categorical data): Categories of data, but no order with respect to one another. (e.g. = Teams of sport).
    -Categories with only 2 possible values (Boolean usually) are referred to as BINARY.
"""
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0:"Grades"}, inplace=True)
print(df)

print(df["Grades"].astype("category").head()) # Typecasts Grade elements to category type. Displays number of elements + type of each element.
grades = df["Grades"].astype("category",categories=['D','D+','C-','C','C+','B-','B','B+','A-','A','A+'],ordered=True) # To indicate categories are in logical order, set ordered = True.
print(grades.head())
print(grades > 'C') # Boolean masking.

"""
pd.getdummies() <- converts values of single column into multiple columns of zeros & 1's, indicating where the dummy variables are.
Also, better to convert ratio scale to Categorical scale in most cases.
"""

# If you just compared lexicographically, then you get 'C-' AND 'C+' being greater than 'C'
# Indicate that there is a clear order in data by using astype keywords, and then use mathematical operation sets to compute them (e.g. max).

df = pd.read_csv('census.csv')
df = (df[df["SUMLEV"]==50]
    .set_index("STNAME").groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
)
print(pd.cut(df['avg'],10,labels=['0','1','2','3','4','5','6','7','8','9']), '\n') #cut dataset for avg and place elements in 10 bins of intervals.
# The neat part is that it displays bin intervals in set notation.
# also, labels let you name the bins.
# Sometimes want number of items to be same in bin instead of spacing of bins.

#PIVOT TABLES
"""
Way of summarizing data in dataFrame for particular purpose. 
A data frame where rows represent variable of interest and columns represent another. Cell is some aggregate value.
Pivot tables also contain marginal values (sum of rows + columns).
They allow for comparison of different values in different columns.
"""

df = pd.read_csv('cars.csv')
print(df.head())
print(df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)) #How to construct a pivot table for column comparison..
print(df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True), '\n')

#DATE FUNCTIONALITY
"""
Timestamp: Associates values with points in time.
Period: Represents single time SPAN.
DateTimeIndex: Timestamp index.
PeriodIndex: Index of period.
"""
print(pd.Timestamp("5/5/2018 7:13PM"))
print(pd.Period("1/2018"))
t1 = pd.Series(list('abc'),[pd.Timestamp('2016-01-01'),pd.Timestamp('2016-01-02'),pd.Timestamp('2016-01-03')]) #list('abc') splits into individual letters...
print(t1)
print(type(t1.index)) #DateTimeIndex

t2 = pd.Series(list('def'),[pd.Period('2016-01-01'),pd.Period('2016-01-02'),pd.Period('2016-01-03')])
print(t2)
print(type(t2.index)) # PeriodIndex

#Converting to Daytime
d1 = ['1 June 2013', "Aug 29, 2019", "1940-03-23", "4/12/1693"]
ts1 = pd.DataFrame(np.random.randint(10,100,(4,2)), index=d1, columns=list('ab'))
print(ts1)
ts1.index = pd.to_datetime(ts1.index) # Will convert datetimes to same format.
print(ts1)
print(pd.to_datetime('4.3.12', dayfirst=True)) # Display date last?

#TimeDeltas
print(pd.Timestamp('12/12/2012') - pd.Timestamp('11/10/2011')) #398 days
print(pd.Timestamp('2016/08/29 11:10:10.342') + pd.Timedelta("12D 3H"))

#Working with dates in dataframe:
dates = pd.date_range('12-02-2017', periods=9, freq="2W-SUN") # Starting date, number of dates to generate, frequency + occurance of date.
print(dates)
df = pd.DataFrame({'Count 1': 100 + np.random.randint(-23,12,9).cumsum(),"Count 2":132 + np.random.randint(-5,10,9)}, index=dates) # cumsum is like the it.accumulate method.s
print(df)
print(df.index.weekday_name) # Checks for day of the week.
print(df.diff()) # Use this to find difference between stored value rows (timedeltas).
print(df.resample("D").mean(), "\n THE DEBUG CODE") # Know the mean value in terms of days.
print(df['2018'],'\n', df['2018-02':]) # You can do indexing and even slices.
print(df.asfreq("W", method='ffill'))# Changes daterange frequency; creates new indices.


#Plotting timedata:
import matplotlib.pyplot as plt
#iPython: %matplotlib inline
#df.plot()