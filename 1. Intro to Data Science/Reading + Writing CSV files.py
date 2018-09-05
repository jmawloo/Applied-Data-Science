import csv

#%precision 2 -> Used in iPython for print point precision.

with open('olympics.csv') as csvfile:
    data = list(csv.DictReader(csvfile)) # Converts all raw datasets into , then organizes all complete sets into list.

print(data[0].keys()) # will look at all the key values used in each dataset element of list.
print(sum(float(d["key"]) for d in data) / len(data)) # Get average of data values.
legs = set(d['leg'] for d in data) # How to organize specific data in set
print(legs)

"""
CtyMpgByCyl = []

for c in cylinders: # iterate over all the cylinder levels
    summpg = 0
    cyltypecount = 0
    for d in mpg: # iterate over all dictionaries
        if d['cyl'] == c: # if the cylinder level type matches,
            summpg += float(d['cty']) # add the cty mpg
            cyltypecount += 1 # increment the count
    CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')

CtyMpgByCyl.sort(key=lambda x: x[0]) # Basically sorts according to 'cylinder' value of tuple elements.
CtyMpgByCyl 


HwyMpgByClass = []

for t in vehicleclass: # iterate over all the vehicle classes
    summpg = 0
    vclasscount = 0
    for d in mpg: # iterate over all dictionaries
        if d['class'] == t: # if the cylinder amount type matches,
            summpg += float(d['hwy']) # add the hwy mpg
            vclasscount += 1 # increment the count
    HwyMpgByClass.append((t, summpg / vclasscount)) # append the tuple ('class', 'avg mpg')

HwyMpgByClass.sort(key=lambda x: x[1])
HwyMpgByClass
"""