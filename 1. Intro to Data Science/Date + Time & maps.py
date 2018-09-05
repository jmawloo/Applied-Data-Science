import datetime as dt
import time as tm

print(tm.time()) # prints data-defined time

dtnow = dt.datetime.fromtimestamp(tm.time())
print(dtnow) # converts data to natural-language time.

print(dtnow.year,dtnow.month)

delta = dt.timedelta(hours = 100) # Time difference of 100 hours
print(delta)
print(dt.date.today()) # Today's date is what?

data1 = [12,32,14,234]
data2 = [1,43,34,87]
cheapest = list(map(min,data1,data2)) # maps data1 and data2 to min function.
print(cheapest)

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person): # My code now bitch
    title = person.split()[0] # Take the 1st element
    lastname = person.split()[-1] # Take the last element
    return '{} {}'.format(title, lastname) # Now look at the format.

print(list(map(split_title_and_name, people)))

def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

print(times_tables() == [i * j for i in range(10) for j in range(10)])

import numpy as np

#Array creation
m = np.array([[1,2,3,4,5,6],[3,4,5,7,6,7]])
print(m.shape, m)

n = np.arange(0,47,2)
n = n.reshape(8,3) # Not void
print(n)
n = np.linspace(2,7,10) # Last argument is how many numbers in list, and it'll split up nums evenly
print(n)
n.reshape(5,2) # Shapes array in place.
print(n)
print(np.ones((3,2))) # Don't forget to shroud dimensions in tuple
print(np.zeros((3,4)))
print(np.eye(3))

print(np.diag([4,5,6])) # returns diagonalized form of 1-D list.
print(np.array([1,2]*9))
print(np.repeat([1,2,3],3))

ones = np.ones([3,2],int)
print(np.vstack([ones*2,ones-1]))
print(np.hstack([ones*2,ones-1]))

#Operations
x = np.array([1,2,3])
y = np.array([4,5,6])
print(x + y) # element by element
print(x * y)
print(x ** 2)
print(x.dot(y)) # Dot product.
z = np.vstack([x,y])
print(z.T) # Transpose attribute
print(z.dtype) # dtype attribute
z = z.astype("f")
print(z)

print(z.sum()) # functions work with multi-dimensional array
print(z.min(), " = min and max = ", z.max())
print(z.mean())
print(z.std()) # standard deviation
print(z.argmin(), z.argmax()) # indices of max and min return.

#indexing and slices.
print(x[0], x[0:2])
print(x[-2:])
print(z[1,1]) # Use commas to separate indices..
print(z[:2,:-1]) # Gets values up to 2nd row and 2nd last column.
print(z[z > 3])
z[z > 3] = 3 # all elements satisfying condition will be assigned new value of 3.
print(z)

z_copy = z.copy() # copy z to preserve it.
z[:,:] = 0
print(z, '\n', z_copy)

#Iteration:
rand = np.random.randint(0, 100, (5,5))
print(rand)
for row in rand: # for i in range(len(test)): <- Can also do this.
    print(row)

for i, row in enumerate(rand): # enumerate returns both the element number and its index
    print("row", i, "is", row)

rand2 = rand * 2
for i, j in zip(rand, rand2): # Convert this to dictionary.
    print(i, '+', j, '=', i * j)