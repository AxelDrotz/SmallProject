import numpy
import pandas as pd

#create random matrix
d = 50
k = [1,10,25,50,100,200,400,800]
randomMatrices =[]
#Assuming dimensions k x d
for i in range(len(k)):
        randomMatrices.append(numpy.random.random((k[i],d)))
# convert to df
dataframes =[]
for i in randomMatrices:
    dataframes.append(pd.DataFrame(i))
# make each column sum to 1, not sure but think this does the trick

for j in dataframes:
    t = 0
    sum = j.sum(axis = 0, skipna = True)
    print(j)
    print(sum)
    for p in range(d):

        j[p] = j[p].div(sum[p])

sum =dataframes[0].sum(axis = 0, skipna = True)

#Convert Images to vectors etc should be 1x2500 matrix?



#Euclidean distance
#|| x1 -x2|| approximated by sqrt(d/k)||Rx1-Rx2|| where x1 and x2 are vectors
