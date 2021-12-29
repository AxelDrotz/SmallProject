import numpy
import pandas as pd

#create random matrix
d = 2500
k = [1,2,4]
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
    for p in range(d):

        j[p] = j[p].div(sum[p])

sum =dataframes[0].sum(axis = 0, skipna = True)

#Convert Images to vectors etc should be 1x2500 matrix?
X = numpy.load('image_array.npy')
R = dataframes
print(R[0].shape)
print(X.shape)
xt = X.transpose()
XRP = []
for i in R:
    XRPtemp = numpy.matmul(i,xt)
    XRP.append(XRPtemp)
#print(XRP)
#Euclidean distance
#|| x1 -x2|| approximated by sqrt(d/k)||Rx1-Rx2|| where x1 and x2 are vectors
