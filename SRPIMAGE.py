import numpy
import math
d = 2500

k = [1,2]

#store matrices
matrices = []

for i in range(len(k)):
    matrices.append(numpy.zeros((k[i],d)))

t = 0
#Make values of each index in matrix: sqrt(3), 0 , -sqrt(3) with probabilities: 1/6, 2/3, 1/6 respectively
for rows in k:
    for row in range(rows):
        for column in range(d):
            p = numpy.random.random_sample()
            matrix = matrices[t]

            if p < 1/6:
                matrix[row][column] = math.sqrt(3)*1
            elif p <5/6:
                 matrix[row][column]= 0
            else:
                matrix[row][column] = -math.sqrt(3)
    t+=1

R = matrices


X = numpy.load('image_array.npy')
print(X.shape)
xt =X.transpose()
XRP = []
print(xt.shape)
for i in R:
    XRPtemp = numpy.matmul(i,xt)
    XRP.append(XRPtemp)

#Euclidean distance
#|| x1 -x2|| approximated by sqrt(d/k)||Rx1-Rx2|| where x1 and x2 are vectors

#X is an NxD size matrix (800x2500)
#Xt is DxN (2500x 800)
#R is a list of kxN size, (kx2500)
#XRP is a list of kXn (kx800)
answers = numpy.zeros((800,800))
for i in range(0,800,1):
    for j in range(0,800,1):
        if i ==j:
            pass
        elif answers[i][j] != 0:
            pass
        else:
            pass
            #take each row of XRP it is RxXt

ans1 = numpy.sqrt(numpy.matmul(numpy.subtract(X[0],X[1]), numpy.subtract(X[0],X[1])))
j = numpy.sqrt(d/k[0])
p = numpy.subtract(numpy.matmul(R[1],X[0]), numpy.matmul(R[1],X[1]))
ans2 = j*numpy.sqrt(numpy.matmul(p,p))
