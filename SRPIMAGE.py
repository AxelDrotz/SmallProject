import numpy
import math
d = 50

k = [1,10,25,50,100,200,400,800]

#store matrices
matrices = []

for i in k:
    matrices.append(numpy.zeros((d,i)))

t = 0
#Make values of each index in matrix: sqrt(3), 0 , -sqrt(3) with probabilities: 1/6, 2/3, 1/6 respectively
for columns in k:
    for column in range(columns):
        for row in range(d):
            p = numpy.random.random_sample()
            matrix = matrices[t]

            if p < 1/6:
                matrix[row][column] = math.sqrt(3)*1
            elif p <5/6:
                 matrix[row][column]= 0
            else:
                matrix[row][column] = -math.sqrt(3)
    t+=1

print(matrices)
