import numpy

d = 50
k = [1,10,25,50,100,200,400,800]

matrices = []

for i in k:
    matrices.append(numpy.zeros((d,i)))

for p in matrices:
    
