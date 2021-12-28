import numpy as np
from PIL import Image

saved_arrays = []
for k in range(1,801):
    if k <10:
        x ='000'+str(k)
    if 10<=k and k<100:
        x ='00'+str(k)
    if k>=100:
        x ='0'+str(k)

    image = Image.open("C:/SmallProjectAdvML/SmallProject\planes/"+ 'gray'+x+'.jpg')
    image_array = np.array(image)
    dims = image.size
    print(dims)
    row = dims[0]//2
    col = dims[1]//2
    row = row-25
    col = col -25
    v=[]
    print(row)
    print(col)
    for r in range(row,row+50):
        for c in range(col,col+50):
            v.append(image_array[c][r])
    saved_arrays.append(v)
a_file = open("plains.txt", "w")

#Normalizing data is this necessary????
for i in range(len(saved_arrays)):
    temp_sum = sum(saved_arrays[i])
    saved_arrays[i] = saved_arrays[i]/temp_sum
#for row in saved_arrays:
#    print(sum(row))
#    np.savetxt(a_file, row)
#a_file.close()
np.save('image_array.npy', saved_arrays)
