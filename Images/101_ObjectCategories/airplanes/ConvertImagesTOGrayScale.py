import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

for k in range(1,801):
    if k <10:
        x ='000'+str(k)
    if 10<=k and k<100:
        x ='00'+str(k)
    if k>=100:
        x ='0'+str(k)
    print(x)
    
    img = Image.open('image_'+x+'.jpg')
    imgGray = img.convert('L')
    imgGray.save('gray'+x+'.jpg')
