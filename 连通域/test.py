import numpy as np 
import cv2 

def get_connect(image):
    x, y = image.shape 
    flag = np.zeros((x, y))
    c = 1
    for i in range(x):
        for j in range(y):
            if flag[i, j] == 0 and image[i, j] == 255:
                # print(i, j, np.amax(flag))
                fun(image, flag, i, j, c)
                c += 1
                # return flag
    return flag

def fun(image, flag, i, j, c):
    x, y = image.shape
    s = [] 
    s.append((i, j))
    while s != []:
        i, j = s.pop()
        if 0 <= i < x and 0 <= j < y and flag[i, j] == 0 and image[i, j] == 255:
            flag[i, j] = c  
            s.append((i-1, j))
            s.append((i+1, j))
            s.append((i, j-1))
            s.append((i, j+1))
    
    
image = np.zeros((10, 10))
image[1:3, 1:3] = 255
image[6:9, 6:9] = 255 
print(image)
flag = get_connect(image)
print(np.amax(flag))