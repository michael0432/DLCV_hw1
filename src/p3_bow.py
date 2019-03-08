import numpy as np
import os
import sys
import cv2

from sklearn.model_selection import train_test_split

img_dir = "../p3_data/"
class_dir = ["banana/","fountain/","reef/","tractor"]
X = np.empty((2000,64*64*3),dtype = np.float32)
Y = np.empty(2000,dtype = np.float32)

def read_data():
    count = 0
    for c in range(4):
        dir_name = img_dir+class_dir[c]
        for file in os.listdir(dir_name):
            filepath = os.path.join(dir_name, file)
            i = cv2.imread(filepath)
            X[count] = i.flatten()
            Y[count] = c
    
if __name__ == '__main__':
    

