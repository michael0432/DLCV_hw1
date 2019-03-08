import numpy as np
import os
import sys
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

img_dir = "../p3_data/"
out_dir = "../p3_out/"
class_dir = ["banana/","fountain/","reef/","tractor"]


def read_data(category):
    count = 0
    X_origin = np.empty((500,64,64,3),dtype = np.uint8)
    X = np.empty((500,16,16,16,3),dtype = np.uint8)
    Y = np.empty(500,dtype = np.uint8)
    dir_name = img_dir+class_dir[category]
    for file in os.listdir(dir_name):
        filepath = os.path.join(dir_name, file)
        i = cv2.imread(filepath)
        X_origin[count] = i
        tmp_index = 0
        for j in range(4):
            for k in range(4):
                X[count][tmp_index] = i[j*16:(j+1)*16,k*16:(k+1)*16]
                tmp_index += 1
        Y[count] = category
        count += 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test , X_origin   

def plot_data(ax,data,r,color):
    ax.scatter(data[r][0],data[r][1],data[r][2], c=color, marker='o', alpha=0.2)

if __name__ == '__main__':

    X_origin = np.empty((2000,64,64,3),dtype = np.uint8)
    X_train = np.empty((1500,16,16,16,3),dtype = np.uint8)
    X_test = np.empty((500,16,16,16,3),dtype = np.uint8)
    y_train = np.empty(1500,dtype = np.float32)
    y_test = np.empty(500,dtype = np.float32)

    for i in range(4):
        X_train[375*i:375*i+375], X_test[125*i:125*i+125], y_train[375*i:375*i+375], y_test[125*i:125*i+125], X_origin[500*i:500*i+500] = read_data(i)

    # problem1
    # count = 0
    # for rand_img in [100,600,1000,1300]:
    #     # rand_img = random.randint(0,1499)
    #     for j in range(3):
    #         random_grid = random.randint(0,15)
    #         cv2.imwrite(out_dir+'grid'+str(count)+str(j)+'.png', X_train[rand_img][random_grid])
    #     count += 1

    ## problem2

    X_train = X_train.reshape(1500*16,16*16*3)
    X_test = X_test.reshape(500*16,16*16*3)
    pca=PCA(n_components=3)
    X_train_3dim = pca.fit_transform(X_train)
    kmeans = KMeans(n_clusters=15, random_state=0, max_iter=5000)
    kmeans.fit(X_train)

    ## problem2 plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(24000):
        if kmeans.labels_[i] == 1:
            plot_data(ax,X_train_3dim,i,'r')
        elif kmeans.labels_[i] == 3:
            plot_data(ax,X_train_3dim,i,'b')
        elif kmeans.labels_[i] == 5:
            plot_data(ax,X_train_3dim,i,'g')
        elif kmeans.labels_[i] == 10:
            plot_data(ax,X_train_3dim,i,'c')
        elif kmeans.labels_[i] == 13:
            plot_data(ax,X_train_3dim,i,'y')
        elif kmeans.labels_[i] == 14:
            plot_data(ax,X_train_3dim,i,'k')
    plt.savefig(out_dir+'p2_result.png')
    kmeans.score(X_test)

    ## problem3
            

