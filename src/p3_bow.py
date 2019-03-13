import numpy as np
import os
import sys
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

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
    return X_train, X_test, y_train, y_test , X_origin[0:375] , X_origin[375:500]   

def plot_data(ax,data,r,color):
    ax.scatter(data[r][0],data[r][1],data[r][2], c=color, marker='o', alpha=0.05)


def plot_hist_data(bow,index):
    fig, ax=plt.subplots()
    for i in range(15):
        ax.bar(i,bow[index][i],color='b')
    plt.savefig(out_dir+'p3_result_new'+str(index)+'.png')

if __name__ == '__main__':

    X_train_origin = np.empty((1500,64,64,3),dtype = np.uint8)
    X_test_origin = np.empty((500,64,64,3),dtype = np.uint8)
    X_train = np.empty((1500,16,16,16,3),dtype = np.uint8)
    X_test = np.empty((500,16,16,16,3),dtype = np.uint8)
    y_train = np.empty(1500,dtype = np.uint8)
    y_test = np.empty(500,dtype = np.uint8)

    for i in range(4):
        X_train[375*i:375*i+375], X_test[125*i:125*i+125], y_train[375*i:375*i+375], y_test[125*i:125*i+125], X_train_origin[375*i:375*i+375], X_test_origin[125*i:125*i+125] = read_data(i)
    print(y_train)
    # problem1
    # count = 0
    # for rand_img in [100,600,1000,1300]:
    #     # rand_img = random.randint(0,1499)
    #     for j in range(3):
    #         random_grid = random.randint(0,15)
    #         cv2.imwrite(out_dir+'grid'+str(count)+str(j)+'.png', X_train[rand_img][random_grid])
    #     count += 1

    ## problem2

    # X_train = X_train.reshape(1500*16,16*16*3)
    # X_test = X_test.reshape(500*16,16*16*3)

    # pca=PCA(n_components=3)
    # X_train_3dim = pca.fit_transform(X_train)
    # kmeans = KMeans(n_clusters=15, random_state=0, max_iter=5000)
    # kmeans.fit(X_train)
    # center_3dim = pca.transform(kmeans.cluster_centers_)
    # ## problem2 plot

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(center_3dim[1][0],center_3dim[1][1],center_3dim[1][2], c='r', marker='o', alpha=1)
    # ax.scatter(center_3dim[3][0],center_3dim[3][1],center_3dim[3][2], c='b', marker='o', alpha=1)
    # ax.scatter(center_3dim[5][0],center_3dim[5][1],center_3dim[5][2], c='g', marker='o', alpha=1)
    # ax.scatter(center_3dim[10][0],center_3dim[10][1],center_3dim[10][2], c='c', marker='o', alpha=1)
    # ax.scatter(center_3dim[13][0],center_3dim[13][1],center_3dim[13][2], c='y', marker='o', alpha=1)
    # ax.scatter(center_3dim[14][0],center_3dim[14][1],center_3dim[14][2], c='k', marker='o', alpha=1)
    # for i in range(24000):
    #     if kmeans.labels_[i] == 1:
    #         plot_data(ax,X_train_3dim,i,'r')
    #     elif kmeans.labels_[i] == 3:
    #         plot_data(ax,X_train_3dim,i,'b')
    #     elif kmeans.labels_[i] == 5:
    #         plot_data(ax,X_train_3dim,i,'g')
    #     elif kmeans.labels_[i] == 10:
    #         plot_data(ax,X_train_3dim,i,'c')
    #     elif kmeans.labels_[i] == 13:
    #         plot_data(ax,X_train_3dim,i,'y')
    #     elif kmeans.labels_[i] == 14:
    #         plot_data(ax,X_train_3dim,i,'k')
    # plt.savefig(out_dir+'p3_result_center.png')

    ###
    # problem3 - edit
    X_train = X_train.reshape(1500*16,16*16*3)
    X_test = X_test.reshape(500*16,16*16*3)
    kmeans = KMeans(n_clusters=15, random_state=0, max_iter=5000)
    kmeans.fit(X_train)
    bow = np.empty((1500,16), dtype = np.float64)
    bow_test = np.empty((500,16), dtype = np.float64)
    for x in range(1500):
        tmp = np.zeros((16,15))
        for i in range(16):
            for j in range(15):
                tmp[i][j] = distance.euclidean(X_train[x*16+i],kmeans.cluster_centers_[j])
            tmp[i] = np.reciprocal(tmp[i])
            tmp[i] = tmp[i]/(tmp[i].sum(keepdims=1))
        bow[x] = np.amax(tmp,axis=1)
    
    for x in range(500):
        tmp = np.zeros((16,15))
        for i in range(16):
            for j in range(15):
                tmp[i][j] = distance.euclidean(X_test[x*16+i],kmeans.cluster_centers_[j])
            tmp[i] = np.reciprocal(tmp[i])
            tmp[i] = tmp[i]/(tmp[i].sum(keepdims=1))
        bow_test[x] = np.amax(tmp,axis=1)
        
    for i in [100,600,1000,1300]:
        plot_hist_data(bow,i)
    ###

    # problem3
    # X_train_origin = X_train_origin.reshape(1500,64*64*3)
    # X_test_origin = X_test_origin.reshape(500,64*64*3)
    # kmeans = KMeans(n_clusters=15, random_state=0, max_iter=5000)
    # kmeans.fit(X_train_origin)
    # bow = np.empty((1500,15), dtype = np.float64)
    # bow_test = np.empty((500,15), dtype = np.float64)
    # # print(kmeans.cluster_centers_.shape)
    # for i in range(1500):
    #     for j in range(15):
    #         bow[i][j] = distance.euclidean(X_train_origin[i],kmeans.cluster_centers_[j])
    #     bow[i] = np.reciprocal(bow[i])
    #     bow[i] = bow[i]/(bow[i].sum(keepdims=1))

    # for i in range(500):
    #     for j in range(15):
    #         bow_test[i][j] = distance.euclidean(X_test_origin[i],kmeans.cluster_centers_[j])
    #     bow_test[i] = np.reciprocal(bow_test[i])
    #     bow_test[i] = bow_test[i]/(bow_test[i].sum(keepdims=1))
    # # plt problem3
    # for i in [100,600,1000,1300]:
    #     plot_hist_data(bow,i)

    ## problem4
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(bow,y_train)
    print(neigh.score(bow_test,y_test))



    



