import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import svd,eig
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

img_dir = "../p2_data/"
out_dir = "../p2_out/"
training_set = np.empty((40,6,2576),dtype = np.uint8)
testing_set = np.empty((40,4,2576),dtype = np.uint8)
picture_h = 0
picture_w = 0

def read_img():
    global training_set
    global testing_set
    global picture_h
    global picture_w
    for file in os.listdir(img_dir):
        filepath = os.path.join(img_dir, file)
        person_index = file.split(".")[0].split("_")[0]
        img_index = file.split(".")[0].split("_")[1]
        i = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        picture_h = i.shape[0]
        picture_w = i.shape[1]
        if int(img_index) <= 6:
            training_set[int(person_index)-1][int(img_index)-1] = i.flatten()
        else:
            testing_set[int(person_index)-1][int(img_index)-7] = i.flatten()
    training_set = training_set.reshape(240,2576)
    testing_set = testing_set.reshape(160,2576)

def get_mean_face():
    mean_data = np.mean(training_set, axis=0)
    new_data = training_set - mean_data
    # write_img(out_dir,'mean_face.png',mean_data)
    return new_data,mean_data

def write_img(dir_name, img_filename ,img):
    img = img.reshape(int(picture_h), int(picture_w))
    cv2.imwrite((dir_name+img_filename), img)

def pca(new_face):
    U , S , V = svd(new_face.T, full_matrices = False)
    print(U)
    eigenface = U
    for i in range(eigenface.shape[1]):
        eigenface[:,i] -= np.min(eigenface[:,i] )
        eigenface[:,i]  /= np.max(eigenface[:,i] )
        eigenface[:,i]  = (eigenface[:,i]  * 255)
    eigenface = eigenface.astype(np.uint8)
    for i in range(4):
        write_img(out_dir,'eigenface'+str(i)+'.png',eigenface[:,i])
    return eigenface,U,S,V

def reconstruct(r_img,mean_face,U,S,V,first_num):
    r_img = np.array(r_img, np.float32)
    reconstruct_image = mean_face + np.dot(U[:,0:first_num] , np.dot(U[:,0:first_num].T , (r_img - mean_face).T))
    reconstruct_image = reconstruct_image.T
    reconstruct_image -= np.min(reconstruct_image)
    reconstruct_image /= np.max(reconstruct_image)
    reconstruct_image = (reconstruct_image * 255)
    reconstruct_image = reconstruct_image.astype(np.uint8)
    print(first_num, mean_squared_error(reconstruct_image, r_img))
    # write_img(out_dir,'reconstruct_140.png',reconstruct_image)
    return reconstruct_image
# load image
# training set / testing set
# pca

def sk_pca(pca):
    eigenfaces = pca.components_
    # print(eigenfaces.shape)
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:,i] -= np.min(eigenfaces[:,i] )
        eigenfaces[:,i]  /= np.max(eigenfaces[:,i] )
        eigenfaces[:,i]  = (eigenfaces[:,i]  * 255)
    eigenfaces = eigenfaces.astype(np.uint8)
    for i in range(4):
        write_img(out_dir,'eigenface'+str(i)+'.png',eigenfaces[i,:])

def cut_validation_and_y(X_train_pca):
    y = np.empty(240, dtype=np.uint8)
    fold1 = np.empty((80,X_train_pca.shape[1]), dtype=np.float32)
    fold2 = np.empty((80,X_train_pca.shape[1]), dtype=np.float32)
    fold3 = np.empty((80,X_train_pca.shape[1]), dtype=np.float32)
    fold1_y = np.empty(80, dtype=np.uint8)
    fold2_y = np.empty(80, dtype=np.uint8)
    fold3_y = np.empty(80, dtype=np.uint8)
    for i in range(240):
        y[i] = i/6

    for i in range(240):
        if i % 3 == 0:
            fold1[i//3] = X_train_pca[i]
            fold1_y[i//3] = y[i]
        elif i % 3 == 1:
            fold2[i//3] = X_train_pca[i]
            fold2_y[i//3] = y[i]
        else:
            fold3[i//3] = X_train_pca[i]
            fold3_y[i//3] = y[i]

    return fold1,fold2,fold3,fold1_y,fold2_y,fold3_y,y

def get_test_y():
    y = np.empty(160, dtype=np.uint8)
    for i in range(160):
        y[i] = i/4
    return y

def knn(n,k,train_f1,train_f2,train_fy1,train_fy2,test_f1,test_fy1):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(np.concatenate((train_f1,train_f2),axis = 0),np.concatenate((train_fy1,train_fy2),axis = 0))
    score = neigh.score(test_f1,test_fy1)
    print(n,k,score)
    return score



if __name__ == '__main__':

    read_img()

    # training_set = np.array(training_set)
    # testing_set = np.array(testing_set)
    new_face, mean_face = get_mean_face()
    # PCA
    # eigenface,U,S,V = pca(new_face)
    n_components = 140
    pca=PCA(n_components=n_components)
    # pca.fit(training_set)
    
    
    X_train_pca = pca.fit_transform(training_set)
    recong = pca.inverse_transform(X_train_pca)

    # First four eigenface
    # sk_pca(pca)
    
    # reconstruct
    loss = ((training_set[0,:] - recong[0,:]) ** 2).mean()
    # write_img(out_dir,'reconstruct_image_229'+'.png',recong[0,:])
    fold1,fold2,fold3,fold1_y,fold2_y,fold3_y,train_y = cut_validation_and_y(X_train_pca)
    
    # for k_n in [1,3,5]:
    #     score = 0 
    #     score += knn(n_components,k_n,fold1,fold2,fold1_y,fold2_y,fold3,fold3_y)
    #     score += knn(n_components,k_n,fold1,fold3,fold1_y,fold3_y,fold2,fold2_y)
    #     score += knn(n_components,k_n,fold2,fold3,fold2_y,fold3_y,fold1,fold1_y)
    #     print("total_score: ",score/3)

    # final result on testing data
    neigh = KNeighborsClassifier(n_neighbors=1)
    X_test_pca = pca.transform(testing_set)
    test_y = get_test_y()
    neigh.fit(X_train_pca,train_y)
    print(neigh.score(X_test_pca,test_y))
    



    
