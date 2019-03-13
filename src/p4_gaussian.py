import cv2
import numpy as np
import math

img_path = "../lena.png"
out_dir = "../p4_out/"

def read_img(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im

def gaussian_filter(sigma):
    ker = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            index_i = i - 1
            index_j = j - 1
            ker[i][j] = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(index_i ** 2 + index_j ** 2)/(2 * (sigma ** 2)))
    return ker

def Ix_filter():
    ker = np.zeros((3,3))
    ker[0][1] = 0.5
    ker[2][1] = 0.5
    return ker

def Iy_filter():
    ker = np.zeros((3,3))
    ker[1][0] = 0.5
    ker[1][2] = 0.5
    return ker


def filter(image , kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    new_image = new_image.astype(np.uint8)
    return new_image

if __name__ == '__main__':
    lena = read_img(img_path)
    gaussian_kernel = gaussian_filter(1 / (2 * math.log(2)))
    Ix_kernel = Ix_filter()
    Iy_kernel = Iy_filter()

    blur_lena = filter(lena,gaussian_kernel)
    Ix_lena = filter(lena,Ix_kernel)
    Iy_lena = filter(lena,Iy_kernel)
    
    cv2.imwrite(out_dir+'gaussian_lena.png',blur_lena)
    cv2.imwrite(out_dir+'Ix_lena.png',Ix_lena)
    cv2.imwrite(out_dir+'Iy_lena.png',Iy_lena)



