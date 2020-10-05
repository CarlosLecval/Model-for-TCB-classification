import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

path = "your path to the TCB images"

#obtain the images path
img_files_no = sorted([os.path.join(path, 'no', file) 
                      for file in os.listdir(path + "/no") 
                      if file.endswith('.jpg')])
img_files_si = sorted([os.path.join(path, 'yes', file) 
                      for file in os.listdir(path + "/yes") 
                      if file.endswith('.jpg')])
    

#pre-processing for images that are not TCBs
img_no = [cv2.imread(i) for i in img_files_no[:]]

#resize images
height = 100
width = 100
dim = (width, height)
res_img_no = []
for i in range(len(img_no)):
    res = cv2.resize(img_no[i], dim, interpolation=cv2.INTER_LINEAR)
    res_img_no.append(res)

#reduce noise
noise_no = []
for i in range(len(res_img_no)):
    blur = cv2.GaussianBlur(res_img_no[i], (5, 5), 0)
    noise_no.append(blur)

#change to gray
gris_no = []
for i in range(len(noise_no)):
    gray = cv2.cvtColor(noise_no[i], cv2.COLOR_RGB2GRAY)
    gris_no.append(gray)

#segmentation
thresh_no = []
for i in range(len(gris_no)):
    ret, thresh_img = cv2.threshold(gris_no[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_no.append(thresh_img)

#final result reshaped to (1, 10000)      
x = thresh_no[0].reshape(1,10000)
for i in range(1, len(thresh_no)):
    imgFin = thresh_no[i]
    imgX = imgFin.reshape(1, 10000)
    x = np.concatenate((x, imgX))

#pre-processing for images that are TCBs
img_si = [cv2.imread(i) for i in img_files_si[:]]

#resize images
height = 100
width = 100
dim = (width, height)
res_img_si = []
for i in range(len(img_si)):
    res = cv2.resize(img_si[i], dim, interpolation=cv2.INTER_LINEAR)
    res_img_si.append(res)

#reduce noise
noise_si = []
for i in range(len(res_img_si)):
    blur = cv2.GaussianBlur(res_img_si[i], (5, 5), 0)
    noise_si.append(blur)

#change to gray
gris_si = []
for i in range(len(noise_si)):
    gray = cv2.cvtColor(noise_si[i], cv2.COLOR_RGB2GRAY)
    gris_si.append(gray)
  
#segmentation
thresh_si = []
for i in range(len(gris_si)):
    ret, thresh_img = cv2.threshold(gris_si[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_si.append(thresh_img)
    
#final result reshaped to (1, 10000)
y = thresh_si[0].reshape(1,10000)
for i in range(1, len(thresh_si)):
    imgFinS = thresh_si[i]
    imgY = imgFinS.reshape(1, 10000)
    y = np.concatenate((y, imgY))
    
#final results for trainning and testing model
#labels
a = np.zeros(4901)
b = np.ones(1900)
c = np.concatenate((a, b))
np.save("results.npy", c)

#images
data = np.concatenate((x,y))
np.save("preprocessed_img.npy", data)
