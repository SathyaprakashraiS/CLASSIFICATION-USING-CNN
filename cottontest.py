# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 03:14:39 2023

@author: SATHYA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 01:33:10 2022

@author: SATHYA
"""

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import datasets,layers,models
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import cv2
print("PROGRAM STARTED")
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
ds_train=train.flow_from_directory('D:/CNN/COTTON PLANT DETECTION/Cotton-Plant-Disease-Prediction-main/Cotton Disease_val/',target_size=(76,76),batch_size=3,class_mode='binary')
ds_test=validation.flow_from_directory('D:/CNN/COTTON PLANT DETECTION/Cotton-Plant-Disease-Prediction-main/Cotton Disease_Test/',target_size=(76,76),batch_size=3,class_mode='binary')
print(ds_train.class_indices)
print(ds_test.class_indices)
#print(ds_test.classes)
cnn=models.Sequential([
        layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(76,76,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(76,76,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(76,76,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(76,76,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation="relu"),
        layers.Dense(4,activation="softmax")
        ])
cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#cnn.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#cnn.summary()

#cnn.fit=cnn.fit(ds_train,steps_per_epoch=25,epochs=50,validation_data=ds_test)
cnn.fit=cnn.fit(ds_train,epochs=5)
print(" train mudinchu")

img=image.load_img("D:/CNN/COTTON PLANT DETECTION/Cotton-Plant-Disease-Prediction-main/Cotton Disease_train/diseased cotton leaf/dis_leaf (4)_iaip.jpg",target_size=(76,76,3))
plt.imshow(img)
plt.show()
newstring=""
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
ans=cnn.predict(images)
print("ans",ans)
newstring=str(ans)
print(newstring)
f=0
onegap=0
twogap=0
threegap=0
j=0
while(j<len(newstring)):
    if(newstring[j]==" " and f==0):
        f+=1
        onegap=j
        one=newstring[2:j-1]
        j+=1
    elif(newstring[j]==" " and f==1):
        f+=1
        twogap=j
        two=newstring[onegap:j-1]
        j+=1
    elif(newstring[j]==" " and f==2):
        f+=1
        threegap=j
        three=newstring[twogap:j-1]
        j+=1
    elif(f==3):
        four=newstring[threegap:-3]
        j+=1
    else:
        j+=1
one=one.replace(" ","0")
one=one.replace("e","0")
two=two.replace(" ","0")
two=two.replace("e","0")
three=three.replace(" ","0")
three=three.replace("e","0")
four=four.replace(" ","0")
four=four.replace("e","0")
for k in range(len(one)):
    if(one[k]=="."):
        one=one[:k+2]
        one=float(one)
        break
for k in range(len(two)):
    if(two[k]=="."):
        two=two[:k+2]
        two=float(two)
        break
for k in range(len(three)):
    if(three[k]=="."):
        three=three[:k+2]
        three=float(three)
        break
for k in range(len(four)):
    if(four[k]=="."):
        four=four[:k+2]
        four=float(four)
        break
if(float(one)>float(two) and float(one)>float(three) and float(one)>float(four)):
    print("class 1")
if(float(two)>float(one) and float(two)>float(three) and float(two)>float(four)):
    print("class 2")
if(float(three)>float(two) and float(three)>float(one) and float(three)>float(four)):
    print("class 3")
if(float(four)>float(two) and float(four)>float(three) and float(four)>float(one)):
    print("class 4")
