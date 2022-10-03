# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:48:34 2022

@author: Musarat Hussain
"""

####     Excercise 4    ###

import skimage.transform 
from skimage.transform import resize
import scipy.stats as stats

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random

from PIL import Image
import cv2


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


#First take a sample of images to see whether the fuction working or not
#X = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_1')
#xtran1 = np.array(X['data'],dtype = np.uint16)
#xtran1 = xtran1[0:50]
#trlabels1 = np.array(X['labels'],dtype = np.uint16)
#TrainLabel = trlabels1[0:50]


trainData1 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_1')
trainData2 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_2')
trainData3 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_3')
trainData4 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_4')
trainData5 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_5')

xtran1 = np.concatenate((trainData1["data"],trainData2["data"],trainData3["data"],trainData4["data"],trainData5["data"]))
TrainLabel = np.concatenate((trainData1["labels"],trainData2["labels"],trainData3["labels"],trainData4["labels"],trainData5["labels"]))


#### This cell needs to be completed first in the morning. ###

test = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
test1 = np.array(test['data'],dtype = np.uint16)
test = test1

yp = test.reshape((len(test),3,32, 32)).transpose(0, 2, 3, 1)
yp = yp.astype(float)
yp = resize(yp,(len(yp),1,1,3),anti_aliasing=True,order = 1)





def cifar10_color(X):   # I am not finding the way to resize it in the required shape using resize fuction. 
  X = X.reshape((len(xtran1),3,32, 32)).transpose(0, 2, 3, 1)
  X = X.astype(float)
  X = resize(X,(len(X),1,1,3),anti_aliasing=True,order = 1)
  return X


xp = cifar10_color(xtran1)
xp.shape


yp = yp.astype(float)
xp = xp.astype(float)



def cifar_10_naivebayes_learn(xp,Y):

  classIndex0 = []
  classIndex1 = []
  classIndex2 = []
  classIndex3 = []
  classIndex4 = []
  classIndex5 = []
  classIndex6 = []
  classIndex7 = []
  classIndex8 = []
  classIndex9 = []


  for ind, i in enumerate(Y):
    #print(i)
    if i == 0:
      classIndex0.append(ind)

    elif i==1:
      classIndex1.append(ind)

    elif i==2:
      classIndex2.append(ind)

    elif i==3:
      classIndex3.append(ind)

    elif i==4:
      classIndex4.append(ind)

    elif i==5:
      classIndex5.append(ind)

    elif i==6:
      classIndex6.append(ind)

    elif i==7:
      classIndex7.append(ind)

    elif i==8:
      classIndex8.append(ind)

    elif i==9:
      classIndex9.append(ind)
    

  Class0 = np.array(xp)[classIndex0]
  Class1 = np.array(xp)[classIndex1]
  Class2 = np.array(xp)[classIndex2]
  Class3 = np.array(xp)[classIndex3]
  Class4 = np.array(xp)[classIndex4]
  Class5 = np.array(xp)[classIndex5]
  Class6 = np.array(xp)[classIndex6]
  Class7 = np.array(xp)[classIndex7]
  Class8 = np.array(xp)[classIndex8]
  Class9 = np.array(xp)[classIndex9]

  #####################################################################################################################################

  Red0 =   []
  Gren0 =  []
  Blue0 =  []

  for index,i in enumerate(Class0):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red0.append(NewImage[0][0][0])
    Gren0.append(NewImage[0][0][1])
    Blue0.append(NewImage[0][0][2])

  muR0 = np.mean(Red0)
  muG0 = np.mean(Gren0)
  muB0 = np.mean(Blue0)

  sigR0 = np.std(Red0)
  sigG0 = np.std(Gren0)
  sigB0 = np.std(Blue0)
  #####################################################################################################################
  Red1 =   []
  Gren1 =  []
  Blue1 =  []

  for index,i in enumerate(Class1):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red1.append(NewImage[0][0][0])
    Gren1.append(NewImage[0][0][1])
    Blue1.append(NewImage[0][0][2])

  muR1 = np.mean(Red1)
  muG1 = np.mean(Gren1)
  muB1 = np.mean(Blue1)

  sigR1 = np.std(Red1)
  sigG1 = np.std(Gren1)
  sigB1 = np.std(Blue1)

  #########################################################################################################################
  Red2 =   []
  Gren2 =  []
  Blue2 =  []

  for index,i in enumerate(Class2):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red2.append(NewImage[0][0][0])
    Gren2.append(NewImage[0][0][1])
    Blue2.append(NewImage[0][0][2])

  muR2 = np.mean(Red2)
  muG2 = np.mean(Gren2)
  muB2 = np.mean(Blue2)

  sigR2 = np.std(Red2)
  sigG2 = np.std(Gren2)
  sigB2 = np.std(Blue2)

  ########################################################################################################################
  Red3 =   []
  Gren3 =  []
  Blue3 =  []

  for index,i in enumerate(Class3):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red3.append(NewImage[0][0][0])
    Gren3.append(NewImage[0][0][1])
    Blue3.append(NewImage[0][0][2])

  muR3 = np.mean(Red3)
  muG3 = np.mean(Gren3)
  muB3 = np.mean(Blue3)

  sigR3 = np.std(Red3)
  sigG3 = np.std(Gren3)
  sigB3 = np.std(Blue3)

  ##############################################################################################################
  Red4 =   []
  Gren4=  []
  Blue4 =  []

  for index,i in enumerate(Class4):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red4.append(NewImage[0][0][0])
    Gren4.append(NewImage[0][0][1])
    Blue4.append(NewImage[0][0][2])

  muR4 = np.mean(Red4)
  muG4 = np.mean(Gren4)
  muB4 = np.mean(Blue4)

  sigR4 = np.std(Red4)
  sigG4 = np.std(Gren4)
  sigB4 = np.std(Blue4)

  ##########################################################
  Red5 =   []
  Gren5 =  []
  Blue5 =  []

  for index,i in enumerate(Class5):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red5.append(NewImage[0][0][0])
    Gren5.append(NewImage[0][0][1])
    Blue5.append(NewImage[0][0][2])

  muR5 = np.mean(Red5)
  muG5 = np.mean(Gren5)
  muB5 = np.mean(Blue5)

  sigR5 = np.std(Red5)
  sigG5 = np.std(Gren5)
  sigB5 = np.std(Blue5)

  ##########################################################
  Red6 =   []
  Gren6 =  []
  Blue6 =  []

  for index,i in enumerate(Class6):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red6.append(NewImage[0][0][0])
    Gren6.append(NewImage[0][0][1])
    Blue6.append(NewImage[0][0][2])

  muR6 = np.mean(Red6)
  muG6 = np.mean(Gren6)
  muB6 = np.mean(Blue6)

  sigR6 = np.std(Red6)
  sigG6 = np.std(Gren6)
  sigB6 = np.std(Blue6)

  ##########################################################
  Red7 =   []
  Gren7 =  []
  Blue7 =  []

  for index,i in enumerate(Class7):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red7.append(NewImage[0][0][0])
    Gren7.append(NewImage[0][0][1])
    Blue7.append(NewImage[0][0][2])

  muR7 = np.mean(Red7)
  muG7 = np.mean(Gren7)
  muB7 = np.mean(Blue7)

  sigR7 = np.std(Red7)
  sigG7 = np.std(Gren7)
  sigB7 = np.std(Blue7)

  ##########################################################
  Red8 =   []
  Gren8 =  []
  Blue8 =  []

  for index,i in enumerate(Class8):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red8.append(NewImage[0][0][0])
    Gren8.append(NewImage[0][0][1])
    Blue8.append(NewImage[0][0][2])

  muR8 = np.mean(Red8)
  muG8 = np.mean(Gren8)
  muB8 = np.mean(Blue8)

  sigR8 = np.std(Red8)
  sigG8 = np.std(Gren8)
  sigB8 = np.std(Blue8)

  ##########################################################
  Red9 =   []
  Gren9 =  []
  Blue9 =  []

  for index,i in enumerate(Class9):
    NewImage = resize(i,(1,1),anti_aliasing=True,order = 1)
    Red9.append(NewImage[0][0][0])
    Gren9.append(NewImage[0][0][1])
    Blue9.append(NewImage[0][0][2])

  muR9 = np.mean(Red9)
  muG9 = np.mean(Gren9)
  muB9 = np.mean(Blue9)

  sigR9 = np.std(Red9)
  sigG9 = np.std(Gren9)
  sigB9 = np.std(Blue9)
  mu = np.array([[muR0,muG0,muB0],[muR1,muG1,muB1],[muR2,muG2,muB2],[muR3,muG3,muB3],[muR4,muG4,muB4],[muR5,muG5,muB5],[muR6,muG6,muB6],[muR7,muG7,muB7],[muR8,muG8,muB8],[muR9,muG9,muB9]])
  sigma = np.array([[sigR0,sigG0,sigB0],[sigR1,sigG1,sigB1],[sigR2,sigG2,sigB2],[sigR3,sigG3,sigB3],[sigR4,sigG4,sigB4],[sigR5,sigG5,sigB5],[sigR6,sigG6,sigB6],[sigR7,sigG7,sigB7],[sigR8,sigG8,sigB8],[sigR9,sigG9,sigB9]])
  return mu, sigma,p




mu, sigma,p = cifar_10_naivebayes_learn(xp,TrainLabel)

mu.shape

sigma.shape

p.shape


from scipy.stats import norm
def cifar10_classifier_naivebayes(X,mu,Sigma,p):

  Class_C = np.zeros(len(X), dtype=object)
  for i in range(len(X)):
    
    Z = np.zeros(10,dtype=object)
    for j in range(len(mu)):

      Red =  X[i][0][0][0]-mu[j][0]
      Gren = X[i][0][0][1]-mu[j][1]
      Blue = X[i][0][0][2]-mu[j][2]

      NR = (1/(np.sqrt(2*np.pi*Sigma[j][0])))* (np.exp((-1/2)*(Red/Sigma[j][0])**2))
      NG = (1/(np.sqrt(2*np.pi*Sigma[j][1])))* (np.exp((-1/2)*(Gren/Sigma[j][1])**2))
      NB = (1/(np.sqrt(2*np.pi*Sigma[j][2])))* (np.exp((-1/2)*(Blue/Sigma[j][2])**2))
                                                                                        
      Num = NR*NG*NB
      Z[j] = Num
    Class_C[i] = np.argmax(Z)

  return Z, Class_C


Z, Class_C = cifar10_classifier_naivebayes(yp,mu,sigma,p)

print(Class_C)


# Define class accuracy fuction
def class_acc(pred, gt):

  y_actual = list(gt)
  predicted = list(pred)
  score = 0
    
  for i, j in zip(predicted, y_actual):
      if i == j:
          score += 1
    
  return round((score / len(y_actual))*100, 2),"%","The correct classified Classes are:", score



def class_acc(pred, gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct += 1
    return round((correct / len(pred)) * 100,1)



test = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
Actual_Lable = np.array(test['labels'],dtype = object)
class_acc(Class_C, Actual_Lable)


#################################################################################################################

###  Question 2  ####

def cifar_10_color(x):
    x = np.transpose(x, [0, 3, 1, 2])
    x = np.reshape(x, [x.shape[0], -1])  # 50000,3072
    r = x[0:50000, 0:1024] #50000x1024, the same as r=x[:, 0:1024]
    g = x[0:50000, 1024:2048] #50000x1024
    b = x[0:50000, 2048:3072] #50000x1024
    r_mean = np.mean(r, axis=1, keepdims=True) #50000x1
    g_mean = np.mean(g, axis=1, keepdims=True)  # 50000x1
    b_mean = np.mean(b, axis=1, keepdims=True)  # 50000x1
    train_images = np.concatenate((r_mean, g_mean, b_mean), axis=1) #50000x3 -> 10x50000x3
    return train_images
    

import warnings
warnings.filterwarnings("ignore")

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Test data. We are reading again to make it a littel simple and easier for us to read the variables. 

datadict = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')

X = datadict["data"]
Y = datadict["labels"]
labeldict = unpickle('/content/drive/MyDrive/Excercise_3/batches.meta')
label_names = labeldict["label_names"]

# the test images will be. 
test_images = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.int)
test_classes = np.array(Y)

# All training data are reading in the following way. 
Training_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
data = []
labels = []
for j in range(5):
    raw_data = unpickle('/content/drive/MyDrive/Excercise_3/' + Training_files[j])
    data.append(raw_data["data"])
    labels.append(raw_data["labels"])

train_images = np.concatenate(data)
train_classes = np.concatenate(labels)

train_images = train_images.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype(np.int)



train_Image = cifar_10_color(train_images)


def cifar_10_bayes_learn(Xf, label):
    ms = []
    cov = []
    for i in range(0, 10): # 10 classes to be split 
        idx = np.where(label == i)  # 
        class_i = Xf[idx]  

        mur = np.mean(class_i[:, 0]) 
        mug = np.mean(class_i[:, 1])
        mub = np.mean(class_i[:, 2])
        #
        ms.append([mur, mug, mub])
        covariance = np.cov(class_i.T)
        cov.append(covariance)
    p = len(class_i) / len(Xf)  # p: prior probability
    return [np.array(ms), np.array(cov), p]


[ms, cov, p] = cifar_10_bayes_learn(train_Image, train_classes)
print("The mean is = :\n", ms)
print("\nThe covariance is = :\n", cov)
print("\nThe covariance is = :\n", p)


def cifar10_classifier_bayes2(x, ms, cov, p):
    prob = np.zeros(10)
    for j in range(0, 10):
        p_total = stats.multivariate_normal.pdf(x, ms[j, :], cov[j, :, :]) 
        prob[j] = p_total*p
    cls = np.argmax(prob)
    return cls

def cifar10_classifier_bayes3(x, ms, cov, p): 
    prob = np.zeros([10000, 10])
    for i in range(0, 10):
        p_total = stats.multivariate_normal.logpdf(x, ms[i, :], cov[i, :, :]) 
        prob[:, i] = p_total*p
    cls = np.argmax(prob, axis=1)
    return cls


test_images2 = cifar_10_color(test_images) # 10000x3
#print(f"The shape of test_images is: {test_images.shape}")
pred_cls2 = np.zeros(10000)
for x in range(0, 10000):
    pred_cls2[x] = cifar10_classifier_bayes2(test_images2[x], ms, cov, p)
    #print(pred_cls2) 
    
    
test = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
Actual_Lable = np.array(test['labels'],dtype = object)


Actual_Lable = np.array(Actual_Lable)


print("The accuracy of Bayesian classifier is = :", class_acc(pred_cls2, Actual_Lable))

########################################################################################################################

                        ### Question 3 ### 

def cifar10_classifier_bayes3(x, ms, cov, p): #x: 1x3, cov:10x3x3
    prob = np.zeros([10000, 10])
    for i in range(0, 10):
        p_total = stats.multivariate_normal.logpdf(x, ms[i, :], cov[i, :, :]) #cov size 1x3x3
        prob[:, i] = p_total*p
    cls = np.argmax(prob, axis=1)
    return cls

def cifar10_2x2_color(images, size=(2, 2)):
    reshaped_image = np.reshape(images, [images.shape[0], 3, 32, 32]).transpose([0, 2, 3, 1])
    reshaped_image = np.array(reshaped_image, dtype='uint8')
    out = np.zeros([images.shape[0], size[0], size[1], 3])
    for i in range(images.shape[0]):
        out[i, :, :, :] = cv2.resize(reshaped_image[i, :, :, :], size)
    out = np.transpose(out, [0, 3, 1, 2])
    out = np.reshape(out, (out.shape[0], -1))
    return np.array(out, dtype=np.int)

def cifar_10_bayes_learn3(Xf, label):
    ms = []
    cov = []
    for i in range(0, 10):
        idx = np.where(label == i)  # get from label-vector ith entries indexes
        class_i = Xf[idx]  # get data according to gotten indexes # 5000x3
        mu = np.mean(class_i, axis=0)
        ms.append(mu)
        covariance = np.cov(class_i.T)
        cov.append(covariance)
    p = len(class_i) / len(Xf)  # p: prior probability
    return [np.array(ms), np.array(cov), p]




import warnings
warnings.filterwarnings("ignore")


Accuray = []
for i in range(0, 6):
    size = 2**i
    Xf3 = cifar10_2x2_color(train_images, size=(size, size))
    [ms, cov, p] = cifar_10_bayes_learn3(Xf3, train_classes)
    imagenes_test = cifar10_2x2_color(test_images, size=(size, size))  # 10000x3
    pred_cls3 = cifar10_classifier_bayes3(imagenes_test, ms, cov, p)
    accuracy_result = class_acc(pred_cls3, test_classes)
    Accuray.append(round(accuracy_result,1))


print("The accuracy for size 1x1 = :", Accuray[0])
print("The accuracy for size 2x2 = :", Accuray[1])
print("The accuracy for size 1x1 = :", Accuray[2])
print("The accuracy for size 1x1 = :", Accuray[3])
print("The accuracy for size 1x1 = :", Accuray[4])
print("The accuracy for size 1x1 = :", Accuray[5])
    
    #print(f"The Bayes accuracy for size {size} x {size} is: {accuracy_metric[i]}")

# GRAPH
x = ["1x1", "2x2", "4x4", "8x8", "16x16", "32x32"]
y = Accuray
plt.plot(x, y, "*:", color='b')
plt.xlabel('Shapes of All Images')
plt.ylabel('Accuracy of each Dimention')
plt.show()








