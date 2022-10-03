# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:06:52 2022

@author: Musarat
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_1')
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

print(X.shape)

labeldict = unpickle('/content/drive/MyDrive/Excercise_3/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
Y = np.array(Y)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)
    
############################################################################
# 2. CIFAR-10– Evaluation

pred = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
predLab = np.array(pred['labels'])

def class_acc(pred, gt):

  y_actual = list(gt)
  predicted = list(pred)
  score = 0
    
  for i, j in zip(predicted, y_actual):
      if i == j:
          score += 1
    
  return round((score / len(y_actual))*100, 2),"%","The correct classified images are:", score

class_acc(Y,Y) # for perfect accuracy

class_acc(predLab,Y)

#############################################################################

# 3. CIFAR-10 – Random classifier

x = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
def cifar10_classifier_random(x):
    lis = []
    x = x['labels']
    for i in range(len(x)):

      ranVal = random.choice(x)
      lis.append(ranVal)
    return lis

lis = cifar10_classifier_random(x)

class_acc(lis,Y)

##############################################################################
# 4. CIFAR-10 – 1-NN classifier

x = unpickle('/content/drive/MyDrive/Excercise_3/test_batch')
x = np.array(x["data"],dtype = np.uint16)

xtr1 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_1')
xtran1 = np.array(xtr1['data'],dtype = np.uint16)
trlabels1 = np.array(xtr1['labels'],dtype = np.uint16)

xtr2 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_2')
xtran2 = xtr2['data']
trlabels2 = np.array(xtr2['labels'],dtype = np.uint16)


xtr3 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_3')
xtran3 = np.array(xtr3['data'],dtype = np.uint16)
trlabels3 = np.array(xtr3['labels'],dtype = np.uint16)


xtr4 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_4')
xtran4 = np.array(xtr4['data'],dtype = np.uint16)
trlabels4 = np.array(xtr4['labels'],dtype = np.uint16)



xtr5 = unpickle('/content/drive/MyDrive/Excercise_3/data_batch_5')
xtran5 = np.array(xtr5['data'],dtype = np.uint16)
trlabels5 = np.array(xtr5['labels'],dtype = np.uint16)


xtran = np.concatenate((xtran1,xtran2,xtran3,xtran4,xtran5),dtype = np.uint16)
trlables = np.concatenate((trlabels1,trlabels2,trlabels3,trlabels4,trlabels5),dtype = np.uint16)


def cifar10_classifier_1nn(x,xtran,trlabels):
    predicted = []
    IndTrlabel = []
    label = []
    for ind,i in enumerate(x):
        minimum = []
        for index,j in enumerate(xtran):
            mini = np.linalg.norm(i-j)
            minimum.append(mini)
        IndTrlabel.append(minimum.index(min(minimum)))
    
    for ind,val in enumerate(IndTrlabel):
        predicted.append(trlabels[val])
    
    return predicted     

predicted = cifar10_classifier_1nn(x,xtran,trlables) # these are the predicted label using 1nn classifier.

x = unpickle('/content/drive/MyDrive/Excercise_3/test_batch') # reading again just for accuracy purpose. 
xlable = np.array(x['labels'],dtype= np.uint16)

class_acc(predicted,xlable)

#############################################################################















