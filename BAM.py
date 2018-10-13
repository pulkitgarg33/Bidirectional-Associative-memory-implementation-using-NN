# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:50:20 2018

@author: pulki
"""


class BAM(object):
    def __init__(self, alpha,beta):
        for i in range(len(alpha)):
            alpha[i] = self.bipolar_conv(alpha[i])
            beta[i] = self.bipolar_conv(beta[i])
        self.M = np.zeros( (len(beta[0]) , len(alpha[0]) )) 
        self.__create_bam(alpha , beta)


    def __create_bam(self , alpha , beta):
        for i in range(len(alpha)):
            self.M += (beta[i].reshape(len(beta[0]) ,1)) @ (alpha[i].reshape(1,len(alpha[0])))

    def get_assoc(self, A):
        A = self.bipolar_conv(A)
        res = self.M @ (A.reshape(len(A) , 1))
        res = self.threshold(res)
        return res
    
    
    def get_bam_matrix(self):
        return self.M


    def threshold(self, vec):
        for i in range(len(vec)):
            if vec[i] > 0:
                vec[i] = 1
            else:
                vec[i] = -1
        return vec

    def bipolar_conv(self, vec):
        for i in range(len(vec)):
            if vec[i] > 0.5:
                vec[i] = 1
            else:
                vec[i] = -1
        return vec

    
#import numpy as np
#import pandas as pd   
#x= []
#y =[]
#
#x.append([1, 0, 1, 0, 1, 0])
#x.append([1, 1, 1, 0, 0, 0])
#y.append([1, 1, 0, 0])
#y.append([1, 0, 1, 0])
#
#x = np.array(x)
#y = np.array(y)
#
#b = BAM(x , y)
#
#y = b.get_assoc(x[0])


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

#importing the dataset
dataset = pd.read_csv('train.csv')
x_test = np.array(pd.read_csv('test.csv'))

x = np.array(dataset.iloc[: , 1:])
y = keras.utils.to_categorical(np.array(dataset.iloc[: , 0]))

image_rows = 28
image_colm = 28
num_classes = 10

from sklearn.model_selection import train_test_split
x_train , x_val , y_train , y_val = train_test_split( x , y , test_size = 0.2)



x_train = x_train.reshape( x_train.shape[0] , image_rows , image_colm , 1)
x_test = x_test.reshape( x_test.shape[0] , image_rows , image_colm , 1)
x_val = x_val.reshape( x_val.shape[0] , image_rows , image_colm , 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train = x_train / 255
#x_test = x_test / 255
#x_val = x_val / 255
