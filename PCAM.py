# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:44:33 2020

@author: ADMIN
"""
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.metrics import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import*
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.utils import plot_model
import numpy
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt


#Initialize a model for loading and understanding the images and its types.
#Note:-This function takes jpg files as input and assumes to store the files in the foloowing format:
#A train,test and a valid folder each folder containg two other directores non tumorus and tumorus
class check_model(): 
    def __init__(self):
        self.train_path='D:\\datasets\\pcamminidataset\\train'
        self.test_path='D:\\datasets\\pcamminidataset\\test'
        self.valid_path='D:\\datasets\\pcamminidataset\\validate'
       
    #Since the pcam data set is huge we can load images in batches using keras inbuilt function
    #Note:-Dont try to load the entire data set ,use batch size according to your system
    def load_data(self):
        self.train_batches=ImageDataGenerator().flow_from_directory(self.train_path,classes=['nontumorus','tumorus'],batch_size=10)
        self.test_batches=ImageDataGenerator().flow_from_directory(self.test_path,classes=['nontumorus','tumorus'],batch_size=10)
        self.valid_batches=ImageDataGenerator().flow_from_directory(self.valid_path,classes=['nontumorus','tumorus'],batch_size=10)
    
    #Plot function is taken from github
    def plots(self,ims, figsize=(12,6), rows=1, interp=False, titles=None):
        if type(ims[0]) is numpy.ndarray:
            ims = numpy.array(ims).astype(numpy.uint8)
            if (ims.shape[-1] != 3):
                ims = ims.transpose((0,2,3,1))
        f = plt.figure(figsize=figsize)
        cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
        for i in range(len(ims)):
            sp = f.add_subplot(rows, cols, i+1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i], interpolation=None if interp else 'none')
    
    def plot_images(self):
        img,labels=next(self.valid_batches)
        self.plots(img,titles=labels)
        
        
class pcam_model():
    def __init__(self):
        self.trainimagepath='D:\\datasets\\abda\\datafolder\\camelyonpatch_level_2_split_train_x.h5'  
        self.trainlabelpath='D:\\datasets\\abda\\labeldata\\camelyonpatch_level_2_split_train_y.h5'
      
        
        self.testimagepath='D:\\datasets\\abda\\datafolder\\camelyonpatch_level_2_split_test_x.h5'
        self.testlabelpath='D:\\datasets\\abda\\labeldata\\camelyonpatch_level_2_split_test_y.h5'
        
        
        self.validimagepath='D:\\datasets\\abda\\datafolder\\camelyonpatch_level_2_split_valid_x.h5'
        self.validlabelpath='D:\\datasets\\abda\\labeldata\\camelyonpatch_level_2_split_valid_y.h5'
        
        
    def read_hdf_file(self,path,key):
        x=HDF5Matrix(path,key)
        return x
        
        
    def load_data(self):
        self.x_train=self.read_hdf_file(self.trainimagepath,'x')
        self.y_train=self.read_hdf_file(self.trainlabelpath,'y')
        self.x_test=self.read_hdf_file(self.testimagepath,'x')
        self.y_test=self.read_hdf_file(self.testlabelpath,'y')
        self.x_valid=self.read_hdf_file(self.validmagepath,'x')
        self.y_valid=self.read_hdf_file(self.validlabelpath,'y')
        print("The number of training imgaes " ,len(self.x_train), " each of size",self.x_train[0].shape)
        print("The number of training labels " ,len(self.y_train), " each of size",self.y_train[0].shape)
        print(" ")
        print("The number of test imgaes " ,len(self.x_test), " each of size",self.x_test[0].shape)
        print("The number of test labels " ,len(self.y_test), " each of size",self.y_test[0].shape)
        print(" ")
        print("The number of validation imgaes " ,len(self.x_valid), " each of size",self.x_valid[0].shape)
        print("The number of validation labels " ,len(self.y_valid), " each of size",self.y_valid[0].shape)
        
        
    def define_model(self,x_input,y_input,z_input):
        self.model=Sequential()
        self.model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='glorot_uniform',input_shape=(x_input,y_input,z_input)))
        self.model.add(Conv2D(64,(3,3),activation='relu'))
        self.model.add(Conv2D(128,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(3,3))
        self.model.add(Conv2D(128,(3,3),padding='valid',activation='relu'))
        self.model.add(Conv2D(256,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(3,3))
        self.model.add(Conv2D(256,(3,3),activation='relu'))
        self.model.add(Conv2D(512,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(2,2))
        self.model.add(Flatten())
        self.model.add(Dense(4096,activation='relu'))
        self.model.add(BatchNormalization(momentum=0.0))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2048,activation='relu'))
        self.model.add(Dense(1024,activation='relu'))
        self.model.add(Dense(2,activation='softmax'))
        
        
    def reshape_labels(self,y):
        labels=[]
        for value in y:
            if(value[0][0]==1):
                labels.append([1.0,0.0])
            else:
                labels.append([0.0,1.0])
        return numpy.array(labels)
    

    def pre_process_data(self):
        self.x_train=numpy.array(self.x_train)
        self.y_train=numpy.array(self.y_train)
        self.x_test=numpy.array(self.x_test)
        self.y_test=numpy.array(self.y_test)
        self.x_valid=numpy.array(self.x_valid)
        self.y_valid=numpy.array(self.y_valid)
        self.y_train=self.reshape_labels(self.y_train)
        self.y_valid=self.reshape_labels(self.y_valid)
        
    def compile_and_fit_predict(self):
        self.model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
        self.history=self.model.fit(self.x_train,self.y_train,batch_size=32,epochs=50,verbose=2)
        self.x_test=self.x_test.astype(float)
        self.y_pred=self.model.predict(self.x_test,batch_size=16,verbose=2)
        self.predictions=[]
        
        #Check is prob(tumorus)>prob(non_tumorus)
        for prob in self.y_pred:
            if(prob[0]>prob[1]):
                self.predictions.append(1)
            else:
                self.predictions.append(0)
                
        self.predictions=numpy.array(self.predictions)
        
        self.test_labels=[]
        for value in self.y_test:
            if(value[0][0]==1):
                self.test_labels.append(1)
            else:
                self.test_labels.append(0)
                
                
    #plot_confusion_matrix taken from git hub            
    def plot_confusion_matrix(self,cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
    def test_analysis(self):
        cnf=confusion_matrix(self.test_labels,self.predictions)
        cm_plot_labels=['nontumorus','tumorus']
        self.plot_confusion_matrix(cnf,cm_plot_labels,title='Confusion matrix')
        acc=accuracy_score(self.test_labels,self.predictions)
        cf=classification_report(self.test_labels,self.predictions)
        print(acc)
        print(cf)
        
        
    def train_analysis(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        
    
        
        

        
    
        
        
    
        
    
        
    
        
    
        
        
    

    
    

        
    
        
        
        
        
        
    
        
        
        
    

                
