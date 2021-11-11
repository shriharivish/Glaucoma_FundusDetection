import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import glob #need to install this

import shutil
import random
import os
import keras.backend as K
from keras.utils import to_categorical
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
 
%matplotlib inline


to_be_moved = random.sample(glob.glob("/content/Glaucoma_images/Normal/*.jpg"), 230)
 
dest_folder = '/content/Glaucoma_images/train_data'

os.mkdir(dest_folder)

for f in to_be_moved :
  g = f.split('/')[4]
  dest = os.path.join(dest_folder,g)
  shutil.move(f,dest)
 
file_path = '/content/Glaucoma_images/Normal'
files = os.listdir(file_path)

dest = '/content/Glaucoma_images/test_data'

os.mkdir(dest)
for f in files:
  src = os.path.join(file_path,f)
  des = os.path.join(dest,f)
  shutil.move(src,des)

to_be_moved = random.sample(glob.glob("/content/Glaucoma_images/Normal/*.jpg"), 230)
 
dest_folder = '/content/Glaucoma_images/train_data'
for f in to_be_moved :
  g = f.split('/')[4]
  dest = os.path.join(dest_folder,g)
  shutil.move(f,dest)
 
file_path = '/content/Glaucoma_images/Normal'
files = os.listdir(file_path)
dest = '/content/Glaucoma_images/test_data'
for f in files:
  src = os.path.join(file_path,f)
  des = os.path.join(dest,f)
  shutil.move(src,des)


file_path = '/content/Glaucoma_images/train_data'
imgs = os.listdir(file_path)
X_train_orig=[]
Y_train_orig=[]
 
for f in imgs:
  num = f.split('.')[0].split('m')[1]
  num = int(num)
  
  img_path = os.path.join(file_path,f)
  img = image.load_img(img_path, target_size=(224, 224))
  img = image.img_to_array(img)
  X_train_orig.append(img)
  if num>255:
    Y_train_orig.append(1)
  else :
    Y_train_orig.append(0)
 
X_train_orig = np.array(X_train_orig)
Y_train_orig = np.array(Y_train_orig)
Y_train = np.reshape(Y_train_orig,(-1,1))
 
file_path = '/content/Glaucoma_images/test_data'
imgs = os.listdir(file_path)
X_test_orig=[]
Y_test_orig=[]
 
for f in imgs:
  num = f.split('.')[0].split('m')[1]
  num = int(num)
  
  img_path = os.path.join(file_path,f)
  img = image.load_img(img_path, target_size=(224, 224))
  img = image.img_to_array(img)
  X_test_orig.append(img)
  if num>255:
    Y_test_orig.append(1)
  else :
    Y_test_orig.append(0)
 
X_test_orig = np.array(X_test_orig)
Y_test_orig = np.array(Y_test_orig)
Y_test = np.reshape(Y_test_orig,(-1,1))


X_train = X_train_orig/255.
X_test = X_test_orig/255.


print(X_train.shape) #verify it should be (410,224,224,3)
print(Y_train.shape) #verify it should be (410,1)
print(X_test.shape) #verify it should be (45,224,224,3)
print(Y_test.shape) #verify it should be (45,1)




def Model_func(input_shape):
   
    X_input = Input(input_shape)
    X = ZeroPadding2D((1, 1))(X_input)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1a')(X)
    X = BatchNormalization(axis = 3, name = 'bn1a')(X)
    X = Activation('relu')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1b')(X)
    X = BatchNormalization(axis = 3, name = 'bn1b')(X)
    X = Activation('relu')(X)    
 
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2a')(X)
    X = BatchNormalization(axis = 3, name = 'bn2a')(X)
    X = Activation('relu')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2b')(X)
    X = BatchNormalization(axis = 3, name = 'bn2b')(X)
    X = Activation('relu')(X)  
 
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3a')(X)
    X = BatchNormalization(axis = 3, name = 'bn3a')(X)
    X = Activation('relu')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3b')(X)
    X = BatchNormalization(axis = 3, name = 'bn3b')(X)
    X = Activation('relu')(X)  
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3c')(X)
    X = BatchNormalization(axis = 3, name = 'bn3c')(X)
    X = Activation('relu')(X)
 
    X = MaxPooling2D((2, 2), name='max_pool3')(X)      
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv4a')(X)
    X = BatchNormalization(axis = 3, name = 'bn4a')(X)
    X = Activation('relu')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv4b')(X)
    X = BatchNormalization(axis = 3, name = 'bn4b')(X)
    X = Activation('relu')(X)  
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv4c')(X)
    X = BatchNormalization(axis = 3, name = 'bn4c')(X)
    X = Activation('relu')(X)
 
    X = MaxPooling2D((2, 2), name='max_pool4')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv5a')(X)
    X = BatchNormalization(axis = 3, name = 'bn5a')(X)
    X = Activation('relu')(X)
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv5b')(X)
    X = BatchNormalization(axis = 3, name = 'bn5b')(X)
    X = Activation('relu')(X)  
 
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv5c')(X)
    X = BatchNormalization(axis = 3, name = 'bn5c')(X)
    X = Activation('relu')(X)
 
    X = MaxPooling2D((2, 2), name='max_pool5')(X)            
 
    X = Flatten()(X)

    X = Dense(1000, activation='relu', name='fc1')(X)
  #  X = Dense(100, activation='relu', name='fc2')(X)
    X = Dense(1, activation='sigmoid', name='sg')(X)
    model = Model(inputs = X_input, outputs = X, name='GlaucomaModel')
    
    
    return model


glaucomaModel = Model_func(X_train.shape[1:])

glaucomaModel.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ["accuracy"])

glaucomaModel.summary()

glaucomaModel.fit(x = X_train , y = Y_train, epochs = 30, batch_size = 32)

preds = glaucomaModel.evaluate(x = X_test, y = Y_test)
 
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))