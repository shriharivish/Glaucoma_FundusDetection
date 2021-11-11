import numpy as np
from keras import layers
from keras import Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


import shutil
import os
import keras.backend as K
from keras.utils import to_categorical
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
 

file_path = '/content/TrialGenerator/Train'
imgs = os.listdir(file_path)
X_train_orig=[]
Y_train_orig=[]

i=1
for f in imgs:
  #print("image count = "+i)
  
  if f[0] != '.':
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

    i+=1
 
X_train_orig = np.array(X_train_orig)
Y_train_orig = np.array(Y_train_orig)
Y_train = np.reshape(Y_train_orig,(-1,1))
 
file_path = '/content/TrialGenerator/Test'
imgs = os.listdir(file_path)
X_test_orig=[]
Y_test_orig=[]
 
for f in imgs:
  if f[0] != '.':
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

Y_train = to_categorical(Y_train_orig)
Y_test = to_categorical(Y_test_orig)

model = Sequential()

model.add(Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1a', kernel_initializer="glorot_uniform", padding="same", activation="relu", input_shape=(224,224,3)))
model.add(BatchNormalization(axis = 3, name = 'bn1a'))


model.add(Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1b', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn1b'))

model.add(MaxPooling2D((2, 2), name='max_pool1'))

model.add(Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2a', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn2a'))

model.add(Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2b', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn2b'))

model.add(MaxPooling2D((2, 2), name='max_pool2'))

model.add(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3a', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn3a'))

model.add(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3b', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn3b'))

model.add(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3c', kernel_initializer="glorot_uniform", padding="same", activation="relu"))
model.add(BatchNormalization(axis = 3, name = 'bn3c'))

model.add(MaxPooling2D((2, 2), name='max_pool5'))
model.add(Flatten())

model.add(Dense(100, activation='relu', name='fc1', kernel_initializer="glorot_uniform"))
model.add(Dense(10, activation='relu', name='fc2', kernel_initializer="glorot_uniform"))
model.add(Dense(2, activation='softmax', name='sm', kernel_initializer="glorot_uniform"))

model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ["accuracy"])


#performing data argumentation by training image generator
dataAugmentaion = ImageDataGenerator(brightness_range= [0.8,1.2], zoom_range = [0.8,1.1],fill_mode = "nearest", width_shift_range = 0.15, height_shift_range = 0.15, horizontal_flip=False)

bs=32
history = model.fit_generator(dataAugmentaion.flow(X_train, Y_train, batch_size = bs), steps_per_epoch = len(X_train)*60 // bs, epochs = 50)


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

preds = model.evaluate(x = X_test, y = Y_test)
 
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
