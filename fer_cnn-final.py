# -*- coding: utf-8 -*-
"""
@author: pavan soundara
"""
# Importing Keras  
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD, RMSprop
# Importing Numpy 
import numpy as np
from numpy import genfromtxt
np.random.seed(1337)
#%%
#importing training and test data from the files provided on kaggle and storing the data in variables
X_train = genfromtxt(r'datasets/train_data.csv', delimiter=',')
y_train = genfromtxt(r'datasets/train_target.csv', delimiter=',')
X_test = genfromtxt(r'datasets/test_data.csv', delimiter=',')

#%%
#Declare image resolution size or matrix form size
img_rows = 48
img_cols = 48

#batch_size to train
batch_size = 128
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20
#%%
#Reshaping inorder to fit the Tensorflow input 
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols , 1)

#Casting the t the test and train pixel d
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Dividing the data with 255 as per the highest number of color value
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)

#%%
model = Sequential()
#Adding the first and second convolution layers
model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(img_rows, img_cols , 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))
model.add(Dropout(0.1))
#Adding third ad 4th covolutional layers with maxpooling
model.add(Convolution2D(128, 5, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))
model.add(Dropout(0.1))
#Adding fifth ad 6th covolutional layers with maxpooling

model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))
model.add(Dropout(0.1))
#Fully Connected Layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes, activation='softmax'))
sgd = SGD(0.001)
rms=RMSprop(0.0003)
adam=Adam(lr=0.001)
#Compiling with optimization by adamn
model.compile(loss='categorical_crossentropy', optimizer=adam , metrics=['accuracy'])
print(model.summary())

#%%
# Training the model on the basics of train_data
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
             verbose=1, validation_split=0.1)
#%%
#Running predictions based on the data learned
predictions = model.predict(X_test, verbose=1)
output = predictions.argmax(axis=1)
# Opening sumbmission file to store the results of predictions
f = open('pavan_soundara_HW2_submission.csv','w')
# Writing the data fields in the submission file
f.write('Id,Category\n')
# Creating for loop to store entire results in the file
for i in range(0, X_test.shape[0]):
    f.write(str(i) + ',' + str(output[i]) + '\n')
# Closing the file after storing the results
f.close()

#%%
#saving weights
fname = "weights-new.hdf5"
model.save_weights(fname,overwrite=True)
#%%
#Loading saved weights
model.load_weights('weights-new.hdf5')



