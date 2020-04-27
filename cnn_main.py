#Dependancies:
#import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # will only output errors from the log
import tensorflow as tf
import argparse

#from tensorflow import keras
from keras import models, layers, losses

from util.print_progress import print_progress, showLossGraph
from util.change_to_array import changeToArray
from util.get_batches import createData, getBatches

#parsing console arguments:
"""
p = ArgumentParser()
p.add_argument('-n', '--name', required=True, help='name of file in "to_diagnose" to be diagnosed', type=str)
arguments = p.parse_args()

file_names = arguments.name.split(',') #parsing file names

file_names = ['test1.jpg', 'test2.jpg'] #TEMPORARY
"""
class MissingTrainingDataError(Exception): #new error for missing training data
    def __init__(self, arg):
        self.strerror = arg

if 'training images' not in os.listdir():
    raise MissingTrainingDataError("Missing the 'training images' directory, please create it as a placeholder")

#Import DATA
data = createData()
imageList = data['images']
labelsList = data['labels']


image_size = 128 #edit this to change size of images used

#creating model
model = models.Sequential()

model.add(layers.Conv2D(image_size, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(image_size * 2, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(image_size * 2, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(image_size * 2, activation='relu'))
model.add(layers.Dense(1))

#compiling and evaluating model
model.compile(optimizer='adam',
              loss = losses.BinaryCrossentropy(),
              metrics=['accuracy'])


train_images = [imageList[:round(0.75*(len(imageList)-1))]]
train_labels = [labelsList[:round(0.75*(len(labelsList)-1))]]

test_images = [imageList[round(0.75*(len(imageList)-1)):len(imageList)]]
test_labels = [labelsList[round(0.75*(len(labelsList)-1)):len(labelsList)]]

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

print("Optimization is done!")
showLossGraph(history.history['loss'])

redir = os.getcwd()
os.chdir('./to_diagnose')
n = 0

for i in os.listdir():
    if i != 'README.md':
        cur_image_array = [changeToArray(i, n)]

        if len(cur_image_array[0]) == image_size:
            prediction = model.predict_classes([cur_image_array])

            str_pred = "have pink eye" if prediction == 1 else "not have pink eye"
            print("Image '{}' is predicted to {}".format(i, str_pred))

        else:
            print("Image '{}' is not an acceptable size".format(i))
    
    n += 1

os.chdir(redir)