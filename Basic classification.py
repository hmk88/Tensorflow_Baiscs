# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:28:05 2019

@author: hhaq
"""

################Source:https://www.tensorflow.org/tutorials/keras/basic_classification
###########Basic classification: Training neural network 
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)   ##checking tensorflow version= 1.13.1

##Importing fashion MNIST dataset. It contains 70,000 greyscale images in 10 categories.
##The images show individual articles of clothing at low resolution 28x28 pixels
##We will use 60,000 images to train the network and 10,000 images to evaluate how
##accurately the network learned to classify images. 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
train_labels.shape

##Pre-processing the the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

###Scale the pixel value in a range from 0 to1 before feeding it to neural network 
###Divide by 255
train_images = train_images/255
test_images = test_images/255


##Displaying the first 25 images from the training set along with the class name below each image
##Verifying that the data is in the correct order 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


##Layer configuration 
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  #transform the format of the image from 2d(28x28 pixels) to a 1d (28x28= 784 pixels)
        keras.layers.Dense(128, activation=tf.nn.relu), #Dense layer has 128 nodes(neurons). Each node contains a score that indicates the probability that the current image belongs to one of the 128 classes
        keras.layers.Dense(10, activation=tf.nn.softmax) #Has 10 nodes softmax layer.
])

##Compiling the model 
model.compile(optimizer='adam', #This is how the model is updated 
              loss='sparse_categorical_crossentropy',   #This measure how accurate the model is during training
              metrics=['accuracy']) #Use to monitor the train and test steps. 

###Training the model 
model.fit(train_images, train_labels, epochs=5)


##Accuracy 
test_loss, test_acc = model.evaluate(test_images, test_labels)

##Test accuracy here is less than the train accuracy of data. This is an example of over-fitting. It is when machine learning performs worse on new data than on training data 
####Prediction
prediction = model.predict(test_images)
prediction[0]

np.argmax(prediction[0])
###the prediction is 9 which has the class name of ankle boot


####################Making graphs of full set of 10 channels 
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
###############################################
  
##Displaying the 1st image's prediction and prediction array 
i = 1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction,  test_labels)
plt.show()


###Plotting multiple images
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, prediction, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, prediction, test_labels)
plt.show()


##Making a prediction for single image 
img=test_images[0]
img.shape

##adding image to a batch where its the only member 
img = (np.expand_dims(img,0))
img.shape

#####Now predict the image 
prediction_single = model.predict(img)
prediction_single

##Plot
plot_value_array(0, prediction_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(prediction_single)



