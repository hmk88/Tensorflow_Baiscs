# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:45:15 2019

@author: hhaq
"""

#Prediction of fuel efficiency: Regression 
##############Source: https://www.tensorflow.org/tutorials/keras/basic_regression
#Using Auto-MPG dataset and building a model to predict the fuel efficiency of 1970's and 80's auto mobile 
#To do this, we provide description of automobile such as cylinder, displacement, horse power and weight 
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#######Downloading the data
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

##Data import 
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

#######Clean the data since there are a few unknown values 
dataset.isna().sum()
#Dropping these rows
dataset = dataset.dropna()
#Currently the data is in categorical form and not in numerical for. 
#Converting the data to numerical form 
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

#Splitting the data into training and testing sets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Inspect the data 
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

#Checking statistics of data 
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#Separating features from labels 
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#Normailizing the data since features are in different scale 
#It is a good practice to normalize the data which use different scale and range 
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
##This normalized data will be used to train the data 

##Building sequential model with two densely connected hidden layers and an output layer which returns a single and continuous value 
def build_model():
    model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

model.summary()


#Trying out the model by taking 10 examples from the dataset
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result
#It seems to be working 

#training the model with 1000 epochs, recording the training and validation accuracy in the history object
#Since there are 1000 epochs, displaying it with a single dot 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#Visualizing model stats which are stored in history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

###Graph
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)
#Graph shows drop in the mean square error from 0 to 100 epochs and then a little rise after that 
#let's update the model with model.fit call to automatically stop training when validation score doesn't improve 
#Using EARLYSTOPPING callback that tests a training condition for each epoch. If a set amount of epochs elapses without showing improvement, automatically stop
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

#######Let's see how well the model did. This gives us an idea of what can be expected from the model to perdict when using in real world application 
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

###Predicting MPG values using data in testing set 
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

####Error distribution
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

