# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:07:07 2019

@author: hhaq
"""

#Source:https://www.tensorflow.org/tutorials/keras/basic_text_classification
#This example clssifies the movie reviews based on possitive or negative text
#IMDB dataset reviews contains 50,000 movie reviews from the internet movie database
#Training and testing dataset are balanced, meaning 25,000 positive and 25,000 negative reviews
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

len(train_data[0]), len(train_data[1])

##Converting integer back to text
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
########################################

decode_review(train_data[0])

#####Preparing the data to feed into neural network
#The array of integers must be converted to tensors 
#Converting the arrays into vectors of 0's and 1's indicating word occurrence 
#A Sequence [3, 5] would become a 10,000 dimensional vector that is all zeros except for indices 3 and 5, which are ones
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
len(train_data[0]), len(test_data[1])
print(train_data[0])


##############################Building the model 
######Stacking layers in neural network requires no. of layers to use? and hidden units to use for each layer?
vocab_size=10000  #movie review=10000
model= keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) #This layer looks up embedding vector for each word-index
model.add(keras.layers.GlobalAveragePooling1D())  #This layer returns a fixed-length output vector by averaging over the sequence dimensions
model.add(keras.layers.Dense(16, activation=tf.nn.relu))  #This layer is connected to 16 hidden units
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))  #this layer is connected with a single output node. Sigmoid activation function provides a floating value between 0 and 1, representing a probability or confidence level    

model.summary()

######Configuring the model 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


##########Creating a validation set 
#Accuracy of the model is checked by model on the data it hasn't seen before.
#Validation set separate 10,000 examples from the original training data.
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]



##Train the model 
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#Evaluate the model
results = model.evaluate(test_data, test_labels)

print(results)

####Graph between accuracy and loss in time
history_dict = history.history
history_dict.keys()

##############Plot
import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

####Loss graph
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure

####Accuracy graph
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

