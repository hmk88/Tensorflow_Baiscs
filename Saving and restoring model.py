# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:56:24 2019

@author: hhaq
"""

#Source: https://www.tensorflow.org/tutorials/keras/save_and_restore_models
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

### Define a model 
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model


# Create a basic model instance
model = create_model()
model.summary()
    
### Save check points during training a model 
# tf.keras.callbacks.ModelCheckpoint is a callback that performs that action
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

###Create a new model
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#Loading weights from checkpoint and re-evaluate
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#checkpoints callback option 
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

###To reset the latest checkpoint 
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

####    Manually save and restore weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

### Saving an entire model 
#As an HDF5 file
model = create_model()
model.fit(train_images, train_labels, epochs=5)
# Save entire model to a HDF5 file
model.save('my_model.h5')

##  Restore an entire model 
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

##Checking restored model's accuracy 
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#############   Creating a fresh model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

saved_model_path = tf.contrib.saved_model.save_keras_model(model, "C:\\Users\\hhaq\\Documents\\GitHub\\Tensorflow_Baiscs")

############    Reloading the fresh model 
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

#####Run the restored model 
# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

