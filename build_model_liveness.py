import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("dataset/train",
                                          target_size=(250, 250),
                                          batch_size=2,
                                          class_mode='binary')

test_dataset = test.flow_from_directory("dataset/val",
                                        target_size=(250, 250),
                                        batch_size=2,
                                        class_mode='binary')
print(test_dataset.class_indices)


model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(16,(3,3),activation='relu', padding="same", input_shape=(250,250,3)))
model.add(keras.layers.AvgPool2D(2,2))

model.add(keras.layers.Conv2D(16,(3,3),activation='relu', padding="same"))
model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Conv2D(32,(3,3),activation='relu', padding="same"))
model.add(keras.layers.AvgPool2D(2,2))

model.add(keras.layers.Conv2D(32,(3,3),activation='relu', padding="same"))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.AvgPool2D(2,2))# Convolutional layer and maxpool layer 3

model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
model.add(keras.layers.Dense(64,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog 
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#steps_per_epoch = train_imagesize/batch_size

history = model.fit_generator(train_dataset,
         steps_per_epoch = 1,
         epochs = 100,
         validation_data = test_dataset
         )
model.save('my_model_liveness.h5')

# summarize history for accuracy
accData = plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
accData.savefig('accuracy1.png')
plt.close(accData)
# summarize history for loss
lossData = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
lossData.savefig('loss1.png')
plt.close(lossData)
