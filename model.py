##################################################
## Creates a images weather model
##################################################
## MIT License
##################################################
## Author: Joaquin Sanchez
## Copyright: Copyright 2019, Weather ML Model
## License: MIT License
## Version: 0.0.1
## Maintainer: Joaquin Sanchez
## Email: sanchezjoaquin1995@gmail.com
## Status: Development
##################################################

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

img_num_channels = 3
img_width_height = 192
input_dir = os.path.join('.', 'input', 'imgs')

num_classes = 3
classes = []
paths = []
folders = os.listdir(input_dir)
classindex = 0
for folder in folders:
	if '.DS_Store' not in folder: # osx automatically created file
		files = os.listdir(os.path.join(input_dir, folder))

		for file in files:
			if '.DS_Store' not in file: # osx automatically created file
				paths.append(os.path.join(input_dir, folder, file))
				label = np.zeros(num_classes)
				label[classindex] = 1.0
				classes.append(label)

		classindex += 1

x = []
y = []
for i in range(0, len(paths)):
	path = paths[i]

	image = cv2.imread(path)
	image = cv2.resize(image, (img_width_height, img_width_height), 0, 0, cv2.INTER_LINEAR)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)

	x.append(image)
	y.append(classes[i])

# Splitting into training and testing
training_size = 0.8
training_size = round(len(x)*training_size)

train_x = x[:training_size]
train_y = y[:training_size]
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = x[training_size:]
test_y = y[training_size:]
test_x = np.array(test_x)
test_y = np.array(test_y)

# define model
model = tf.keras.Sequential()
# Input layer must know which is the input shape
model.add(tf.keras.layers.Conv2D(256, (3,3), input_shape=(192, 192, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(256, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# Always flatten before fully connected layers
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64))

# output layer must have the same units as number of classes
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('sigmoid'))

# loss categorical_crossentropy needs labels of type [0 1 0] (here class 2 is the right one)
model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])

print(model.summary())

# train the model
print('Training...')
model.fit(train_x, train_y, epochs=10)

# test the training
print('Testing...')
model.evaluate(test_x, test_y)

# show training results
scores = model.predict(test_x)
print(np.argmax(scores))
