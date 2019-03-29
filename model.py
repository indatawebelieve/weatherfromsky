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

#tf.enable_eager_execution()

img_num_channels = 3
img_width_height = 192
input_dir = './input/imgs/'

num_classes = 2
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
				classes.append(classindex)

		classindex += 1

train_x = []
train_y = []
for i in range(0, len(paths)):
	path = paths[i]

	image = cv2.imread(path)
	image = cv2.resize(image, (img_width_height, img_width_height), 0, 0, cv2.INTER_LINEAR)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)

	train_x.append(image)
	train_y.append(classes[i])

train_x = np.array(train_x).reshape(-1, img_width_height, img_width_height, 3)
train_y = np.array(train_y)

print(train_y)

# define model
model = tf.keras.Sequential()
# Input layer must know which is the input shape
model.add(tf.keras.layers.Conv2D(256, (3,3), input_shape=train_x.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(256, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])

print(len(model.trainable_variables))
print(model.summary())


model.fit(train_x, train_y, epochs=10)

'''
# show training results
scores = model.predict(test_x)
print(np.argmax(scores)) '''