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

tf.enable_eager_execution()

img_num_channels = 3
img_width_height = 192
input_dir = './input/imgs/'

classes = []
paths = []
folders = os.listdir(input_dir)
for i in range(0, len(folders)):
	folder = folders[i]
	if '.DS_Store' not in folder: # osx automatically created file
		files = os.listdir(os.path.join(input_dir, folder))

		for file in files:
			if '.DS_Store' not in file: # osx automatically created file
				paths.append(os.path.join(input_dir, folder, file))
				classes.append(i-1)

train_length = len(paths) * 0.8
filenames = tf.constant(paths)
labels = tf.constant(classes)

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
def read_files(filename, label):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=img_num_channels)
    # All images must have the same size
    image = tf.image.resize_images(image, size=[img_width_height, img_width_height], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image /= 255.0
    
    return image, label

dataset = dataset.map(read_files)
dataset = dataset.shuffle(buffer_size=len(paths)) # Add some randomless
# Split into training and testing
trainds = dataset.take(int(train_length))
testds = dataset.skip(int(train_length))

def show_image():
	for n,(image, label) in enumerate(trainds.take(1)):
		plt.imshow(image)
		plt.show()

trainds = trainds.repeat()
trainds = trainds.batch(32)

# define model
model = tf.keras.Sequential()
# Input layer must know which is the input shape
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu, input_shape=[img_width_height, img_width_height, img_num_channels]))
model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])

print(len(model.trainable_variables))
print(model.summary())


'''
model.fit(train_x, train_y, epochs=10)

# show training results
scores = model.predict(test_x)
print(np.argmax(scores)) '''