import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

tf.enable_eager_execution()

print(tf.VERSION)

input_dir = './input/imgs/'
files = os.listdir(input_dir)
if '.DS_Store' in files:
	founded_index = files.index('.DS_Store')
	del files[founded_index]

paths = []
for file in files:
	if '.DS_Store' in file:
		del file
	else:
		paths.append(input_dir + file)

del files

# step 1
filenames = tf.constant(paths)
labels = tf.constant([0, 1])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

def read_files(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image /= 255.0
    
    return image

dataset = dataset.map(read_files)

for n,image in enumerate(dataset.take(4)):
  plt.imshow(image)
  plt.show()


'''
# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu, input_shape=2))

# train model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10)

# show training results
scores = model.predict(test_x)
print(np.argmax(scores)) '''