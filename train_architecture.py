# Anmol Kapoor
# Pokemon Cards One Shot Recognition

import os, sys
import logging, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["PYTHONHASHSEED"] = '1'
os.environ["TF_CUDNN_DETERMINISTIC"] = '1'
os.environ["TF_DETERMINISTIC_OPS"] = '1'
import numpy as np
np.random.seed(1)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import tensorflow as tf
tf.random.set_seed(1)
tf.compat.v2.random.set_seed(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Dropout, Flatten, GaussianNoise
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

training = True
w, h = 300, 420 # original 300, 420

databank = os.listdir("images/")

def convert(predictions):
	answers = []
	for pred in predictions.argmax(1):
		answers.append(databank[pred])
	return answers

if training:
	# Set the seed to get reproducible results
	#np.random.seed(1)
	#tf.random.set_seed(1)
	#random.seed(1)

	files = os.listdir("images/")
	for ind, file in enumerate(files):
		print(ind, "\t", file)

	images = []
	for file in files:
		images.append(Image.open(f"images/{file}"))

	dataset = np.array([np.asarray(img.resize((w, h))) for img in images])
	num_classes = len(dataset)
	labels = list(range(num_classes))
	aug_dataset = dataset.copy()
	aug_labels = np.array(list(range(num_classes)))

	data_augmentor = Sequential()
	data_augmentor.add(preprocessing.RandomRotation(0.00833, seed=1)) # rotate randomly between -3 and 3 degrees (~0.00833% of full circle)
	data_augmentor.add(preprocessing.Rescaling(1./255))
	data_augmentor.add(preprocessing.RandomContrast(0.3, seed=1))
	data_augmentor.add(GaussianNoise(0.7))

	for i in range(15): # create n+1 copies of each image total (don't forget the original)
		aug_dataset = np.append(aug_dataset, data_augmentor(dataset).numpy(), 0)
		aug_labels = np.append(aug_labels, list(range(num_classes)))

	print(aug_dataset.shape)
	print(aug_labels.shape)

	model = Sequential()
	model.add(Conv2D(128, input_shape=(h, w, 3), kernel_size=(5,5)))
	model.add(AveragePooling2D(pool_size=(5, 5)))
	model.add(Flatten())
	model.add(Dense(len(dataset), activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	#train_ds = Dataset.from_tensor_slices((aug_dataset, aug_labels))
	history = model.fit(aug_dataset, to_categorical(aug_labels), epochs=20)

	print("Saving Model")
	model.save("poke.h5")
elif not training and "poke.h5" in os.listdir():
	model = load_model("poke.h5")
	test_images = []
	for file in os.listdir("test/"):
		test_images.append(Image.open(f"test/{file}"))

	test_dataset = np.array([np.asarray(img.resize((w, h))) for img in test_images])
	print("File:       ", os.listdir("test/"))
	start_time = time.time()
	preds = model.predict(test_dataset)
	elapsed_time = time.time() - start_time
	print("Predictions:", convert(preds))
	print(f"Took {time.time() - start_time} seconds to generate predictions.")
