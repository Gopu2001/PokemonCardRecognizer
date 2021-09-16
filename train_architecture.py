# Anmol Kapoor
# Pokemon Cards One Shot Recognition

# Imports Part 1
import os, sys, time, warnings, math, pprint
start_time = time.time()

# Constants

training = False
w, h = int(300 / 4), int(420 / 4) # original 300, 420
# note that all lines that use these variables to resize will be rewritten without them
# we will assume that all the images are already of size 300 x 420 for the sake of
# lessening the computation required

databank = os.listdir("images/")
databank_len = len(databank)
num_classes = databank_len  # num_classes is an alias of databank_len

num_epochs = 20
batch_size = 64
num_alterations = 20 # Deprecated and not supported
split_size = 1000
num_splits = math.ceil(databank_len / split_size)
train_iter = 1

'''
Helper Function Definitions
'''

def log(messages):
	passed_time = time.time() - start_time
	for message in messages.split("\n"):
		message = str(message).replace("\t", "    ")
		message_len = len(str(message))
		term_size = os.get_terminal_size()[0]
		one_chunk = term_size - 16
		if one_chunk < 0:
			warnings.warn("Unable to smart log message. Printing message normally to the console...", UserWarning)
			print(message)
			return
		if message_len > one_chunk:
			messages = [str(message)[i:i+one_chunk] for i in range(0, message_len, one_chunk)]
			for m in messages:
				if m != messages[-1]:
					print(m)
			print(messages[-1] + ' '*(term_size - len(messages[-1]) - 15) + '{:.11f}'.format(passed_time /    1)[:11] + ' sec')
		else:
			print(str(message) + ' '*(term_size - message_len - 15) + '{:.11f}'.format(passed_time /    1)[:11] + ' sec')

def format_dict(dictionary={}, row1="=", row2="~"):
	length = max([len(k) for k in dictionary.keys()], default=0)
	string = "{"
	for index, key in enumerate(dictionary.keys()):
		string += f"\n\t'{key}' "
		if index % 2 == 0:
			char = row1
		else:
			char = row2
		string += char * (length - len(str(key)) + 2)
		string += f"{char}> '{dictionary[key]}'"
		if index != len(dictionary.keys()) - 1:
			string += ","
	if len(dictionary.keys()) > 0:
		string += "\n"
	string += "}"
	return string

def stop():
	print("(Stopping Execution)")
	passed_time = time.time() - start_time
	print(f"Elapsed Time: {'{:.10f}'.format(passed_time /    1)} seconds.")
	print(f"Elapsed Time: {'{:.10f}'.format(passed_time /   60)} minutes.")
	print(f"Elapsed Time: {'{:.10f}'.format(passed_time / 3600)} hours.")
	sys.exit()

def convert(predictions):
	answers = []
	for pred in predictions.argmax(1):
		answers.append(databank[pred])
	return answers

def train_model(model, train_X, train_labels, epochs=num_epochs, batch_size=batch_size):
	assert(type(train_X) == list and type(train_X[0]) == str)
	# Here, load the images listed in the train_X list of filename labels
	X = np.array([np.asarray(Image.open(f"images/{file}").resize((w, h))) for file in train_X])
	log(f"Loaded {len(train_X)} images from the 'images/' folder as NumPy arrays.")
	model.fit(X, train_labels, epochs=num_epochs, batch_size=batch_size)
	return model

def test_model(model, test_dir, solution):
	assert(type(test_dir) == str and type(solution) in (list, np.ndarray))
	st_time = time.time()
	y_images = np.array([np.asarray(Image.open(test_dir + '/' + img).resize((w, h))) for img in os.listdir(test_dir)])
	y_predicted = convert(model.predict(y_images))
	el_time = time.time() - st_time
	log(f"Accuracy Score against {test_dir}'s images : {accuracy_score(solution, y_predicted)}")
	log(f"Evaluated model in {str(format(el_time       , '0.10f')).ljust(12)[:12]} seconds.")
	log(f"Evaluated model in {str(format(el_time / 60  , '0.10f')).ljust(12)[:12]} minutes.")
	log(f"Evaluated model in {str(format(el_time / 3600, '0.10f')).ljust(12)[:12]} hours.")

def get_preprocessing_model():
	# Augment the data to be slightly different
	# No trainable parameters
	data_augmentor = Sequential()
	#data_augmentor.add(preprocessing.RandomRotation(0.00833, seed=1, input_shape=(h, w, 3))) # rotate predictably randomly between -3 and 3 degrees (~0.00833% of full circle)
	data_augmentor.add(preprocessing.RandomRotation(0.01944, seed=1, input_shape=(h, w, 3))) # rotate predictably randomly between -6.5 and 6.5 degrees (~0.01944% of full circle)
	data_augmentor.add(preprocessing.Rescaling(1./255))
	data_augmentor.add(preprocessing.RandomContrast(0.4, seed=1))
	data_augmentor.add(GaussianNoise(1.5))
	return data_augmentor

def get_model(preprocessing=[]):
	assert(type(preprocessing) == list)
	if len(preprocessing) == 0:
		model = Sequential()
	else:
		model = Sequential(preprocessing)
	model.add(Conv2D(128, input_shape=(h, w, 3), kernel_size=(5,5)))
	model.add(AveragePooling2D(pool_size=(3, 3)))
	model.add(Conv2D(128, kernel_size=(3,3))) # New Layer to reduce final Dense Layer problem
	model.add(AveragePooling2D(pool_size=(3, 3))) # New Layer to reduce final Dense Layer problem
	model.add(Dropout(0.15)) # New Layer to reduce final Dense Layer problem
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax')) ## Problem fixed with above model edits
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Imports Part 2
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
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, GaussianNoise, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
##  #from tensorflow.data import Dataset
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # maybe doesn't do anything???
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
log("Successfully Imported and Configured Modules")

def main():
	if training:
		log("Starting the Training Procedure.")
		#files = os.listdir("images/")
		#images = [Image.open(f"images/{file}") for file in databank]
		#log(f"Loaded all {databank_len} images from the 'images/' folder.")

		# dataset = np.array([np.asarray(img.resize((w, h))) for img in images])
		#dataset = np.array([np.asarray(img) for img in images])
		#num_classes = len(databank)
		labels = list(range(num_classes))
		#log("Loaded numpy dataset without any modifications. (Assumed all images are 300 x 420 pixels)")

		# Augment the data to be slightly different
		# No trainable parameters
		data_augmentor = get_preprocessing_model()
		log("Created data augmentation model. Applying this model to the training model")

		# Define the model to be trained
		# Train Only Parameters
		model = get_model(preprocessing=[data_augmentor])
		model.summary(line_length=os.get_terminal_size()[0])
		log("Created the training model with the attached data augmentation model.")

		# The Data Generation Step has been combined with the training model. No need to manually generate images

		#log("Completing this training {train_iter} times")
		#for train_num in range(train_iter):
		log(f"Need to split the training into {num_splits} parts. This may take a while")
		for split_no in range(num_splits):
			model = train_model(
				model,
				databank[split_size * split_no : split_size * (split_no + 1)],											# dataset split into num_splits parts
				to_categorical(labels[split_size * split_no : split_size * (split_no + 1)], num_classes=num_classes),	# labels ordered in the same way as dataset
				epochs=num_epochs,																						# using same number of training epochs for every split
				batch_size=batch_size																					# using batch size 64 (was 32) unless OOM error occurs
			)
			log(f"Finished Training Split {split_no + 1} of {num_splits}!")
		log("Saving Model")
		model.save("poke.h5")

		# Start to Finish Time Statistics
		training_elapsed_time = time.time() - start_time
		log(f"Evaluated model in {str(format(training_elapsed_time / 1   , '0.10f')).ljust(12)[:12]} seconds.")
		log(f"Evaluated model in {str(format(training_elapsed_time / 60  , '0.10f')).ljust(12)[:12]} minutes.")
		log(f"Evaluated model in {str(format(training_elapsed_time / 3600, '0.10f')).ljust(12)[:12]} hours.")

		log("Testing the Just-Trained Model")
		test_model(model, 'images/', databank)

	elif not training and "poke.h5" in os.listdir():
		model = load_model("poke.h5")
		log("LOADED MODEL")
		test_model(model, 'images/', databank)
		#test_images = []
		#for file in os.listdir("test/"):
		#	test_images.append(Image.open(f"test/{file}"))

		#test_dataset = np.array([np.asarray(img.resize((w, h))) for img in test_images])
		##log(f"File:        {os.listdir('test/')}")
		#preds_time = time.time()
		#preds = model.predict(test_dataset)
		#elapsed_time = time.time() - preds_time
		#convert_time = time.time()
		##log(f"Predictions:  {convert(preds)}")
		#output = dict(zip(os.listdir('test/'), convert(preds)))
		#log(format_dict(output))
		#log(f"Took {str(elapsed_time)[:17]              } seconds to generate predictions.")
		#log(f"Took {str(time.time() - convert_time)[:17]} seconds to format   predictions.")

if __name__ == "__main__":
	main()
