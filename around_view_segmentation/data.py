import os
import cv2
import numpy as np
import pickle
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def data_generator(train_path, batch_size):
	data_paths = glob.glob(os.path.join(train_path, 'data/*.jpg'))
	label_paths = glob.glob(os.path.join(train_path, 'label/*.jpg'))

	train_len = len(data_paths)

	if len(data_paths) != len(label_paths):
		print "the number of training data and the number of training labels are different."
		raise AssertionError()
	
	while True:
		data_paths = shuffle(data_paths)

		data = []
		label = []

		for data_path in data_paths:
			img = cv2.imread(data_path)
			img = cv2.resize(img, (224, 360), interpolation=cv2.INTER_AREA)
			gt = cv2.imread(os.path.join(train_path, os.path.join('label', data_path.split('/')[-1])))
			gt = cv2.resize(gt, (224, 360), interpolation=cv2.INTER_AREA)
			if gt is None:
				print "{} does not exist".format(os.path.join(train_path, os.path.join('label', data_paths[index].split('/')[-1])))
				raise AssertionError()
			data.append(img)
			label.append(gt)

			if len(data) == batch_size:
				X = np.array(data, dtype='float64') / 255.
				Y = np.array(label, dtype='float64') / 255.
				data = []
				label = []
				yield X, Y

def val_data(validation_path):
	data_paths = glob.glob(os.path.join(validation_path, 'data/*.jpg'))
	label_paths = glob.glob(os.path.join(validation_path, 'label/*.jpg'))

	if len(data_paths) != len(label_paths):
		print "the number of validation data and the number of validation labels are different."
		raise AssertionError()

	data = []
	label = []

	for i in range(len(data_paths)):
		img = cv2.imread(data_paths[i])
		img = cv2.resize(img, (224, 360), interpolation=cv2.INTER_AREA)
		gt = cv2.imread(os.path.join(validation_path, os.path.join('label', data_paths[i].split('/')[-1])))
		gt = cv2.resize(gt, (224, 360), interpolation=cv2.INTER_AREA)
		if gt is None:
			print "{} does not exist".format(os.path.join(validation_path, os.path.join('label', data_paths[i].split('/')[-1])))
			raise AssertionError()
		data.append(img)
		label.append(gt)

	X = np.array(data, dtype='float64') / 255.
	Y = np.array(label, dtype='float64') / 255.

	return X, Y