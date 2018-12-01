import sys
import numpy as np
import pandas as pd
import copy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical
from keras.optimizers import SGD


def XLSread(filename):
	data = pd.read_excel(filename)

	data_dict = {}

	for i in data.columns:
		temp_arr = np.array(data[i])
		data_dict[temp_arr[0]] = list(temp_arr[1:])

	return data_dict

filename = '../data/default of credit card clients.xls'
data_dict = XLSread(filename)

keys = [key for key in data_dict]

X = np.c_[np.ones(len(data_dict[keys[0]]))]

for i in range(0, len(keys) - 1):
	X = np.c_[X, data_dict[keys[i]]]

Y = data_dict[keys[-1]]

if (sys.argv[1] == "b") or (sys.argv[1] == 'All'):

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

	LogReg = LogisticRegression()
	LogReg.fit(X_train, Y_train)
	pred = np.array(LogReg.predict(X_test))

	#positive = [1 for i in pred if i[1]  > 0.5]
	#positive2 = [i for i in Y_test if i > 0]

	print (pred.shape, np.array(Y_test).shape)

	print (LogReg.score(X_test, Y_test))
	print (metrics.accuracy_score(Y_test, pred))	

	summer = 0
	for i in range(0, len(pred)):
		if Y_test[i] == 1 and pred[i] == Y_test[i]:
			print (pred[i], Y_test[i])
			summer += 1

	print (summer/float(len(pred)))

if (sys.argv[1] == "c") or (sys.argv[1] == 'All'):

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
	Y_train = to_categorical(Y_train)


	print(X_train)

	model = Sequential()
	model.add(Dense(64, input_shape = (X_train.shape[1], )))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	

	sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
	model.compile(loss='categorical_crossentropy', 
	              optimizer=sgd, 
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, 
	          epochs=10, # Keras 2: former nb_epoch has been renamed to epochs
	          batch_size=32, 
	          verbose=1) 

	Y_train_pred = model.predict_classes(X_train, verbose=0)
	print('First 3 predictions: ', Y_train_pred[:3])

	train_acc = np.sum(Y_train == Y_train_pred, axis=0) / X_train.shape[0]
	print('Training accuracy: %.2f%%' % (train_acc * 100))

	Y_test_pred = model.predict_classes(X_test, verbose=0)
	test_acc = np.sum(Y_test == Y_test_pred, axis=0) / X_test.shape[0]
	print('Test accuracy: %.2f%%' % (test_acc * 100))

	if sys.argv[1] == 'All':
		print(max(pred), max(Y_train_pred))
		for i in range(0, len(pred)):
			if pred[i] != Y_train_pred[i]:
				print (pred[i], Y_train_pred[i])
else:
	print ('Write in task you wish to perform')