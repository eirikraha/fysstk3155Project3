import sys
import numpy as np
import pandas as pd
import copy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
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
ones_index = []
zeros_index = []

for i in enumerate(Y):
	if i[1] == 1:
		ones_index.append(i[0])
	else:
		zeros_index.append(i[0])


#print (len(zeros_index)/(len(ones_index)))

#Brute force, because the smart way didn't work
X_ones = np.zeros((len(ones_index), len(X[0])))
Y_ones = np.ones(len(ones_index))
for i in enumerate(ones_index):
	X_ones[i[0]] = X[i[1]]

#print (X_ones.shape, X.shape, X.shape[0] + X_ones.shape[0]*3, len(Y))

# X = np.r_[X, X_ones, X_ones, X_ones] #, X_ones[:int(np.floor(X_ones.shape[0]*0.52))]]
# Y = np.r_[Y, Y_ones, Y_ones, Y_ones] #, Y_ones[:int(np.floor(Y_ones.shape[0]*0.52))]]

# Y_counter = 0
# for i in enumerate(Y):
# 	if i[1] == 1:
# 		Y_counter += 1

# print ('Ratio of ones to zeros:', Y_counter/float(len(Y)))

# print (X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if (sys.argv[1] == "b") or (sys.argv[1] == 'All'):

	

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
			summer += 1

	print (summer/float(len(pred)), ' If there is bad balane, this would be 0')

if (sys.argv[1] == "c") or (sys.argv[1] == 'All'):

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

	

	sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)
	model.compile(loss='categorical_crossentropy', 
	              optimizer=sgd, 
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, 
	          epochs=50, # Keras 2: former nb_epoch has been renamed to epochs
	          batch_size=128, 
	          verbose=1) 

	Y_train_pred = model.predict_classes(X_train, verbose=0)
	print('Average predictions: ', np.mean(Y_train_pred))

	train_acc = np.sum(Y_train == Y_train_pred, axis=0) / X_train.shape[0]
	print('Training accuracy: %.2f%%' % (train_acc * 100))

	Y_test_pred = model.predict_classes(X_test, verbose=0)
	test_acc = np.sum(Y_test == Y_test_pred, axis=0) / X_test.shape[0]
	print('Test accuracy: %.2f%%' % (test_acc * 100))

else:
	print ('Write in task you wish to perform')