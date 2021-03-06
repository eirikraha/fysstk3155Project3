import sys
import numpy as np
import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
import scikitplot as skplt	
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical
from keras.optimizers import SGD
from keras import regularizers

import functions as fcn

def ColorPlotter(x, y, z, method, xlabel, ylabel, cbartitle, mini=0, maxi=1,
	 fs1 = 20, fs2 = 20, fs3 = 20, show = False):
	#Creates colorplots

	fig, ax = plt.subplots(figsize = (18,9))

	yheaders = ['%1.2e' %i  for i in y]
	xheaders = ['%1.2f' %i for i in x]

	print (z)
	heatmap = ax.pcolor(z, edgecolors = "k", linewidth = 2, vmin = mini, vmax = maxi)
	cbar = plt.colorbar(heatmap, ax = ax)
	cbar.ax.tick_params(labelsize= fs3) 
	cbar.ax.set_title(cbartitle, fontsize = fs2)


	#ax.set_title(method, fontsize = fs1)
	ax.set_xticks(np.arange(z.shape[1]) +.5, minor = False)
	ax.set_yticks(np.arange(z.shape[0]) +.5, minor = False)

	ax.set_xticklabels(xheaders,rotation=90, fontsize = fs3)
	ax.set_yticklabels(yheaders, fontsize = fs3)

	ax.set_xlabel(xlabel, fontsize = fs2)
	ax.set_ylabel(ylabel, fontsize = fs2)

	plt.tight_layout()

	plt.savefig('../figures/%s-%s-%s-%s-%s-mini%1.2f-maxi%1.2f.png' %(method, xlabel, 
				ylabel, yheaders[-1],cbartitle, mini, maxi))
	if show:
		plt.show()
	else:
		plt.close()

def XLSread(filename):
	data = pd.read_excel(filename)

	data_dict = {}

	for i in data.columns:
		temp_arr = np.array(data[i])
		data_dict[temp_arr[0]] = list(temp_arr[1:])

	return data_dict

def gain(X, Y, Y_prob, Y_class):
	sorted_index = np.argsort(Y)
	sorted_prob_index = np.argsort(Y_prob[1, :])

	total_ones = 0
	for i in Y:
		if i == 1:
			total_ones += 1

	x = np.linspace(0, len(Y), len(Y))
	y_known = np.zeros(len(Y))
	y_unknown = np.zeros(len(Y))


	print (Y_class)
	print (Y_class[sorted_prob_index])
	print (min(Y_class), min(Y))

	foo
	
	for i in range(0, len(Y)):
		y_unknown[i] = np.sum((Y_class[sorted_prob_index])[0:i])
		y_known[i] = np.sum((Y[sorted_index])[0:i])

	plt.plot(x, x/float(x[-1]))
	plt.plot(x, y_known)

	plt.show()

def randomforrest_optimizer(rnd_clf, X_train_rf, Y_train,
									X_test, Y_test):

	rnd_clf.fit(X_train_rf, Y_train)

	y_pred_rf = rnd_clf.predict(X_test_rf)
	Y_train_pred = rnd_clf.predict(X_train)

	train_acc = np.sum(Y_train == Y_train_pred, axis=0) / X_train.shape[0]
	# print('Training accuracy: %.2f%%' % (train_acc * 100))

	test_acc = np.sum(Y_test == y_pred_rf, axis=0)/X_test_rf.shape[0]
	# print('Test accuracy: %.2f%%' % (test_acc * 100))

	return train_acc, test_acc

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
X_train_rf = X_train
X_test_rf = X_test
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if (sys.argv[1] == "b") or (sys.argv[1] == 'All'):

	if (len(sys.argv) > 2) and (sys.argv[2] == "Optimal"):
		lmbs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
		C = [1./i for i in lmbs]

		train_acc_list = []
		train_area_list = []
		test_acc_list = []
		test_area_list = []

		for i in C:
			LogReg = LogisticRegression(C = i)
			LogReg.fit(X_train, Y_train)
			pred = np.array(LogReg.predict(X_test))
			proba = np.array(LogReg.predict_proba(X_test))
			proba_train = np.array(LogReg.predict_proba(X_train))


			train_acc = LogReg.score(X_train, Y_train)
			train_acc_list.append(train_acc)
			#print('Training accuracy: %.2f%%' % (train_acc * 100))

			test_acc = LogReg.score(X_test, Y_test)
			test_acc_list.append(test_acc)
			#print('Test accuracy: %.2f%%' % (test_acc * 100))

			train_area_list.append(fcn.area_ratio(Y_train, proba_train))

			test_area_list.append(fcn.area_ratio(Y_test, proba))



		title_fontsize="large"
		text_fontsize="medium"

		fig, ax = plt.subplots(1, 2, figsize = (18,9))

		ax[0].set_title(' ', fontsize=title_fontsize)

		ax[0].plot(lmbs, test_acc_list, '--*', label = 'Test')
		ax[0].plot(lmbs, train_acc_list, '--o', label = 'Train')

		# ax.set_xlim([0.0, 1.0])
		ax[0].set_xscale('log')
		ax[0].set_ylim([0.78, 0.82])

		ax[0].set_xlabel('Lambdas',fontsize=text_fontsize)
		ax[0].set_ylabel('Accuracy', fontsize=text_fontsize)
		#ax.tick_params(labelsize=text_fontsize)
		ax[0].grid('on')
		ax[0].legend(loc='lower right', fontsize=text_fontsize)

		#Second subplot
		ax[1].set_title(' ', fontsize=title_fontsize)

		ax[1].plot(lmbs, test_area_list, '--*', label = 'Test')
		ax[1].plot(lmbs, train_area_list, '--o', label = 'Train')

		# ax.set_xlim([0.0, 1.0])
		ax[1].set_xscale('log')
		ax[1].set_ylim([0.38, 0.48])

		ax[1].set_xlabel('Lambdas', fontsize=text_fontsize)
		ax[1].set_ylabel('Area Ratio', fontsize=text_fontsize)
		#ax.tick_params(labelsize=text_fontsize)
		ax[1].grid('on')
		ax[1].legend(loc='lower right', fontsize=text_fontsize)

		plt.savefig('../figures/LogRegOpti.png')
		plt.show()

	else:
		LogReg = LogisticRegression(C = 1./1e-5)
		LogReg.fit(X_train, Y_train)
		pred = np.array(LogReg.predict(X_test))
		proba = np.array(LogReg.predict_proba(X_test))
		proba_train = np.array(LogReg.predict_proba(X_train))


		train_acc = LogReg.score(X_train, Y_train)
		print('Training accuracy: %.2f%%' % (train_acc * 100))

		test_acc = LogReg.score(X_test, Y_test)
		print('Test accuracy: %.2f%%' % (test_acc * 100))

		#skplt.metrics.plot_cumulative_gain(Y_test, proba)

		fcn.plot_cumulative_gain(Y_test, proba)
		plt.savefig('../figures/gain-LogReg.png')
		#plt.show()

		train_area_ratio = fcn.area_ratio(Y_train, proba_train)
		print('Train area ratio: %.2f' % (train_area_ratio))

		test_area_ratio = fcn.area_ratio(Y_test, proba)
		print('Test area ratio: %.2f' % (test_area_ratio))


elif (sys.argv[1] == "c") or (sys.argv[1] == 'All'):

	Y_train_temp = Y_train
	Y_train = to_categorical(Y_train)

	net_type = "Standard"

	# filename_save = '../benchmarks/mom_NN%s_data.txt' % (net_type)

	# f = open(filename_save, 'w')
	# f.write('### Start of neural net file ### \n\n')
	# f.close()


	epochs_array = [140] #[i*20 for i in range(1, 10)]
	nbs = [256] #[2**i for i in range(6, 12)]
	lrs = [0.1]#[0.1, 0.01, 0.001, 0.0001]
	decays = [1e-8] #[1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
	momentums = [.9] #[.7, .8, .9, 1.0, 1.1, 1.2]
	lmbs = [1e-5] #[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


	x_list = lmbs
	x_title = 'Lambda'
	
	y_list = lrs
	y_title = 'Learning rate'

	z_list = np.zeros((len(y_list), len(x_list)))
	z2_list = np.zeros((len(y_list), len(x_list)))

	# print (z_list)

	i = 0
	j = 0

	for x in x_list:
		momentum = momentums[0]
		lmb = lmbs[0]
		epochs = epochs_array[0]
		nb = nbs[0]
		lr = lrs[0]
		decay = decays[0]
		cbartitle = 'Accuracy'	

		for y in y_list:
			#lr = y

			#print(y, y_list)


			if (len(sys.argv) > 2) and (sys.argv[2] == "Deep"):

				net_type = "Deep"

				model = Sequential()
				model.add(Dense(60, input_shape = (X_train.shape[1], )), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(40), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(10), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(2), kernel_regularizer=regularizers.l2(lmb))
				model.add(Activation('softmax'))

			elif (len(sys.argv) > 2) and (sys.argv[2] == "Wide"):

				net_type = "Wide"

				model = Sequential()
				model.add(Dense(512, input_shape = (X_train.shape[1], )))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(512))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(2))
				model.add(Activation('softmax'))

			else:
				model = Sequential()
				model.add(Dense(64, input_shape = (X_train.shape[1], ), kernel_regularizer=regularizers.l2(lmb)))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(32, kernel_regularizer=regularizers.l2(lmb)))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(32, kernel_regularizer=regularizers.l2(lmb)))
				model.add(Activation('relu'))
				model.add(Dropout(0.2))
				model.add(Dense(2, kernel_regularizer=regularizers.l2(lmb)))
				model.add(Activation('softmax'))



			sgd = SGD(lr=lr, decay=decay, momentum=momentum)
			model.compile(loss='categorical_crossentropy', 
		              optimizer=sgd, 
		              metrics=['accuracy'])

			model.fit(X_train, Y_train, 
			          epochs=epochs, # Keras 2: former nb_epoch has been renamed to epochs
			          batch_size=nb, 
			          verbose=1) 


			f = open(filename, 'a')

			print ('Type: %s' %net_type)
			print ('Epochs: %d' %epochs)
			print ('Batch size: %d' %nb)
			print ('Learning rate: %1.2e' %lr)
			print ('Decay: %1.2e' %decay)
			print ('Momentum: %1.2f' %momentum)
			print ('Lambda: %1.2e' %lmb)
			
			Y_train_pred = model.predict_classes(X_train, verbose=0)
			print('Average predictions: ', np.mean(Y_train_pred))

			train_acc = np.sum(Y_train_temp == Y_train_pred, axis=0) / X_train.shape[0]
			
			print('Training accuracy: %.2f%%' % (train_acc * 100))

			Y_test_pred = model.predict_classes(X_test, verbose=0)
			test_acc = np.sum(Y_test == Y_test_pred, axis=0) / X_test.shape[0]

			print('Test accuracy: %.2f%%' % (test_acc * 100))

			# print (len(y_list) - 1)
			# print (len(x_list) - 1)

			if j > len(x_list) - 1:
				j = 0

			if i > len(y_list) - 1:
				i = 0
				j += 1
			

			# print([i, j])  #To make col x row work!
			# print(z_list.shape)

			z_list[i,j] = train_acc
			z2_list[i,j] = test_acc

			i += 1

			# f.write('Epochs: %d \n' %epochs)
			# f.write('Batch size: %d \n' %nb)
			# f.write('Learning rate: %1.2e \n' %lr)
			# f.write('Decay: %1.2e \n' %decay)
			# f.write('Momentum: %1.2f \n' %momentum)
			# f.write('Lambda: %1.2e\n' %lmb)
			# f.write('Average predictions: %.2f%%  \n' % np.mean(Y_train_pred))
			# f.write('Training accuracy: %.2f%%  \n' % (train_acc * 100))
			# f.write('Test accuracy: %.2f%%  \n' % (test_acc * 100))

			# f.write('\n\n')
			# f.close()

	# Activate here for plotting
	# ColorPlotter(x_list, y_list, z_list, net_type, x_title, y_title, cbartitle, 
	# 	mini = 0.78, maxi = np.max(z_list), show = True)
	# ColorPlotter(x_list, y_list, z2_list, '%s-S2' %net_type , x_title, y_title, cbartitle, 
	# 	mini =0.78, maxi = np.max(z_list), show = True)

	proba = model.predict(X_test)
	proba_train = model.predict(X_train)


	fcn.plot_cumulative_gain(Y_test, proba)
	plt.savefig('../figures/gain-NN.png')

	train_area_ratio = fcn.area_ratio(Y_train_temp, proba_train)
	print('Train area ratio: %.2f' % (train_area_ratio))

	test_area_ratio = fcn.area_ratio(Y_test, proba)
	print('Test area ratio: %.2f' % (test_area_ratio))
	

	# train_area_ratio = fcn.area_ratio(Y_train, proba_train)
	# print('Train area ratio: %.2f%%' % (train_area_ratio))

	# test_area_ratio = fcn.area_ratio(Y_test, proba)
	# print('Test area ratio: %.2f%%' % (test_area_ratio))


elif (sys.argv[1] == "d") or (sys.argv[1] == 'All'):

	if (len(sys.argv) > 2) and (sys.argv[2] == "Optimal"):
		n_estimators_list = [60*i for i in range(1, 11)]
		max_leaf_nodes_list = [5*i for i in range(1, 11)]
		min_samples_split_list = [i for i in range(2, 12)]
		min_samples_leaf_list = [i for i in range(1, 11)]

		test_arr = np.zeros((len(n_estimators_list), len(n_estimators_list)))
		train_arr = np.zeros((len(n_estimators_list), len(n_estimators_list)))
		x = 0
		y = 0

		for i in n_estimators_list:
			for j in max_leaf_nodes_list:
				rnd_clf = RandomForestClassifier(n_estimators = i, max_leaf_nodes = j)
				test, train = randomforrest_optimizer(rnd_clf, X_train_rf, Y_train,
									X_test, Y_test)

				if y > len(n_estimators_list) - 1:			
					y = 0

				if x > len(n_estimators_list) - 1:
					x = 0
					y += 1

				train_arr[x, y] = train
				test_arr[x, y] = test

				print(x, y)
				x += 1

		ColorPlotter(n_estimators_list, max_leaf_nodes_list, test_arr, '-test-', 
			'n estimators', 'Max leaf nodes', 'Accuracy', mini =0.78, maxi = np.max(test_arr), 
			show = False)
		ColorPlotter(n_estimators_list, max_leaf_nodes_list, train_arr, '-train-', 
			'n estimators', 'Max leaf nodes', 'Accuracy', mini =0.78, maxi = np.max(train_arr), 
			show = False)

		x = 0
		y = 0

		for i in min_samples_split_list:
			for j in min_samples_leaf_list:
				rnd_clf = RandomForestClassifier(min_samples_split = i, min_samples_leaf = j)
				test, train = randomforrest_optimizer(rnd_clf, X_train_rf, Y_train,
									X_test, Y_test)

				if y > len(n_estimators_list) - 1:			
					y = 0

				if x > len(n_estimators_list) - 1:
					x = 0
					y += 1

				train_arr[x, y] = train
				test_arr[x, y] = test

				print(x, y)
				x += 1

		ColorPlotter(min_samples_split_list, min_samples_leaf_list, test_arr, '-test-', 
			'Min samples split', 'Min samples leaf', 'Accuracy', mini =0.78, maxi = np.max(test_arr), 
			show = False)

		ColorPlotter(min_samples_split_list, min_samples_leaf_list, train_arr, '-train-', 
			'Min samples split', 'Min samples leaf', 'Accuracy', mini =0.78, maxi = np.max(train_arr), 
			show = False)
		x = 0
		y = 0


		for i in n_estimators_list:
			for j in min_samples_leaf_list:
				rnd_clf = RandomForestClassifier(n_estimators = i, min_samples_leaf = j)
				test, train = randomforrest_optimizer(rnd_clf, X_train_rf, Y_train,
									X_test, Y_test)

				if y > len(n_estimators_list) - 1:			
					y = 0

				if x > len(n_estimators_list) - 1:
					x = 0
					y += 1

				train_arr[x, y] = train
				test_arr[x, y] = test

				print(x, y)
				x += 1

		ColorPlotter(n_estimators_list, min_samples_leaf_list, test_arr, '-test-', 
			'n estimators', 'Min samples leaf', 'Accuracy', mini =0.78, maxi = np.max(test_arr), 
			show = False)

		ColorPlotter(n_estimators_list, min_samples_leaf_list, train_arr, '-train-', 
			'n estimators', 'Min samples leaf', 'Accuracy', mini =0.78, maxi = np.max(train_arr), 
			show = False)
		x = 0
		y = 0

		for i in min_samples_split_list:
			for j in max_leaf_nodes_list:
				rnd_clf = RandomForestClassifier(min_samples_split = i, max_leaf_nodes = j)
				test, train = randomforrest_optimizer(rnd_clf, X_train_rf, Y_train,
									X_test, Y_test)

				if y > len(n_estimators_list) - 1:			
					y = 0

				if x > len(n_estimators_list) - 1:
					x = 0
					y += 1

				train_arr[x, y] = train
				test_arr[x, y] = test

				print(x, y)
				x += 1

		ColorPlotter(min_samples_split_list, max_leaf_nodes_list, test_arr, '-test-', 
			'Min samples split', 'Max leaf nodes', 'Accuracy', mini =0.78, maxi = np.max(test_arr), 
			show = False)

		ColorPlotter(n_estimators_list, max_leaf_nodes_list, train_arr, '-train-', 
			'Min samples split', 'Max leaf nodes', 'Accuracy', mini =0.78, maxi = np.max(train_arr), 
			show = False)
		x = 0
		y = 0

	else:
		# rnd_clf = RandomForestClassifier(n_estimators = 360, max_leaf_nodes = 25, 
		# 	min_samples_leaf = 6, min_samples_split = 9)
		rnd_clf = RandomForestClassifier(n_estimators = 600, max_leaf_nodes = 30)
		rnd_clf.fit(X_train_rf, Y_train)

		y_pred_rf = rnd_clf.predict(X_test_rf)
		Y_train_pred = rnd_clf.predict(X_train)

		proba = rnd_clf.predict_proba(X_test)
		proba_train = rnd_clf.predict_proba(X_train)

		train_acc = np.sum(Y_train == Y_train_pred, axis=0) / X_train.shape[0]
		print('Training accuracy: %.2f%%' % (train_acc * 100))

		test_acc = np.sum(Y_test == y_pred_rf, axis=0)/X_test_rf.shape[0]
		print('Test accuracy: %.2f%%' % (test_acc * 100))

		# Feature importance
		# Make X_names array
		X_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",
				 "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20",
				  "X21", "X22", "X23"]

		for name, score in zip(X_names, rnd_clf.feature_importances_):
			print(name, score)

		fcn.plot_cumulative_gain(Y_test, proba)
		plt.savefig('../figures/gain-RF.png')
		#plt.show()

		train_area_ratio = fcn.area_ratio(Y_train, proba_train)
		print('Train area ratio: %.2f' % (train_area_ratio))

		test_area_ratio = fcn.area_ratio(Y_test, proba)
		print('Test area ratio: %.2f' % (test_area_ratio))

else:
	print ('Write in task you wish to perform')