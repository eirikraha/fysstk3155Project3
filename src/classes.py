import numpy as np
import pandas as pd

class ReadXLS():

	def read(self, filename):
		data = pd.read_excel(filename)

		data_dict = {}

		for i in data.columns:
			temp_arr = np.array(data[i])
			data_dict[temp_arr[0]] = list(temp_arr[1:])

		return data_dict

		#print (np.array(data[data.columns[0]])[1:4])


if __name__ == '__main__':
	a = ReadXLS()
	a.read('../data/default of credit card clients.xls')