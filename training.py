from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import finance
import pandas as pd
import numpy as np
import sklearn


class LSTMPrediction:
    training_percent = 0.65

    def __init__(self, data):
        # Scale data to LSTM-friendly 0-1 range
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = scaler.fit_transform(np.array(data).reshape(-1, 1))
        print(f'{self.data.size = }\n{self.data = }')

    def get_train_test_data(self):
        training_size = int(len(self.data) * self.training_percent)
        training_data = self.data[:training_size]
        test_data = self.data[training_size:]
        print(f'{training_data.size = }\n{training_data = }')
        print(f'{test_data.size = }\n{test_data = }')
        return training_data, test_data

    def get_xy_data(self, look_back=1):
        x_data = []
        y_data = []
        for i in range(len(self.data) - look_back - 1):
            x = self.data[i:(i + look_back), 0]
            x_data.append(x)
            y_data.append(self.data[(i + look_back), 0])
        return np.array(x_data), np.array(y_data)

    def reshape_lstm(self, dataset):
        return dataset.reshape(dataset.shape[0], dataset.shape[1], 1)


if __name__ == '__main__':
    print('Getting \'AAPL\' info...')
    company = finance.CompanyStock('AAPL')
    prediction = LSTMPrediction(company.get_close())
    prediction.get_train_test_data()

