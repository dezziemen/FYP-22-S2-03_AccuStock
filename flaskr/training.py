from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math


class LSTMPrediction:
    training_percent = 0.65

    def __init__(self, data):
        # Scale data to LSTM-friendly 0-1 range
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(np.array(data.iloc[:, [1]]).reshape(-1, 1))

    def get_train_test_data(self):
        training_size = int(len(self.data) * self.training_percent)
        training_data = self.data[:training_size]
        test_data = self.data[training_size:]
        return training_data, test_data

    def get_xy_data(self, dataset, look_back=1):
        x_data = []
        y_data = []
        for i in range(len(dataset) - look_back - 1):
            x = dataset[i:(i + look_back), 0]
            x_data.append(x)
            y_data.append(dataset[(i + look_back), 0])
        return np.array(x_data), np.array(y_data)

    def prepare_model(self, look_back, *, x_train, y_train, x_test, y_test):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=8,           # Training iterations
            # batch_size=64,      # Number of batch per epoch
            verbose=1,
            multiprocessing=True,
        )
        return model

    def plot_prediction(self, train_predict, test_predict):
        # Shift train predictions for plotting
        look_back = 100
        train_predict_plot = np.empty_like(self.data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

        # Shift test predictions for plotting
        test_predict_plot = np.empty_like(self.data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict) + (look_back*2) + 1:len(self.data) - 1, :] = test_predict

        # Plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(self.data))
        plt.plot(train_predict_plot)
        plt.plot(test_predict_plot)

    def reshape(self):
        training_data, test_data = self.get_train_test_data()
        look_back = 100
        x_train, y_train = self.get_xy_data(training_data, look_back)
        x_test, y_test = self.get_xy_data(test_data, look_back)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

        return look_back, x_train, x_test, y_train, y_test, test_data

    def train(self, model, *, x_train, x_test, y_train, y_test, test_data):
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)

        math.sqrt(mean_squared_error(y_train, train_predict))
        math.sqrt(mean_squared_error(y_test, test_predict))

    def predict(self, *, days, model, test_data):
        x_input = test_data[-100:].reshape(1, -1)
        temp_input = list(x_input)[0].tolist()
        lst_output = []
        n_steps = 100
        i = 0

        while i < days:
            if len(temp_input) > 100:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())

            i = i + 1

        day_new = np.arange(1, 101)
        day_prediction = np.arange(101, 101 + days)
        df3 = self.data.tolist()
        df3.extend(lst_output)
        data_inversed = self.scaler.inverse_transform((self.data[-100:]))
        predicted_data_inversed = self.scaler.inverse_transform(lst_output)
        plt.plot(day_new, data_inversed)
        plt.plot(day_prediction, predicted_data_inversed)

        print('Prediction done!')

        return predicted_data_inversed

    def start(self, *, days, fig_path):
        look_back, x_train, x_test, y_train, y_test, test_data = self.reshape()
        model = self.prepare_model(look_back, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        self.train(model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_data=test_data)
        predicted_data = self.predict(days=days, model=model, test_data=test_data)
        plt.savefig(fig_path)
        plt.clf()

        return predicted_data
