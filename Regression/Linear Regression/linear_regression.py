import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data_frame = pd.read_csv('../1-prostate-training.csv', header=None)
data = data_frame.values

data_frame_test = pd.read_csv('../20154002-test.csv', header=None)
data_test = data_frame_test.values

N = data.shape[0]
dimension = data.shape[1] - 1

x_train = data[:,0:dimension]
x_bar = np.concatenate((np.ones((N,1)), x_train), axis=1)
y_train = data[:,-1].reshape(N,1)

N_test = data_test.shape[0]
x_test = data_test[:,0:dimension]
x_bar_test = np.concatenate((np.ones((N_test,1)), x_test), axis=1)
y_test = data_test[:,-1].reshape(N_test,1)



w = np.dot(np.dot(np.linalg.pinv(np.dot(x_bar.T,x_bar)), x_bar.T), y_train)
y_predict = np.dot(x_bar_test, w)
y_predict_train = np.dot(x_bar, w)

print(w)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(x_bar, y_train)

print(regr.coef_)

plt.plot(y_train, 'r')
plt.plot(y_predict_train, 'b')