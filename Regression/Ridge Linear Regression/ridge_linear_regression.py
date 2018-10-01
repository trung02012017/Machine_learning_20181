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
i = np.identity(x_bar.shape[1])

N_test = data_test.shape[0]
x_test = data_test[:,0:dimension]
x_bar_test_total = np.concatenate((np.ones((N_test,1)), x_test), axis=1)
y_test_total = data_test[:,-1].reshape(N_test,1)


def ridge(x_bar, y_train, x_bar_test, y_test):
    lamda = np.linspace(1e-5, 100, 100000)
    mse_final = 1e10
    for lamda_i in lamda:
        w = np.dot(np.dot(np.linalg.pinv(np.dot(x_bar.T,x_bar) + lamda_i*np.identity(x_bar.shape[1])), x_bar.T), y_train)
        y_predict = np.dot(x_bar_test, w)
        mse = np.mean(np.abs(y_predict - y_test))
        if mse < mse_final:
            mse_final = mse
            w_final = w
            lamda_final = lamda_i
    
    return mse_final, w_final, lamda_final

N_test = 5
x_bar_test = x_bar_test_total[0:N_test,:].reshape((N_test,x_bar.shape[1]))
y_test = y_test_total[0:N_test,:].reshape((N_test,1))

mse_final, w_final, lamda_final = ridge(x_bar, y_train, x_bar_test, y_test)


regr = linear_model.RidgeCV(np.linspace(1e-5, 100, 100000))
regr.fit(x_bar, y_train)

lamda_sklearn = regr.alpha_
w_sklearn = regr.coef_

#plt.plot(y_train, 'r')
#plt.plot(y_predict_train, 'b')

x_bar_predict = x_bar_test_total[N_test:10, :].reshape((5,9))
y_predict = np.dot(x_bar_predict, w_final)
 