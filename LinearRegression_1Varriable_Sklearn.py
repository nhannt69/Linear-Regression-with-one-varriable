import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import pandas as pd

data = pd.read_csv('cost_House.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

one = np.ones((x.shape[0], 1))
xbar = np.concatenate((one, x), axis = 1)

regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(xbar, y)
print(regr.coef_)




