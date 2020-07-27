import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('cost_House.csv').values
X = np.array([data[:, 0]]).T
Y = np.array([data[:, 1]]).T
N = X.shape[0]
one = np.ones((N, 1)) 
Xbar = np.concatenate((one, X), axis = 1) #Xbar = np.hstack(one, X)

W = np.array([[0],[1]])
def grad(w):
	return (1/N)*(np.dot(Xbar.T, (np.dot(Xbar, w)-Y)))
def GradientDescent(w, learning_rate, numberOfInteration):
	w =[W]
	for it in range(1, numberOfInteration):
		w_new = w[-1] - learning_rate*grad(w[-1])
		if (np.linalg.norm(grad(w_new)/len(X)) < 1e-3):
			break
		w.append(w_new)
	return (w_new[0][0], w_new[1][0], it)

(w0, w1) = GradientDescent(W, 0.0003, 300000)
x1 = float(input("Enter an area to predict a house's cost: "))
print(w0 + w1*x1)	
#kiểm tra bằng đồ thị
(w0, w1, it) = GradientDescent(W, 0.0001, 100)
(w0_n, w1_n, it_n) = GradientDescent(W, 0.0003, 300000)
print(w0, w1, it, w0_n, w1_n, it_n)
x1, x1_n = np.linspace(30, 100, 5), np.linspace(30, 100, 5)
y1, y1_n = x1*w1 + w0, x1_n*w1_n +w0_n
plt.scatter(X, Y)
plt.plot(x1, y1, label = 'Line with eta = 0.0001')
plt.plot(x1_n, y1_n, label = 'Line with eta = 0.0003')
plt.legend(loc = 'best')
plt.show()