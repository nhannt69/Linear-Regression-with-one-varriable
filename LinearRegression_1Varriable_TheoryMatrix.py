import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data  = pd.read_csv('cost_House.csv').values
x = data[:, 0].reshape(-1, 1)
print(x)
X = np.array([data[:, 0]]).T
Y = np.array([data[:, 1]]).T
'''plt.plot(X, Y, 'ro')
plt.title('To predict a house\'s cost base on an area')
plt.xlabel('area')
plt.ylabel('cost')
plt.show()
Xuất phát từ hàm loss ta đạo hàm L(W) ta được: dL(W)/dW = 1/N *(Xbar.T(XbarW- Y))
ta thấy ta không đạo hàm riêng theo từng biến số trong bộ tham số W mà ta trực tiếp khai
thác đạo hàm thông qua ma trận.
*Ta giải phương trình đạo hàm bằng 0 để tìm nghiệm của W, tuy nhiên có hai vấn đề:
1. Một hàm số có thể có nhiều điểm mà tại đó đạo hàm bằng 0 nhưng nó chưa hẳn là local minimum=> chưa tối ưu tham số
2. Bắt buộc ta phải tính được đạo hàm của hàm số nhưng chưa hẳn trường hợp nào cũng có thể tính được đạo hàm mọt cách dễ dàng
*Ta giải: Xbar.T*Xbar*W = Xbar.T*Y => W = Xbar.T*Y/Xbar.T*Xbar. Ta xét hai khả năng có thể xảy ra:
1. Tìm được ma trận nghịch đảo hay tích Xbar.T*Xbar khả nghịch => Có một nghiệm =>W = inv(Xbar.T*Xbar)*(Xbar.T* Y)
2. Ma trận tích Xbar.T*Xbar không khả nghịch => Có vô số nghiệm hoặc không có nghiệm=>W = pinv(Xbar.T*Xbar)*(Xbar.T* Y)'''

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

W = np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, Y))

w0, w1 = W[0][0], W[1][0]
x1 = np.linspace(30, 100, 2)
y1 = x1*w1 + w0
plt.plot(X, Y, "ro")
plt.plot(x1, y1)
plt.show()

xi = float(input("Enter an area to predict your a house's cost: "))
print(xi*w1 + w0)

