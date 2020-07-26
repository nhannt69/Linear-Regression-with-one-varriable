import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
#import data from .csv file
data = pd.read_csv('cost_House.csv').values
N = data.shape[0] #lay so observation trong tranning set
#X = np.array([data[:,0]]).T#do du lieu cot dien tich observation x va bien doi thanh ma tran cot X
x = data[:,0].reshape(-1, 1) #do du lieu cot dien tich observation x va bien doi thanh mang mot chieu x
y = data[:,1].reshape(-1, 1) #do du lieu cot gia nha label y
#ve do thi theo du lieu
'''plt.plot(x, y, "ro") #bieu dien theo chi dinh 
#plt.scatter(x, y) #bieu dien phan tan bang cac diem cham
plt.xlabel('Area(square meter)')
plt.ylabel("The house's house")
plt.title("Predict the house's cost base on an area")
plt.show()'''

'''Cơ sở lý thuyết: kỹ thuật gradient descent là kỹ thuật tối ưu ứng dụng rộng trong hầu hết các hàm đơn giản đến phức tạp
cũng xuất phát từ ý tưởng tối thiểu hóa giá trị bộ tham số w để hàm mất mát nhận giá trị nhỏ nhất tức là đạo hàm của hàm mất
mát tiệm cận về 0.
*Xét tại một điểm dữ liệu x ta được L = 1/2(y_pre - y)^2= 1/2(w0 + x1w1 - y)(1) chú ý rằng w0 hay intercept hay bias gọi là
hệ số điều chỉnh để model được linh hoạt và tính dự đoán cao hơn
*Ta tính đạo hàm của L(w): do hàm số trên phụ thuộc hai biến số w0, w1 nên ta đạo hàm riêng từng biến
*dL(w)/d(w0) = w0  + x1w1 - y = XbarW - Y (2)
 dL(w)/d(w1) = x1(w0 + x1w1 - y) = X(XbarW-Y) (3) chú ý công thức trên do tích của XW thực chất chính là tích ma trận Xbar và W phân biệt 
 với tích của X với biểu thức là tích của từng phần tử trong X nhân với từng hàng trong (XW-Y) và X chính là ma trận cột dữ liệu ban đầu
 *TỔNG QUÁT: xét trên N observation tức là N dòng dữ liệu, ta sẽ tìm bộ tham số w để tổng của hàm Loss trên tất cả các điểm dữ liệu
 là nhỏ nhất, công thức (1) biến đổi: L = 1/(2N) * sum((y_pre_i - y_i)^2) khi tính đạo hàm cho hàm số ta có thể bỏ qua N
 			  công thức (2) biến đổi dL(w)/dw0 = sum(XbarW - Y)
 			  công thức (3) biến đổi dL(w)/dw1 = multiply(sum(XbarW - Y),X)
*Xuất phát từ bộ tham số w cho trước, ta tiến hành lặp k lần sao cho truy xuất thấy giá trị hàm Loss nhỏ nhất tức đạo hàm tiệm cận về 0
*Áp dụng giả thiết trong cơ sở nền tảng của Gradient(đạo hàm) Descent(Đi ngược): ta tìm các giá trị w0, w1 rồi tính lại đạo hàm
* w0=w0 - learning_rate*dL(w)/dw0=  w0 - learning_rate*	sum(XbarW - Y)
  w1=w1 - learning_rate*dL(w)/dw0 = w1 -(sum(mutiply(XbarW - Y),X))'''

N = x.shape[0]
one = np.ones((N, 1))
Xbar = np.hstack((one, x))
#Xbar = np.concatenate((one, X), axis = 1)  
learning_rate = 0.000001
w = np.array([0., 1.]).reshape(-1, 1) #chú ý khi gán giá trị khởi tạo cho w ta cần để định dạng số thực

for it in range(1, 10000):
	w[0] = w[0] - learning_rate*np.sum((np.dot(Xbar, w) - y))
	w[1] = w[1] - learning_rate*(np.sum(np.multiply((np.dot(Xbar,w) - y), x)))

x0 = np.linspace(30, 100, 5)
y0 = x0*w[1] + w[0]
plt.plot(x, y, "ro")
plt.plot(x0, y0)
plt.show()
x1 = float(input("Enter an area that you want to: "))
print(w[0] + w[1]*x1)

