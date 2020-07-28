import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('cost_House.csv').values
X = np.array([data[:, 0]]).T
Y = np.array([data[:, 1]]).T
N = X.shape[0]
W = np.array([0.,1.]).reshape(-1, 1)
one = np.ones((N, 1))
Xbar = np.concatenate((one, X), axis = 1)

'''Nếu cập nhật tham số w theo kỹ thuật Batch GD thì cần khá nhiều
thời gian để hoàn thành duyệt tất cả các diểm dữ liệu và cập nhật
lại giá trị w. Do đó ta ứng dụng kỹ thuật stochastic GD, mỗi epoch 
là một lần duyệt một điểm dữ liệu và ta có tối thiểu là N lần
duyệt và cập nhật.Câu hỏi đặt ra làm sao lấy được điểm dữ liệu nào
và điểm đấy có nhiễu không? Ta sử dụng module trong thư viện numpy
để lấy ngẫu nhiên và xáo trộn lại mảng ban đầu mảng này là mảng
chứa số thứ tự N điểm dữ liệu'''

def singlePointGra(W, i, id): #i là vị trí của điểm dữ liệu muốn tính trong mảng sinh số sắp xếp ngẫu nhiên
#id là mảng được sinh ra có nhiệm vụ đảm bảo shuffle data 
	xi = Xbar[id[i], :]
	yi = Y[id[i],:]
	grad = (xi*(np.dot(xi, W)-yi)).reshape(2, 1) #chú ý tích của hiệu ma trận trên là một scalar nên ta không
	#thực hiện nhân ma trận chuyển vị mà sử dụng reshape
	return grad
def StochasticGradientdDescent(W, learning_rate):
	w = [W]	
	count = 0
	interation = 50
	for it in range(1,10):
		id = np.random.permutation(N) #sinh mảng ngẫu nhiên 
		for i in range(1, N):
			count+=1
			w_new = w[-1] - learning_rate*singlePointGra(w[-1], i, id)
			if (count % interation ==0): #interation là số lần duyệt các điểm dữ liệu	
				if (np.linalg.norm(singlePointGra(w_new, i, id))/len(X) < 1e-3):
					return w_new
		w.append(w_new)		
	return w[-1]
print(StochasticGradientdDescent(W, 0.0003))
'''Chú ý phân biệt các thuật ngữ:
1. Epoch- mỗi lần cập nhật tham số và duyệt hết các sample trong tranning set
2. Sample- điểm dữ liệu
3. Batch phân chia thành các sets parts để đẩy nhanh tiến trình hội tụ
 3.1 Batch size số lượng điểm dữ liệu tham gia 1 lần duyệt và cập nhật tham số phụ thuộc vào đây để chia ra
 ba loại: Batch, Stochastic, Mini Batch
 3.2 Number of Batch/Ineration số lượng Batch cần thiết để hoàn thành 1 epoch'''