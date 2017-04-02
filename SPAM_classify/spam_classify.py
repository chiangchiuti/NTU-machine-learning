import numpy as np
import matplotlib.pyplot as plt
import csv


class spam_classify:
	def __init__(self, data, ans):
		self.training_set = data[0:int(9 * data.shape[0] / 10), :]
		self.traing_ans = ans[0:int(9 * data.shape[0] / 10), :]
		self.testing_set = data[int((9 * data.shape[0] / 10) + 1):, :]
		self.testing_ans = ans[int((9 * data.shape[0] / 10) + 1):, :]
		print('traing data:', self.traing_ans.shape[0], 'valid ata', self.testing_ans.shape)
		weight_num = data.shape[1]
		self.Weight_array = np.random.rand(weight_num, 1)
		self.bias = np.random.rand(1)
		self.learning_rate = 0.5
		self.learning_rate_bias = self.learning_rate
		self.batch_size = 500
		self.interation = 1500

	def training(self):
		cost_history = np.zeros((self.interation, 1))
		accu_rate_history = np.zeros((self.interation, 1))
		gradient_history = []
		gradient_bias_history = []

		learning_rate_array = np.full((57, 1), self.learning_rate)  # fill ndarray size (57, 1) as learning rate
		m = self.training_set.shape[0]
		self.comput_cost(self.training_set, self.traing_ans)
		for i in range(self.interation):
			for k in range(int(m / self.batch_size) + 1):  # get batch
				if k == int(m / self.batch_size) + 1:
					X_temp = self.training_set[k * self.batch_size:, :]
					Y_temp = self.traing_ans[k * self.batch_size:, :]
				else:
					X_temp = self.training_set[k * self.batch_size:(k + 1) * self.batch_size, :]
					Y_temp = self.traing_ans[k * self.batch_size:(k + 1) * self.batch_size, :]

				predict = (1.0 / (1.0 + np.exp(-(X_temp.dot(self.Weight_array).flatten() + self.bias))))[:, np.newaxis]  # sigmoid
				error = predict - Y_temp  # calculate the error
				Weight_gradient, bias_gradient = self.compute_gradient(X_temp, error)  # compute the gradient of curren batch
				gradient_history.append(Weight_gradient)
				gradient_bias_history.append(bias_gradient)

				learning_rate_array = self.adagrad(self.learning_rate, gradient_history)  # update the learning rate by adagrad
				learning_rate_bias = self.adagrad(self.learning_rate, gradient_bias_history)

				self.Weight_array = self.Weight_array - \
					np.multiply(learning_rate_array, Weight_gradient)  # update the weight
				self.bias = self.bias - learning_rate_bias * bias_gradient  # update the bias
			cost_history[i] = self.comput_cost(self.training_set, self.traing_ans)
			predict = (1.0 / (1.0 + np.exp(-(self.training_set.dot(self.Weight_array).flatten() + self.bias))))[:, np.newaxis]  # sigmoid
			accu_rate_history[i] = self.count_correct_rate(self.traing_ans, predict)
			if i % 100 == 0:
				print('epoch:{} cost:{} accurancy:{}'.format(i, cost_history[i], accu_rate_history[i]))
		print("Weight_array:", self.Weight_array)
		print("bias", self.bias)

		self.draw_picture(cost_history, accu_rate_history)  # draw the error rate and cost figure
		self.Write_model()

		return self.Weight_array, self.bias

	def testing(self):
		predict = (1.0 / (1.0 + np.exp(-(self.testing_set.dot(self.Weight_array).flatten() + self.bias))))[:, np.newaxis]  # sigmoid
		accu_rate = self.count_correct_rate(self.testing_ans, predict)
		print('testing accuracy{}'.format(accu_rate))

		return accu_rate

	def Write_model(self):
		'''
		save the final weight and bias
		'''
		with open('result.csv', 'w') as out:
			writer = csv.writer(out, delimiter=',')
			for row in self.Weight_array:
				writer.writerow(row)
			writer.writerow(self.bias)

	def adagrad(self, learning_rate, gradient_history):
		gradient_array = np.array(gradient_history)
		gradient_array = gradient_array**2
		gradient_array = gradient_array.sum(axis=0)
		gradient_array = np.sqrt(gradient_array)
		# print('learning_rate',learning_rate)
		learning_rate_array = learning_rate / gradient_array
		return learning_rate_array
		# print('gradient_sum:',gradient_sum)
		# gradient_sum = gradient_sum.square

	def comput_cost(self, X, Y):
		m = Y.shape[0]  # how many data
		predict = (
			1.0 / (1.0 + np.exp(-(X.dot(self.Weight_array).flatten() + self.bias))))[:, np.newaxis]  # y = ax + b

		# y_pre = np.array([1 if l > 0.5 else 0 for l in predict])[:, np.newaxis]

		cross_entropy = ((-1) / m) * (sum(Y * np.log(
			predict +
			1e-50) + (1 - Y) * np.log(1 - predict + 1e-50)))
		# cost =0
		# for i ,y_pre_ele in enumerate(predict):
		# cost+= np.log(y_pre_ele+1e-50)*Y[i] + (1-Y[i])*np.log(1-y_pre_ele+1e-50)

		# cost = (-cost)/m
		return cross_entropy

	def compute_gradient(self, x, error):
		m = x.shape[0]
		XTrans = x.transpose()
		# print('XTrans:',XTrans.shape,'error',error.shape)
		Weight_gradient = (XTrans.dot(error)) / m
		bias_gradient = (np.sum(error)) / m
		return Weight_gradient, bias_gradient

	def draw_picture(self, cost_history, err_rate_history):
		p1x = range(cost_history.shape[0])
		p1 = plt.subplot(211)
		p1.set_ylabel("cost value")
		p1.set_xlabel("interation")
		p1.plot(p1x, cost_history, label='cost')
		plt.legend()
		p2x = range(err_rate_history.shape[0])
		p2 = plt.subplot(212)
		p2.set_ylabel("training Accurancy")
		p2.set_xlabel('interation')
		p2.plot(p2x, err_rate_history, label='accuracy')
		plt.legend()
		plt.show()

	def count_correct_rate(self, y, predict):
		m = y.shape[0]
		y_pre = np.array([1 if l > 0.5 else 0 for l in predict])[:, np.newaxis]
		error = y_pre - y
		err_num = np.count_nonzero(error)
		err_rate = err_num / m
		# print("correct rate is :", 1 - err_rate)
		return 1 - err_rate


def feature_normalize(traing_X):
	global mean, std
	mean = np.mean(traing_X, axis=0)
	std = np.std(traing_X, axis=0)
	traing_X_normalize = (traing_X - mean) / std

	# print(mean,std)
	return(traing_X_normalize)


def read_training_data(file_name):
	with open(file_name, newline='') as csvfile:
		reader = csv.reader(csvfile)
		data = []
		ans = []
		for row in reader:
			data.append(row[1:58])
			ans.append(row[-1])

	traing_data = np.array(data[:], dtype='float64')
	ans = np.array(ans, dtype='float64')[:, np.newaxis]
	traing_data = feature_normalize(traing_data)
	return traing_data, ans


if __name__ == '__main__':
	file_name = 'spam_train.csv'
	data, ans = read_training_data(file_name)
	spam_cl = spam_classify(data, ans)
	result_weigt, result_bias = spam_cl.training()
	spam_cl.testing()
