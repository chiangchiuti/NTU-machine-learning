import numpy as np
import matplotlib.pyplot as plt
import csv
from math import sqrt

file_name = 'spam_train.csv'

np.random.seed(1)
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
	return(traing_data, ans)


class Neural_Network:
	def __init__(self, traing_set, testing_set):
		self.interation = 10000
		self.global_inter = 0
		self.gradient_history_threshold = 50
		self.learning_rate = 0.01
		self.train_X, self.train_Y = traing_set
		self.test_X, self.test_Y = testing_set

		self.total_training_data = self.train_X.shape[0]
		'''
			build a two layer nerual network
		'''
		self.input_dim = self.train_X.shape[1]  # layer 1 input size
		self.layer_1_output = 20  # hidden layer 1 output size
		self.layer_2_output = 1  # output layer

		self.layer = []
		self.layer.append(self._add_layer(self.input_dim, self.layer_1_output, 'hidden_layer_1'))  # hidden layer 1
		self.layer.append(self._add_layer(self.layer_1_output, self.layer_2_output, 'output_layer'))  # outptu layer

	def _add_layer(self, input_dim, output_dim, layer_name=None):
		weights = np.random.rand(input_dim, output_dim) * sqrt(2.0 / (input_dim * output_dim))
		bias = np.random.rand((output_dim)) * sqrt(2.0 / output_dim)
		gradient_history = []
		momentum_weight = np.zeros((input_dim, output_dim))
		momentum_bias = np.zeros((output_dim))
		layer = {
			'layer_name': layer_name,
			'weights': weights,
			'bias': bias,
			'gradient_history': gradient_history,
			'momentum_weight': momentum_weight,
			'momentum_bias': momentum_bias,
			'current_output': 0,
			'current_delta': 0
		}

		return layer

	def _sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x + 1e-30))

	def _derivative(self, x):
		# sigmoid derivative
		return x * (1.0 - x)

	def _adagrad(self, learning_rate, gradient_history):
		gradient_array = np.power(np.array(gradient_history), 2)
		gradient_array = np.sqrt(gradient_array.sum(axis=0))
		return learning_rate / gradient_array

	def _run_neural_network(self, input_x):
		# forward propagation
		pre_layer = input_x
		each_layer_output = {}
		for i, layer in enumerate(self.layer):
			layer_output = self._sigmoid(pre_layer.dot(self.layer[i]['weights']) + self.layer[i]['bias'])
			self.layer[i]['current_output'] = layer_output
			pre_layer = layer_output
			each_layer_output[self.layer[i]['layer_name']] = layer_output
		return each_layer_output

	def _run_backpropogation(self, traing_ans, layer_output):
		each_layer_delta = {}
		# compute back propogation for each layer
		# output layer
		output_layer_derivative = self.layer[-1]['current_output'] - traing_ans
		ouptut_layer_delta = output_layer_derivative * self._derivative(self.layer[-1]['current_output'])  # mutiply derivative sigmoid function

		self.layer[-1]['current_delta'] = ouptut_layer_delta
		each_layer_delta[self.layer[-1]['layer_name']] = ouptut_layer_delta
		next_layer_delta = ouptut_layer_delta
		# coumpute hidden layer
		for i, layer in enumerate(reversed(self.layer[:-1])):  # doesn't contain output layer
			layer_index = len(self.layer) - i - 2
			layer_derivative = next_layer_delta.dot(self.layer[layer_index + 1]['weights'].T)  # next layer's delta dot next layer's weights
			layer_delta = layer_derivative * self._derivative(self.layer[layer_index]['current_output'])  # layer delta  mutiply sigmoid's derivative

			self.layer[layer_index]['current_delta'] = layer_delta
			each_layer_delta[self.layer[layer_index]['layer_name']] = layer_delta
			next_layer_delta = layer_delta
		return each_layer_delta

	def _update_gradient(self, input_x, layer_output, layer_delta):
		# compute gradient
		for i, layer in enumerate(self.layer):
			layer_index = i
			# gradient = current layer's delta dot pre layer's output
			if layer_index is not 0:
				gradient = self.layer[layer_index - 1]['current_output'].T.dot(self.layer[layer_index]['current_delta']) / self.total_training_data
			else:
				gradient = input_x.T.dot(self.layer[layer_index]['current_delta']) / self.total_training_data
			if self.global_inter % self.gradient_history_threshold == 0:
				self.layer[layer_index]['gradient_history'] = []
			self.layer[layer_index]['gradient_history'].append(gradient)
			new_learning_rate = self._adagrad(self.learning_rate, self.layer[layer_index]['gradient_history'])
			# update weight
			self.layer[layer_index]['weights'] = self.layer[layer_index]['weights'] - np.multiply(new_learning_rate, gradient) - self.layer[layer_index]['momentum_weight']
			# print(self.layer[layer_index]['weights'].shape, new_learning_rate.shape, gradient.shape, self.layer[layer_index]['momentum_weight'].shape)
			# update bias
	
			self.layer[layer_index]['bias'] = self.layer[layer_index]['bias']
			- (self.learning_rate / self.total_training_data) * self.layer[layer_index]['current_delta'].sum()
			- self.layer[layer_index]['momentum_bias']
			# update momentum
			self.layer[layer_index]['momentum_weight'] += gradient
			self.layer[layer_index]['momentum_bias'] += self.layer[layer_index]['current_delta'].sum() / self.total_training_data

	def _count_accurancy(self, y_pre, y_ans):
		correct = 0
		for i in range(len(y_pre)):
			if(y_pre[i][0] > 0.5):
				y_pre[i][0] = 1
			else:
				y_pre[i][0] = 0
			if(y_pre[i][0] == y_ans[i]):
				correct += 1

		accu = correct * 100 / len(y_pre)
		print('total:{} correct:{} accurancy:{:.4f}'.format(len(y_pre), correct, accu))
		return accu

	def _verify_data(self, input_x, input_y):
		# evaluation on testing data
		layer_output = self._run_neural_network(input_x)
		output = layer_output['output_layer']
		print('testing set:', end=' ')
		accu = self._count_accurancy(output, input_y)
		return accu

	def _plot_accuracy(self, training_accu, testing_accu):
		plt.subplot(1, 1, 1)
		plt.cla()
		plt.plot(training_accu, 'g-', label='training accurancy')
		plt.plot(testing_accu, 'r-', label='testing accurancy')
		plt.legend()
		plt.draw()
		plt.pause(0.001)

	def training(self):
		training_accu_history = []
		testing_accu_history = []
		plt.ion()
		plt.figure()
		for epoch in range(self.interation):
			each_layer_output = self._run_neural_network(self.train_X)
			each_layer_delta = self._run_backpropogation(self.train_Y, each_layer_output)
			self._update_gradient(self.train_X, each_layer_output, each_layer_delta)
			self.global_inter = epoch
			if epoch % 100 == 0 and epoch is not 0:
				training_accu = self._count_accurancy(each_layer_output['output_layer'], self.train_Y)
				testing_accu = self._verify_data(self.test_X, self.test_Y)
				training_accu_history.append(training_accu)
				testing_accu_history.append(testing_accu)
				print('epoch:{} training_accu:{:.4f} testing_accu:{:.4f}'.format(epoch, training_accu, testing_accu))
				self._plot_accuracy(training_accu_history, testing_accu_history)
		print('training finish!')
		self._verify_data(self.test_X, self.test_Y)
		plt.ioff()
		plt.show()
		print('hidden layer weights:', self.layer[-2]['weights'])
		print('output layer wiehgts:', self.layer[-1]['weights'])
		print('output layer outputs:', self.layer[-2]['current_output'])


if __name__ == '__main__':
	data, ans = read_training_data(file_name)

	# divide data set to training set and testing set
	traing_set = data[0:int(8 * data.shape[0] / 10), :]
	traing_ans = ans[0:int(8 * data.shape[0] / 10), :]
	testing_set = data[int((8 * data.shape[0] / 10) + 1):, :]
	testing_ans = ans[int((8 * data.shape[0] / 10) + 1):, :]

	neural_network = Neural_Network([traing_set, traing_ans], [testing_set, testing_ans])
	neural_network.training()
	
