import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv


class PM2_5_linear_Regression:
	def __init__(self, x_list, y_list, testing_x_list):
		'''
		input:
			x_list, y_list, testing_x_list are list stuct not numpy ndarray
		'''
		self.learning_rate = 0.9
		self.training_epochs = 1500
		self.display_step = 50

		# prepare training and testing data
		self.train_X = np.asarray(x_list, dtype=np.float32)  # convert list to numpy ndarray
		self.train_Y = np.asarray(y_list, dtype=np.float32)
		self.test_X = np.asarray(testing_x_list, dtype=np.float32)
		self.n_sample = self.train_X.shape[0]

		self.train_X, self.test_X = self.feature_normalize(self.train_X, self.test_X)  # normalize training and testing data

		# placeholder in tensorflow
		self.X = tf.placeholder(tf.float32)
		self.Y = tf.placeholder(tf.float32)
		# network parameter
		self.W = tf.Variable(np.random.rand(9).reshape(
			[9, 1]), name="Weight", dtype=tf.float32)
		self.b = tf.Variable(tf.random_normal([1]), name="bias", dtype=tf.float32)

		# tensor operation
		self.predict = tf.add(tf.reduce_sum(tf.matmul(self.X, self.W), 1), self.b)  # y = wx +b
		# cost function
		self.cost = tf.reduce_sum(tf.pow(self.predict - self.Y, 2)) / (2 * self.n_sample)
		self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

		self.init = tf.global_variables_initializer()

	def average_error(self, train_X, train_Y, Weight_array, Bias):
		X = train_X
		Y = train_Y
		m = Y.shape[0]
		predict = X.dot(Weight_array).flatten() + Bias
		error = abs(predict - Y)
		error = error.sum()

		avg_error = error / m
		print('average error:', avg_error)

	def feature_normalize(self, traing_X, test_X):
		global mean, std
		mean = np.mean(traing_X, axis=0)
		std = np.std(traing_X, axis=0)
		traing_X_normalize = (traing_X - mean) / std
		test_X_normalize = (test_X - mean) / std

		# print(mean,std)
		return(traing_X_normalize, test_X_normalize)

	def draw_cost_function(self, cost_history):
		x = range(len(cost_history))
		plt.plot(x, cost_history, label="initial learning rate={0}".format(0.9))
		plt.ylabel("cost value")
		plt.xlabel("interation")
		plt.legend(title='Legend')
		plt.show()

	def output_predict(self, test_x, Weight_array, Bias):
		m = test_x.shape[0]
		predict = test_x.dot(Weight_array).flatten() + Bias
		with open('tensorflow_result.csv', 'w') as out:
			writer = csv.writer(out, delimiter=',')
			writer.writerow(['id', 'value'])
			for index in range(m):
				writer.writerow(['id_' + str(index), predict[index]])

	def training(self):
		'''
		call this function to run training
		'''
		cost_history = []
		with tf.Session() as sess:
			sess.run(self.init)
			# print(sess.run(W),sess.run(b))
			# predict_value=sess.run(predict,feed_dict={X:train_X,Y:train_Y})
			# print(predict_value)
			# cost_value = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
			# print(cost_value)

			for epoch in range(self.training_epochs):
				_, cost_value, weight, bias = sess.run(
					[self.optimizer, self.cost, self.W, self.b],
					feed_dict={self.X: self.train_X, self.Y: self.train_Y})

				cost_history.append(cost_value)
				# Display logs per epoch step
				if(epoch + 1) % self.display_step == 0:
					print("Epoch:{} cost:{}  weight:{} bias:{}".format(
						epoch + 1, cost_value, weight, bias))

			print('Optimization finished')
			result_training_cost, result_weight, result_bias = sess.run([self.cost, self.W, self.b], feed_dict={self.X: self.train_X, self.Y: self.train_Y})

			print("Training cost=", result_training_cost, "W=", result_weight, "b=", result_bias, '\n')

		self.average_error(self.train_X, self.train_Y, result_weight, result_bias)  # compute the average between predict value and real

		self.output_predict(self.test_X, result_weight, result_bias)
		self.draw_cost_function(cost_history)


def read_data(x_list, y_list, testing_list):
	# read the training data
	with open('train.csv', newline='')as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		date_temp = '2014/1/1'
		x_feature = []
		for row in reader:
			if row[0] == date_temp and row[2] == 'PM2.5':  # only use PM2.5 as feature
				x_feature = row[3:12]  # 1~9 hour
				x_feature = list(map(int, x_feature))
				y_list.append(int(row[13]))  # no.10 hour as our target value
				x_list.append(x_feature)  # add to x_array
			else:
				date_temp = row[0]

	with open('test_X.csv', newline='')as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		x_feature = []
		id_index = 'id_0'
		for row in reader:
			if row[0] == id_index and row[1] == 'PM2.5':
				x_feature = row[2:11]
				x_feature = list(map(int, x_feature))
				testing_x_list.append(x_feature)
			else:
				id_index = row[0]

	return x_list, y_list, testing_list


if __name__ == '__main__':
	x_list = []  # training x
	y_list = []  # training y
	testing_x_list = []
	x_list, y_list, testing_x_list = read_data(x_list, y_list, testing_x_list)
	pm25 = PM2_5_linear_Regression(x_list, y_list, testing_x_list)
	pm25.training()
	