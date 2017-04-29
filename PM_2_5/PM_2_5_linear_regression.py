import numpy as np
import matplotlib.pyplot as plt
import csv
'''
this file is for NTU ML class' first homework: predict the PM2.5
    X_10 = W_1 * X_1 + W_2 * X_2 + W_3 * X_3 + W_4 * X_4 + ..... + W_9 * X_9
    X_1 to X_9 represent the last 9 hour respectively and it should predict the X_10's value
'''


def adagrad(learning_rate, gradient_history):
    '''
        adagrad will return the suitable learning rate for each weights
        by each weight's gradient history
    '''
    gradient_array = np.array(gradient_history)
    gradient_array = gradient_array**2
    gradient_array = gradient_array.sum(axis=0)
    gradient_array = np.sqrt(gradient_array)
    learning_rate_array = learning_rate / gradient_array
    return learning_rate_array


def average_error(x_list, y_list, Weight_array, Bias):
    '''
        calculate the average error, just using |predict - real Y|
    '''
    X = np.array(x_list)
    Y = np.array(y_list)
    m = len(y_list)
    predict = X.dot(Weight_array).flatten() + Bias
    error = abs(predict - Y)
    error = error.sum()

    avg_error = error / m
    print('average error:', avg_error)


def comput_cost(x_list, y_list, Weight_array, Bias):
    X = np.array(x_list)
    Y = np.array(y_list)
    m = len(y_list)

    predict = X.dot(Weight_array).flatten() + Bias
    sqErrors = (predict - Y)**2
    J = (1.0 / (2 * m)) * sqErrors.sum()
    # print('weight vector:',Weight_array.shape,Weight_array)
    # print('x set vector',xset.shape,xset)
    # print('weight array dot xset',np.dot(Weight_array,xset))
    # print('cost function:',J)
    return J


def gradient_descent(x_list, y_list, Weight_array, Bias, learning_rate, number_iters):
    '''
    the main training fucntion
    write the gradient decent by myself
    '''
    J_history = np.zeros(shape=(number_iters, 1))
    X = np.array(x_list)
    Y = np.array(y_list)
    m = len(y_list)
    XTrans = X.transpose()
    gradient_history = []
    gradient_bias_history = []
    for i in range(number_iters):  # training loop
        predict = X.dot(Weight_array).flatten() + Bias  # get the predict value by current weights and bias
        error = predict - Y  # get the error by minusing the real Y
        gradient = np.dot(XTrans, error) / m  # calculate the gradient, we have m training data
        gradient_history.append(gradient)  # record the gradient history
        learning_rate_array = adagrad(learning_rate, gradient_history)  # using adagrad as optimizing alg

        # update the weight and bias values
        Weight_array = Weight_array - \
            np.multiply(learning_rate_array, gradient)
        # print('Weight_array',Weight_array)

        gradient_bias = error.sum() / m  # bias's gradient
        gradient_bias_history.append(gradient_bias)
        learning_rate_bias = adagrad(learning_rate, gradient_bias_history)
        Bias = Bias - (learning_rate_bias * gradient_bias)  # update the bias value
        J_history[i, 0] = comput_cost(x_list, y_list, Weight_array, Bias)  # cost history
    return Weight_array, Bias, J_history[:, 0]


def test_file(Weight_array, Bias):
    '''
    1.predict the result
    2.save the result to file
    '''
    # test file also need normalize by traing file's mean and variance
    x_list = []
    with open('test_X.csv', newline='')as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x_feature = []
        id_index = 'id_0'
        for row in reader:
            if row[0] == id_index and row[1] == 'PM2.5':
                x_feature = row[2:11]
                x_feature = list(map(int, x_feature))
                x_list.append(x_feature)
            else:
                id_index = row[0]

    X = np.array(x_list)
    m = len(x_list)

    predict = X.dot(Weight_array).flatten() + Bias
    # print('predict array',predict)
    # write to the file
    with open('result.csv', 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'value'])
        for index in range(m):
            writer.writerow(['id_' + str(index), predict[index]])


def draw_cost_function(J_history_array, number_iters, learning_rate):
    '''
    draw different case's cost history in a figure
    '''
    X = range(number_iters)

    plt.ylabel("cost value")
    plt.xlabel("interation")
    plt.title("comparision of  different learngin rate")
    plt.ylim([0, 500])
    labels = []

    for i, Y_element in enumerate(J_history_array):
        labels.append(Y_element)
        rate = round(learning_rate[i], 5)
        plt.plot(X, Y_element, label="initial learning rate ={0}".format(rate))
        # print(Y_element)
    plt.legend(title='Legend')
    plt.show()


def different_case(x_list, y_list, Weight_array, Bias, learning_rate, number_iters):
    # print(learning_rate.size,learning_rate)

    J_history_array = []  # store the cost history for each learning rate case
    for index in range(learning_rate.size):  # iterate each learning rate case
        rate = learning_rate[index]  # current learning rate
        print('learning rate :', rate)
        Weight_array, Bias, J_history = gradient_descent(
            x_list, y_list, Weight_array, Bias, rate, number_iters)
        J_history_array.append(J_history)
        print('result Weight_array ', Weight_array)
        print('result bias', Bias)
        average_error(x_list, y_list, Weight_array, Bias)
        print()

    test_file(Weight_array, Bias)
    draw_cost_function(J_history_array, number_iters, learning_rate)


def feature_scaling(x_list):
    '''
        do the feature scaling among different x (x_1 to x_9)
    '''
    X = np.array(x_list)

    mean_array = np.mean(X, axis=0)
    standard_dev_array = np.std(X, axis=0)
    X_result = np.zeros(shape=(X.shape[0], X.shape[1]))

    for index, mean in enumerate(mean_array):
        X_temp = X[:, index] - mean
        X_temp = X_temp / standard_dev_array[index]
        X_result[:, index] = X_temp
    return X_result.tolist()


if __name__ == '__main__':
    x_list = []
    y_list = []
    Weight_array = np.random.rand(9)
    Bias = np.random.rand(1)
    learning_rate = np.arange(0.01, 1, 0.1)  # learning rate array from 0.01 to 1, step 0.1

    number_iters = 1500
    with open('train.csv', newline='')as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        date_temp = '2014/1/1'
        x_feature = []
        for row in reader:
            if row[0] == date_temp and row[2] == 'PM2.5':
                '''
                    only use PM2.5 column here, that's mean I use past nine hours value to predcit the last hour's value
                '''
                x_feature = row[3:12]  # 1~9 hour
                x_feature = list(map(int, x_feature))
                y_list.append(int(row[13]))  # Number 10 hour
                x_list.append(x_feature)  # add to x_array
            else:
                date_temp = row[0]

    x_list = feature_scaling(x_list)

    print('trian set :', len(x_list))
    print('y_head number:', len(y_list))
    print('original cost:', comput_cost(x_list, y_list, Weight_array, Bias))

    different_case(x_list, y_list, Weight_array, Bias, learning_rate, number_iters)

# total_loss = comput_cost(x_list,y_list,Weight_array,Bias)


