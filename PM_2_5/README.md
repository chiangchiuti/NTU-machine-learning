
* PM_2_5_linear_regression.py
* PM_2_5_tensorflow.py

These two files do the same work.

PM_2_5_tensorflow uses tensorflow, so it does't need to implement gradient discent by myself

In PM_2_5 linear_regression.py:
	
	X_10 = W_1 * X_1 + W_2 * X_2 + W_3 * X_3 + W_4 * X_4 + ..... + W_9 * X_9
    X_1 to X_9 represent the last 9 hour respectively and it should predict the X_10's value
	what we want is to train the weights: W_1 ... W_9 's values.