import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.python.framework import ops
# import math

def add_layer(inputs, in_size, out_size, activation_function = None):
	# add one more layer and return the output of this layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# biases = tf.Variable(tf.random_normal([1, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs, Weights, biases
   
# ops.reset_default_graph()		# start by clearing and resetting the computational graph
sess = tf.Session()

xmin = -5
xmax = 5

data_test1 = np.loadtxt(r'C:\backup_d\exp\PBL_exp\logit\input\test_all.csv', delimiter = ",", skiprows = 0) 
y_test1 = data_test1[:, 0]
x_test1 = data_test1[:, 1]
x_test1 = (x_test1 - xmin) / (xmax - xmin)
y_test1 = y_test1.reshape(y_test1.shape[0], 1)
x_test1 = x_test1.reshape(x_test1.shape[0], 1)

lr_r = 0.05
lamda = 0.0			# regularization parameter
num_iter = 2500
eval_every = 50
tra_rate = 0.75

# Model input and output
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
layer1, Weights_1, biases_1 = add_layer(xs, 1, 16, activation_function = tf.nn.relu)
layer2, Weights_2, biases_2 = add_layer(layer1, 16, 16, activation_function = tf.nn.relu)		# tf.tanh
prediction, Weights_3, biases_3 = add_layer(layer2, 16, 1, activation_function = tf.sigmoid)

c = tf.Variable(tf.random_uniform((1, 1), minval = 0.2, maxval = 0.5, dtype=tf.float32))	# c=[0,1); c2 = (1 - c) / c 
prediction2 = prediction / (prediction + (1 - c) / c)		# PBL
# prediction2 = prediction * c								# PUL

l2_reg = lamda * (tf.nn.l2_loss(Weights_1) + tf.nn.l2_loss(Weights_2) + tf.nn.l2_loss(Weights_3))
cross_ent = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction2) + (1 - ys) * tf.log(1 - prediction2), reduction_indices = [1]))	# cross entropy
loss = tf.add(cross_ent, l2_reg)
loss_check = cross_ent
train_step = tf.train.GradientDescentOptimizer(lr_r).minimize(loss)		# optimizer

par = np.zeros([10, 4])
for k in range(10):
# for k in range(1):
	input1 = r'C:\backup_d\exp\PBL_exp\logit\input\train_200_' + str(k + 1) + '_PU.csv' 
	data_train = np.loadtxt(input1, delimiter = ",", skiprows = 0)

	input2 = r'C:\backup_d\exp\PBL_exp\logit\input\test_200_' + str(k + 1) + '_PU.csv' 
	data_test2 = np.loadtxt(input2, delimiter = ",", skiprows = 0)
	# y_test2 = data_test2[:, 0]
	x_test2 = data_test2[:, 1]
	x_test2 = (x_test2 - xmin) / (xmax - xmin)
	# y_test2 = y_test2.reshape(y_test2.shape[0], 1)
	x_test2 = x_test2.reshape(x_test2.shape[0], 1)
	
	y_data = data_train[:, 0]
	x_data = data_train[:, 1]
	x_data = (x_data - xmin) / (xmax - xmin)
	y_data = y_data.reshape(y_data.shape[0], 1)
	x_data = x_data.reshape(x_data.shape[0], 1)

	train_ind = np.random.choice(len(x_data), round(len(x_data) * tra_rate), replace = False)
	val_ind = np.array(list(set(range(len(x_data))) - set(train_ind)))	
	x_train = x_data[train_ind]
	x_val = x_data[val_ind]
	y_train = y_data[train_ind]
	y_val = y_data[val_ind]
	train_dict = {xs: x_train, ys: y_train}	
	val_dict = {xs: x_val, ys: y_val}	
	
	# training loop
	init = tf.global_variables_initializer()
	sess.run(init) 					# reset values to wrong

	train_losses = []
	valid_losses = []
	for i in range(num_iter):
		sess.run(train_step, feed_dict = train_dict)
		if (i + 1) % eval_every == 0:		
			temp_train_loss = sess.run(loss_check, feed_dict = train_dict)
			temp_val_loss = sess.run(loss_check, feed_dict = val_dict)
			train_losses.append(temp_train_loss)
			valid_losses.append(temp_val_loss)

	# Plot loss over time
	eval_indices = range(0, num_iter, eval_every)
	plt.plot(eval_indices, train_losses, 'k-', label = 'Train loss')
	plt.plot(eval_indices, valid_losses, 'r--', label = 'Validation loss')
	plt.title('Cross entropy loss per iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	# plt.legend(loc = 'top right')
	plt.legend(loc = 'best')
	plt.show()
	
	ypre1 = sess.run(prediction, {xs: x_test1})		
	ypre2 = sess.run(prediction, {xs: x_test2})

	plt.plot(x_test1, y_test1, 'k-')
	plt.plot(x_test1, ypre1, 'r--')
	plt.show()		# hold on
	
	par[k, 0] = k
	# par[k, 1] = sess.run(biases)
	# par[k, 2] = sess.run(Weights)
	par[k, 3] = sess.run(c)
	# print(sess.run(biases))
	# print(sess.run(Weights))
	print(sess.run(c))	
	
	output1 = r'C:\backup_d\exp\PBL_exp\logit\output\pbl_pre_200_' + str(k + 1) + '_all.csv'
	output2 = r'C:\backup_d\exp\PBL_exp\logit\output\pbl_pre_200_' + str(k + 1) + '_PU.csv'
	np.savetxt(output1, ypre1, delimiter=',', fmt='%f')
	np.savetxt(output2, ypre2, delimiter=',', fmt='%f')
	
output3 = r'C:\backup_d\exp\PBL_exp\logit\output\par.csv'
np.savetxt(output3, par, delimiter=',', fmt='%f')

# reset