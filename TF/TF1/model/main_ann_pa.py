import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.python.framework import ops
# import math
import gc

def add_layer(inputs, in_size, out_size, dropout_keep_prob, is_dropout, activation_function = None):
	# add one more layer and return the output of this layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# biases = tf.Variable(tf.random_normal([1, out_size]))			
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)						# preferred
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	
	if is_dropout == 1:
		outputs = tf.nn.dropout(outputs, dropout_keep_prob)
	
	return outputs, Weights, biases

input1 = r'C:\backup_d\exp\PBL_exp\logit\input\test_all.csv'
data_test = np.loadtxt(input1, delimiter = ",", skiprows = 0) 
n_feat = 1
y_test = data_test[:, 0]
x_test = data_test[:, 1:n_feat+1]
y_test = y_test.reshape(y_test.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], n_feat)
del data_test
gc.collect()

lr_r0 = 0.01
lr_decay = 0.9
decay_steps = 100
lamda = 0.0			# regularization parameter
num_iter = 5000         # 5000
eval_every = 50
tra_rate = 0.75
batch_size = 512		
keep_prob_tra = 1.0
keep_prob_val = 1.0
nk = 10

# Model input and output (15-10-5-1; 15-15-10-1)
xs = tf.placeholder(tf.float32, [None, n_feat])
ys = tf.placeholder(tf.float32, [None, 1])
dropout_keep_prob = tf.placeholder(tf.float32)
layer1, Weights_1, biases_1 = add_layer(xs, n_feat, 10, dropout_keep_prob, 1, activation_function = tf.nn.relu)
layer2, Weights_2, biases_2 = add_layer(layer1, 10, 5, dropout_keep_prob, 1, activation_function = tf.nn.relu)		# tf.tanh
# prediction, Weights_3, biases_3 = add_layer(layer2, 5, 1, dropout_keep_prob, 0, activation_function = tf.sigmoid)
prediction, Weights_3, biases_3 = add_layer(layer2, 5, 1, dropout_keep_prob, 0, activation_function = None)

# c = tf.Variable(tf.random_uniform((1, 1), minval = 0.2, maxval = 0.5, dtype=tf.float32))	# c=[0,1); c2 = (1 - c) / c 
# prediction2 = prediction / (prediction + (1 - c) / c)		# PBL
# prediction2 = prediction * c								# PUL

l2_reg = lamda * (tf.nn.l2_loss(Weights_1) + tf.nn.l2_loss(Weights_2) + tf.nn.l2_loss(Weights_3))
# cross_ent = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) + (1 - ys) * tf.log(1 - prediction), reduction_indices = [1]))	# cross entropy
cross_ent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = ys))
loss = tf.add(cross_ent, l2_reg)
loss_check = cross_ent

# optimizer
global_step = tf.Variable(0) 			# count the number of steps taken.
# start_learning_rate = lr_r0
# lr_r = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, lr_decay, staircase = True)
lr_r = lr_r0
# train_step = tf.train.GradientDescentOptimizer(lr_r).minimize(loss, global_step = global_step)		
train_step = tf.train.AdamOptimizer(lr_r).minimize(loss, global_step = global_step)

# nT = 200
sam_size = [200, 1000, 5000]

for s_id in range(2, 3):
	nT = sam_size[s_id]
	
	par = np.zeros([10, n_feat + 5])
	for k in range(0, nk):
	# for k in range(5, 10):
		input2 = r'C:\backup_d\exp\PBL_exp\logit\input\train_' + str(nT) + '_' + str(k + 1) + '_PA.csv'
		data_train = np.loadtxt(input2, delimiter = ",", skiprows = 0)
		
		y_data = data_train[:, 0]
		x_data = data_train[:, 1:n_feat+1]
		y_data = y_data.reshape(y_data.shape[0], 1)
		x_data = x_data.reshape(x_data.shape[0], n_feat)
		
		ypre = np.zeros([x_test.shape[0], 1])
		
		# for n in range(10):
		for n in range(0, nk):
			train_ind = np.random.choice(len(x_data), round(len(x_data) * tra_rate), replace = False)
			val_ind = np.array(list(set(range(len(x_data))) - set(train_ind)))	
			x_train = x_data[train_ind]
			x_val = x_data[val_ind]
			y_train = y_data[train_ind]
			y_val = y_data[val_ind]
			# train_dict = {xs: x_train, ys: y_train}	
			val_dict = {xs: x_val, ys: y_val, dropout_keep_prob: keep_prob_val}	

			# ops.reset_default_graph()		# start by clearing and resetting the computational graph
			sess = tf.Session()
			
			# training loop
			init = tf.global_variables_initializer()
			sess.run(init) 					# reset values to wrong

			train_losses = []
			valid_losses = []
			for i in range(num_iter):					
				ind = np.random.choice(len(x_train), size = batch_size, replace = True)
				batch_x = x_train[ind]
				batch_y = y_train[ind]
				train_dict = {xs: batch_x, ys: batch_y, dropout_keep_prob: keep_prob_tra}
				sess.run(train_step, feed_dict = train_dict)
				if (i + 1) % eval_every == 0:		
					temp_train_loss = sess.run(loss_check, feed_dict = train_dict)
					temp_val_loss = sess.run(loss_check, feed_dict = val_dict)
					train_losses.append(temp_train_loss)
					valid_losses.append(temp_val_loss)
			
			# Plot loss over time
			if n == 0:
				eval_indices = range(0, num_iter, eval_every)
				plt.plot(eval_indices, train_losses, 'k-', label = 'Train loss')
				plt.plot(eval_indices, valid_losses, 'r--', label = 'Validation loss')
				plt.title('Cross entropy loss per iteration')
				plt.xlabel('Iteration')
				plt.ylabel('Loss')
				plt.legend(loc = 'best')		# plt.legend(loc = 'top right')
				plt.show()
				del eval_indices
				gc.collect()
			
			# ypre = sess.run(prediction, {xs: x_test, dropout_keep_prob: keep_prob_val})	
			# ypre = ypre + sess.run(tf.sigmoid(sess.run(prediction, {xs: x_test, dropout_keep_prob: keep_prob_val})))	
			ypre = ypre + sess.run(tf.sigmoid(prediction), {xs: x_test, dropout_keep_prob: keep_prob_val})
			
			# par[k, 1] = par[k, 1] + sess.run(biases)
			
			# w = sess.run(Weights)
			# for i in range(n_feat):
				# par[k, 2 + i] = par[k, 2 + i] + w[i]	
				
			# par[k, 3] = par[k, 3] + c
			par[k, 4] = par[k, 4] + min(train_losses)
			par[k, 5] = par[k, 5] + min(valid_losses)
			
			del sess
			del train_losses, valid_losses, train_ind, val_ind, x_train, x_val, y_train, y_val
			del batch_x, batch_y, ind, train_dict, val_dict
			gc.collect()
			
		ypre = ypre / nk
		
		par[k, 0] = k

		for i in range(1, n_feat + 5):
			par[k, i] = par[k, i] / nk
				
		# print(par[k, 3])	
		print(par[k, 4])
		print(par[k, 5])

		plt.plot(x_test, y_test, 'k-')
		plt.plot(x_test, ypre, 'r--')
		plt.show()		# hold on
		
		output1 = r'C:\backup_d\exp\PBL_exp\logit\output\pre_' + str(nT) + '_' + str(k + 1) + '_ann_pa.csv'
		np.savetxt(output1, ypre, delimiter=',', fmt='%f')
		
		del ypre, data_train, x_data, y_data
		gc.collect()
		
	output2 = r'C:\backup_d\exp\PBL_exp\logit\output\par_' + str(nT) + '_ann_pa.csv'
	np.savetxt(output2, par, delimiter=',', fmt='%f')

del x_test, y_test
gc.collect()
# reset