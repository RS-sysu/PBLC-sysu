# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
# import math


# data
x_data = np.linspace(-5,5,100)[:, np.newaxis]
y_data = np.exp(1.5 * x_data + 0.5) / (np.exp(1.5 * x_data + 0.5) + 1);
r = np.random.rand(x_data.shape[0], x_data.shape[1])
label = y_data >= r
# label = np.array(y_data >= r)
plt.plot(x_data, y_data)
plt.scatter(x_data, label)
plt.show()		# hold on

x_test = np.linspace(-5,5,50)[:, np.newaxis]
y_test = np.exp(1.5 * x_test + 0.5) / (np.exp(1.5 * x_test + 0.5) + 1);


# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# glm_model = tf.placeholder(tf.float32)
Weights = tf.Variable(tf.random_normal([1, 1]))
biases = tf.Variable(tf.zeros([1, 1]) + 0.1)
Wx_plus_b = tf.matmul(x, Weights) + biases
glm_model = tf.exp(Wx_plus_b) / (tf.exp(Wx_plus_b) + 1)

# loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(glm_model - y), reduction_indices=[1]))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(1000):
	sess.run(train, feed_dict = {x: x_data, y: label})
	if i % 50 == 0:				
		print(sess.run(loss, feed_dict={x: x_data, y: label}))			# to see the step improvement

		
ypre = sess.run(glm_model, {x: x_test})		
		
	
plt.plot(x_test, y_test, 'r')
plt.plot(x_test, ypre, 'b')
plt.show()		# hold on