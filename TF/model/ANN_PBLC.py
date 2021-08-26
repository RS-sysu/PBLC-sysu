import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import layers, optimizers, datasets,Sequential
from tools.config import *
import os

def ANN(train_db,Input_shape,save_path,test_db = None):                   ######################训练模型

    net = [
        # layers.Dense(50, activation='selu'),
        # layers.BatchNormalization(),
        layers.Dense(10, activation='selu'),
        layers.BatchNormalization(),
        layers.Dense(5, activation='selu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ]     #搭建网络层数

    c = tf.Variable(tf.random.uniform([1, 1], minval=0.2, maxval=0.5, dtype=tf.float32))
    fc_net = Sequential(net)
    fc_net.build(input_shape=[None,Input_shape])
    optimizer = optimizers.Adam(lr=Lr)
    fc_net.summary()
    variables = fc_net.trainable_variables + [c]

    train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
    loss_all = []

    for epoch in range(Epoch):
        for step, (x_train, y_train) in enumerate(train_db):  # 训练样本
            with tf.GradientTape() as tape:
                y = fc_net(x_train)   #正负
                y1 = y / (y + (1 - c) / c)
                y2 = tf.cast(y_train, tf.float32)
                c2_reg = lamda2 * tf.square(tf.subtract(tf.reduce_max(y), max_pr_y1))  # smooth l1
                losses = tf.reduce_mean(-tf.reduce_sum((y2 * tf.math.log(y1) + (1 - y2) * tf.math.log(1 - y1)), axis=1))
                loss = tf.add(losses, c2_reg)
                loss_all.append(losses)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

        los = tf.reduce_mean(loss_all)
        print("Epoch {}, loss: {}".format(epoch, los))
        print('*******************************epoch*************************************')
        train_loss_results.append(los)
        loss_all = []  # loss_all归零，为记录下一个epoch的loss做准备

    # 绘制 loss 曲线
    plt.title('Loss Function Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_results, label="$Loss$")
    plt.legend()
    plt.show()

    fc_net.save(os.path.join(save_path, 'model.h5'))  #保存模型
    np.savetxt(os.path.join(save_path, 'train_loss.csv'), train_loss_results, fmt='%.5f') #保存Loss

    if test_db == None:
        print('C:{}'.format(c))
        print('*******************finish the trained network*****************************')
    else:
        for step, (x_test, y_test) in enumerate(test_db):  ###测试样本
            y = fc_net(x_test)
            if step == 0:
                test_pre = np.array(y)
                test_truth = np.array(y_test)
            else:
                test_pre = np.vstack((test_pre,np.array(y)))
                test_truth = np.vstack((test_truth,np.array(y_test)))

        test_pre = np.squeeze(test_pre,axis=1)
        test_truth = np.squeeze(test_truth,axis=1)

        test_pre = test_pre[np.argsort(test_truth)]
        test_truth = test_truth[np.argsort(test_truth)]
        RMS = np.mean(np.square((test_pre-test_truth)))
        print('the test RMS :',RMS)
        plt.title('the test Curve')
        plt.plot(test_pre, label="$prediction$")
        plt.plot(test_truth,label="$Truth$")
        plt.legend()
        plt.show()
        np.savetxt(os.path.join(save_path, 'test_prediction.csv'), test_pre, fmt='%.1f')
        np.savetxt(os.path.join(save_path, 'test_truth.csv'), test_truth, fmt='%.1f')
        print('C:{}'.format(c))
        print('*******************finish the trained network*****************************')








