import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from config import *
from tooktik import *
from tensorflow.keras import layers, optimizers, datasets,Sequential

gpus = tf.config.experimental.list_physical_devices('GPU')##自适应显存
tf.config.experimental.set_memory_growth(gpus[0], True)

global c

def read_data(path,batch_size,seed):
    x_train, y_train = Read_file(path,seed)
    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = y_train.astype(int)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)  ########打包成batch
    return train_db

def train_model(train_db,epoch):                   ######################训练模型
    train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
    loss_all = []

    for epoch in range(epoch):
        for step, (x_train, y_train) in enumerate(train_db):  # 训练样本
            with tf.GradientTape() as tape:  # with结构记录梯度信息
                y = fc_net(x_train)
                y1 = y / (y + (1 - c) / c)
                y2 = tf.cast(y_train, tf.float32)
                c2_reg = lamda2 * tf.square(tf.subtract(tf.reduce_max(y), max_pr_y1))  # smooth l1
                losses = tf.reduce_mean(-tf.reduce_sum((y2 * tf.math.log(y1) + (1 - y2) * tf.math.log(1 - y1)), axis=1))
                loss = tf.add(losses, c2_reg)
                loss_all.append(losses)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

        # 每个epoch，打印loss信息
        los = tf.reduce_mean(loss_all)
        print("Epoch {}, loss: {}".format(epoch, los))
        if epoch % 10 == 0:
            train_loss_results.append(los)  # 将4个step的loss求平均记录在此变量中
        loss_all = []  # loss_all归零，为记录下一个epoch的loss做准备

    # 绘制 loss 曲线
    plt.title('Loss Function Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Loss')  # y轴变量名称
    plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
    plt.legend()  # 画出曲线图标
    plt.show()  # 画出图像
    print('*******************************epoch*************************************')

    return c

def test_model(test_db,c):

    c1 = c / (1 - c)
    TN, FP, FN, TP = 0, 0, 0, 0

    for step,(x_test,y_tes) in enumerate(test_db):  ###测试样本
        y = fc_net(x_test)
        if step == 0:
            y_pre = y
            y_test = y_tes
        else:
            y_pre = tf.concat((y_pre, y))
            y_test = tf.concat((y_test, y_tes))

    y_pre = np.array(y_pre)
    y_test = np.array(y_test)
    print('y_test.shape:',y_test)
    y1 = np.zeros((y_pre.shape[0]))
    for i in range(y_pre.shape[0]):
        if y_pre[i] >= 0.5:
            y1[i] = 1
    TN, FN, FN, TP = metrics.confusion_matrix(y_test, y1).ravel()
    accu = metrics.accuracy_score(y_test, y1)
    fcpb = (2 * TP) / (TP + FN + c1 * FP)
    fpb = (2 * TP) / (TP + FN + FP)

    fpr, tpr, thresholds = metrics.roc_curve(y1, y_pre)
    ROC = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    return fcpb, fpb, accu, ROC


if __name__ =='__main__':
    train_path = Read_feature + ''  ##############################写入训练地址
    test_path = Read_feature + ''  ##############################写入测试地址

    train_db = read_data(train_path,512,100)   ##############路径/batch大小/乱序种子
    test_db = read_data(test_path,128,120)

    net = [
        layers.Dense(80, activation='relu'),
        layers.Dense(40, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ]

    lr = 0.00003  # 学习率
    lamda2 = 0.05  # regularization parameter
    max_pr_y1 = 0.99
    c = tf.Variable(tf.random.uniform([1, 1], minval=0.2, maxval=0.5, dtype=tf.float32))

    fc_net = Sequential(net)
    fc_net.build(input_shape=[None, 7])
    optimizer = optimizers.SGD(lr=lr)
    fc_net.summary()
    variables = fc_net.trainable_variables + [c]

    c = train_model(train_db,epoch=4000)
    test_model(test_db,c)
    fcpb, fpb, accu, ROC = test_model(test_db,c)
