import numpy as np
import tensorflow as tf
def Read_file(pth,seed):   ##########################读取数据并给乱序种子
    file = np.loadtxt(pth, delimiter=",", skiprows=1)
    y = file[:, 0]
    x = file[:,1:]
    y = y.reshape(y.shape[0], 1)
    np.random.seed(seed) #####打乱顺序
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(x)
    tf.random.set_seed(seed)
    return x, y

def read_data(path,batch_size,seed):
    x_train, y_train = Read_file(path,seed)
    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = tf.cast(y_train,dtype=tf.float32)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)  ########打包成batch
    return train_db