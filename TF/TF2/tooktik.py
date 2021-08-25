def Read_file(pth,seed):   ##########################读取数据并给乱序种子
    import numpy as np
    import tensorflow as tf
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

