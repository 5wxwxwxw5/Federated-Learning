import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from CNN_model import DNNModel, Train_Model

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# 正负样本数量
n_positive, n_negative = 2000, 2000

# 生成正样本, 小圆环分布
r_p = 5.0 + tf.random.truncated_normal([n_positive, 1], 0.0, 1.0)
theta_p = tf.random.uniform([n_positive, 1], 0.0, 2 * np.pi)
Xp = tf.concat([r_p * tf.cos(theta_p), r_p * tf.sin(theta_p)], axis=1)
Yp = tf.ones_like(r_p)


# 生成负样本, 大圆环分布
r_n = 8.0 + tf.random.truncated_normal([n_negative, 1], 0.0, 1.0)
theta_n = tf.random.uniform([n_negative, 1], 0.0, 2 * np.pi)
Xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
Yn = tf.zeros_like(r_n)

# 汇总样本
X = tf.concat([Xp, Xn], axis=0)
Y = tf.concat([Yp, Yn], axis=0)
# 多分类编码
lables = Y.numpy().reshape(-1)
Y = tf.one_hot(indices=lables, depth=2, on_value=1.0, off_value=0.0, axis=-1)

# 切分数据集，测试集占 30%
num_examples = len(X)
indices = list(range(num_examples))
np.random.shuffle(indices)  # 样本的读取顺序是随机的
split_size = int(num_examples * 0.7)
indexs = indices[0: split_size]
X_train, Y_train = tf.gather(X, indexs), tf.gather(Y, indexs)
indexs = indices[split_size: num_examples]
X_test, Y_test = tf.gather(X, indexs), tf.gather(Y, indexs)

# 可视化
# plt.figure(figsize=(6, 6))
# plt.scatter(Xp[:, 0].numpy(), Xp[:, 1].numpy(), c="r")
# plt.scatter(Xn[:, 0].numpy(), Xn[:, 1].numpy(), c="g")
# plt.legend(["positive", "negative"])
# plt.show()

# print("wxw")
# print(Xp[0:5, 0])
# print(Xp[0:5, 1])

layer=[2,32,32,2]
model = DNNModel(layer=layer, drop_out=0.2)
# 模型初始化
model.build()
train = Train_Model(model)
loss, metric, test_accuracy = train.train_model(100, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, batch_size=32, learning_rate=0.01)
# tf.print(f'Train loss: {loss}, Train accuracy: {metric}, Test accuracy: {test_accuracy}')