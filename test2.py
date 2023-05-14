import codecs
import pickle
from flask import jsonify
import tensorflow as tf

# w = tf.Variable(tf.random.truncated_normal([3,2], stddev=0.1), name='weights' + str(1))
# # y = tf.zeros_like(w)

# # print(w)
# # print(y)
# print(w.shape[0])
# b_temp = tf.Variable(tf.zeros([10], name='biases' + str(1)))
# x = tf.zeros_like(b_temp)
# y = tf.Variable(tf.zeros_like(b_temp))
# print(b_temp)
# print(x)
# print(y)

# x = tf.Variable(tf.random.truncated_normal([2, 3], stddev=0.1), name='weights' + str(1))
# y = tf.Variable(tf.zeros(x.shape, tf.float32))
# print(x)
# print(y)

def js():
    y_true = tf.Variable(tf.constant([[1, 1, 3],[2, 3, 1]], tf.float32))
    y = []
    y.append(y_true)
    y.append(y_true)
    # codecs.encode(pickle.dumps(sorted(update.items())), "base64").decode()
    x = {
        "x": codecs.encode(pickle.dumps(y), "base64").decode()
    }
    return x


z = js()
z = pickle.loads(codecs.decode(z['x'].encode(), "base64"))
print(z)


# y_pred = tf.random.truncated_normal([2, 3], stddev=0.1)

# eps = 1e-7
# y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

# x = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

# print(y_pred)
# print(y_true)
# # print(x)

# print(tf.reduce_mean(x))

# 导入数据集
# import pickle


# def load_data(name="./data/mnist.d"):
#     with open(name,"rb") as f:
#         return pickle.load(f)

# # 获取数据集的信息
# def get_dataset_details(dataset):
#     for k in dataset.keys():
#         print(k,dataset[k].shape)

# data = load_data('./data/mnist.d')

# get_dataset_details(data)