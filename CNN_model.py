"""
搭建深度学习模型
"""

import tensorflow as tf
import numpy as np

input_size = 784
class_num = 10


class DNNModel(tf.Module):
    def __init__(self, layer, drop_out=0.2, name=None):
        super(DNNModel, self).__init__(name=name)
        self.layer = layer
        self.drop_out = drop_out
        self.num = len(layer) - 1
        self.num_classes = layer[-1]
        self.weights = []
        self.biases = []
        self.train_variables = []
        self.grads = []


    def build2(self, base):
        self.weights.clear()
        self.biases.clear()
        self.train_variables.clear()
        self.grads.clear()
        for i in range(self.num):
            w_temp = base[2*i]
            b_temp = base[2*i + 1]
            w_grads = tf.Variable(tf.zeros_like(w_temp))
            b_grads = tf.Variable(tf.zeros_like(b_temp))

            self.weights.append(w_temp)
            self.biases.append(b_temp)

            self.train_variables.append(w_temp)
            self.train_variables.append(b_temp)
            self.grads.append(w_grads)
            self.grads.append(b_grads)

    
    def build(self):
        # 随机初始化网络参数
        self.weights.clear()
        self.biases.clear()
        self.train_variables.clear()
        self.grads.clear()
        for i in range(self.num):
            w_temp = tf.Variable(tf.random.truncated_normal([self.layer[i], self.layer[i+1]], stddev=0.1), name='weights' + str(i + 1))
            b_temp = tf.Variable(tf.zeros([self.layer[i+1]], name='biases' + str(i + 1)))
            self.weights.append(w_temp)
            self.biases.append(b_temp)
            self.grads.append(tf.Variable(tf.zeros([self.layer[i], self.layer[i+1]]), name='weights_grads' + str(i + 1)))
            self.grads.append(tf.Variable(tf.zeros([self.layer[i+1]], name='biases_grads' + str(i + 1))))
        for i in range(self.num):
            self.train_variables.append(self.weights[i])
            self.train_variables.append(self.biases[i])


    # 定义网络前向传播过程
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_size], dtype=tf.float32, name='dnn')])
    def __call__(self, X):
        # 将输入数据转化为浮点类型
        X = tf.cast(X, tf.float32)
        for i in range(self.num - 1):
            X = tf.nn.relu(tf.matmul(X, self.weights[i]) + self.biases[i], name=f'layer{i+1}')
        # 输出结果（将计算结果归一化）
        Y = tf.nn.softmax(tf.matmul(X, self.weights[self.num - 1]) + self.biases[self.num - 1], name=f'layer{self.num}')
        return Y


    # 损失函数(二元交叉熵)
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, class_num], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, class_num], dtype=tf.float32)])
    def loss_func(self, y_true, y_pred):
        # 将预测值限制在 1e-7 和 1-1e-7 之间，避免梯度消失和梯度爆炸
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        # 将数据统一为浮点类型
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        loss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        # 返回均值
        return tf.reduce_mean(loss)


    # 精确度
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, class_num], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, class_num], dtype=tf.float32)])
    def metric_func(self, y_true, y_pred):
        preds = tf.argmax(y_pred, axis=1)  # 取值最大的索引，正好对应字符标签
        labels = tf.argmax(y_true, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return accuracy


class Train_Model(object):

    def __init__(self, model):
        self.model = model


    # 构建数据管道迭代器
    def data_iter(self, features, labels, batch_size=8):
        num_examples = len(features)
        # 构建样本索引
        indices = list(range(num_examples))
        # 将样本数据打乱
        np.random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            indexs = indices[i: min(i + batch_size, num_examples)]
            yield tf.gather(features, indexs), tf.gather(labels, indexs)


    @tf.function
    def train_step(self, features, labels, learning_rate=0.001):
        # 正向传播求损失
        with tf.GradientTape() as tape:
            # 调用 __call__() 函数 获取预测值
            predictions = self.model(features)
            # 获取损失函数值
            loss = self.model.loss_func(labels, predictions)

        # 通过反向传播求梯度
        grads = tape.gradient(loss, self.model.train_variables)

        # 梯度优化裁剪
        grads = [tf.clip_by_value(g, -0.5, 0.5) for g in grads if g is not None]

        # 执行梯度下降更新模型参数
        for p, dloss_dp in zip(self.model.train_variables, grads):
            p.assign(p - learning_rate * dloss_dp)

        # 累计梯度
        for p, dloss_dp in zip(self.model.grads, grads):
            p.assign(p + dloss_dp)

        # 计算训练准确度
        metric = self.model.metric_func(labels, predictions)

        return loss, metric


    def train_model(self, epochs, X_train, Y_train, X_test, Y_test, batch_size=8, learning_rate=0.001):
        # 将模型梯度清空为0
        for i in range(len(self.model.grads)):
            self.model.grads[i] = tf.Variable(tf.zeros(self.model.grads[i].shape, tf.float32))
            
        # 共训练 epochs 轮
        for epoch in tf.range(1, epochs + 1):
            # 分批次训练数据
            for features, labels in self.data_iter(X_train, Y_train, batch_size=batch_size):
                loss, metric = self.train_step(features, labels, learning_rate)
            # break
            # 打印训练信息
            if epoch % 1 == 0:
                # 计算模型在测试集上的预测准确度
                Y_test_pred = self.model(X_test)
                test_accuracy = self.model.metric_func(Y_test, Y_test_pred)
                tf.print(f'Epoch [{epoch}/{epochs}], Train loss: {loss}, Train accuracy: {metric}, Test accuracy: {test_accuracy}')
        # print(type(self.model.train_variables))
        # for i in range(self.model.num):
        #     print(self.model.train_variables[i])
        #     print("================")
        #     print(self.model.grads[i])
        #     break
        # for i in range(self.model.num):
        #     print(self.model.grads[i])
        return loss, metric, test_accuracy







