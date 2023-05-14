"""
负责服务器端的事务处理逻辑
"""

import pathlib
import pickle
from CNN_model import DNNModel, Train_Model
from data.extract_data import load_data
import tensorflow as tf

class Federated_learner():
    # 服务器初始化
    def __init__(self, layer, drop_out=0.2, path='./data/mnist.d', batch_size=32, learning_rate=0.01, epochs=10):
        # 服务端模型
        self.model = DNNModel(layer=layer, drop_out=drop_out)
        self.model.build()
        self.train = Train_Model(self.model)
        # 服务端训练数据地址
        self.path = path
        # 当前迭代次数
        self.baseindex = 0
        # 模型训练参数
        # 批次大小
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        # 迭代次数
        self.epochs = epochs
    

    # 获取服务器端训练数据
    def get_data(self):
        data = load_data(self.path)
        # 训练数据和测试数据
        self.X_train = tf.constant(data['train_images'], tf.float32)
        self.Y_train = tf.constant(data['train_labels'], tf.float32)
        self.X_test = tf.constant(data['test_images'], tf.float32)
        self.Y_test = tf.constant(data['test_labels'], tf.float32)


    # 训练模型
    def train_model(self):
        loss, metric, test_accuracy = self.train.train_model(self.epochs, X_train=self.X_train, Y_train=self.Y_train, X_test=self.X_test,Y_test=self.Y_test, batch_size=self.batch_size, learning_rate=self.learning_rate)
        # 当前模型的损失值、训练精度和预测精度
        self.loss = loss
        self.metric = metric
        self.test_accuracy = test_accuracy

        tf.print(f'----------Current model: Train loss: {loss}, Train accuracy: {metric}, Test accuracy: {test_accuracy}----------')


    # 保存模型
    def save_model(self):
        model_data = {
            "loss": self.loss,
            "metric": self.metric,
            "test_accuracy": self.test_accuracy,
            "parameter": self.model.train_variables,
            "baseindex": self.baseindex
        }
        path = './global_model'
        model_name = f'global_model{self.baseindex}.d'
        save_path = f'{path}/{model_name}'

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        self.baseindex = self.baseindex + 1


    # 聚合模型
    def compute_global_model(self):
        total_grads = []
        datasize = []
        total_datasize = 0
        # 依次读取客户端传来的梯度参数
        files = pathlib.Path('./grads').glob("*.d")
        for file in files:
            """
            客户端传输的数据格式
            {
                # 客户端id
                'client': ...,
                # 当前版本号
                'baseindex': ...,
                # 客户端在本地训练完之后更新的梯度
                'update_grads': ...,
                # 数据集的规模
                'datasize': ...,
                # 处理时间
                'computing_time': ...
            }
            """
            data = load_data(file)
            total_grads.append(data["update_grads"])
            datasize.append(data['datasize'])
            total_datasize = total_datasize + data['datasize']
        # 聚合客户端的梯度参数
        for i in range(len(total_grads)):
            client_weight = datasize[i] / total_datasize
            for p, dloss_dp in zip(self.model.train_variables, total_grads[i]):
                p.assign(p - client_weight * self.learning_rate * dloss_dp)
            
        # 更新模型的权重与偏置列表
        self.model.weights.clear()
        self.model.biases.clear()
        for i in range(int(len(self.model.train_variables)/2)):
            self.model.weights.append(self.model.train_variables[i])
            self.model.biases.append(self.model.train_variables[i+1])
            