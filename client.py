import codecs
import pickle
import time
import requests
from CNN_model import DNNModel, Train_Model
from data.extract_data import load_data
import tensorflow as tf
from argparse import ArgumentParser


class Client:
    def __init__(self, server_address, layer, client_index=0, Epoch=10, drop_out=0.2, batch_size=32, learning_rate=0.01, epochs=10):
        # 服务端地址
        self.server_address = server_address
        # 客户端编号
        self.client_index = client_index
        # 联邦学习迭代次数
        self.epoch = Epoch
        # 当前迭代轮次
        self.baseindex = 1
        # 本地模型
        self.model = DNNModel(layer=layer, drop_out=drop_out)
        self.train = Train_Model(self.model)
        # 模型训练参数
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        # 迭代次数
        self.epochs = epochs
        # 本地训练数据地址
        self.path = f'./data/federated_data_{client_index}.d'

    # 获取客户端训练数据
    def get_data(self):
        data = load_data(self.path)
        # 训练数据和测试数据
        self.X_train = tf.constant(data['train_images'], tf.float32)
        self.Y_train = tf.constant(data['train_labels'], tf.float32)
        self.X_test = tf.constant(data['test_images'], tf.float32)
        self.Y_test = tf.constant(data['test_labels'], tf.float32)

    def get_status(self):
        response = requests.get(f'http://{self.server_address}/FL/status')
        data = response.json()
        return data["status"]
    
    def get_model(self):
        response = requests.get(f'http://{self.server_address}/FL/model')
        data = response.json()
        data["global_model"] =  pickle.loads(codecs.decode(data["global_model"].encode(), "base64"))
        data["accuracy"]["loss"] = pickle.loads(codecs.decode(data["accuracy"]["loss"].encode(), "base64"))
        data["accuracy"]["metric"] = pickle.loads(codecs.decode(data["accuracy"]["metric"].encode(), "base64"))
        data["accuracy"]["test_accuracy"] = pickle.loads(codecs.decode(data["accuracy"]["test_accuracy"].encode(), "base64"))
        return data
    
    def send_update(self, computing_time):
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
        args = {
            "client": self.client_index,
            "baseindex": self.baseindex,
            "update_grads": codecs.encode(pickle.dumps(self.model.grads), "base64").decode(),
            'datasize': self.X_train.shape[0],
            'computing_time': computing_time
        }
        response = requests.post(f'http://{self.server_address}/FL/grads', data=args)
        data = response.json()
        return data
        # print(f'Client-{self.client_index}===Baseindex-{self.baseindex}==={data["message"]}')

    def update_model(self, base):
        self.model.build2(base)
        loss, metric, test_accuracy = self.train.train_model(self.epochs, X_train=self.X_train, Y_train=self.Y_train, X_test=self.X_test,
                                                              Y_test=self.Y_test, batch_size=self.batch_size, learning_rate=self.learning_rate)
        # 当前模型的损失值、训练精度和预测精度
        self.loss = loss
        self.metric = metric
        self.test_accuracy = test_accuracy

    def work_step(self):
        # 从服务端获取全局模型
        data = self.get_model()
        while(data["global_model"] == None or data["baseindex"] != self.baseindex):
            print(f"----------Waiting for getting global_model{self.baseindex-1}----------")
            print(self.baseindex, data["baseindex"])
            time.sleep(10)
            data = self.get_model()
        # 同步客户端的迭代次数
        # if data["baseindex"] > self.baseindex:
        #     self.baseindex = data["baseindex"]
        print(f'----------Client:{self.client_index}----------Baseindex:{self.baseindex}----------start----------')
        print(f'----------Global_model:{self.baseindex-1}----------loss:{data["accuracy"]["loss"]}----------metric:{data["accuracy"]["metric"]}----------test_accuracy:{data["accuracy"]["test_accuracy"]}----------')

        # 客户端训练模型
        print(f'----------Client:{self.client_index}----------Baseindex:{self.baseindex}----------Train local models----------')
        start_time = time.time()
        self.update_model(data['global_model'])
        end_time = time.time()

        # 客户端向服务端发送更新的模型梯度
        cpt_time = end_time - start_time
        send_info = self.send_update(cpt_time)

        if send_info["send_status"] == "successful":
            print(f'----------Client:{self.client_index}----------Baseindex:{self.baseindex}----------{send_info["message"]}----------')
        else:
            print(f'----------Client:{self.client_index}----------Baseindex:{self.baseindex}----------Send grads error.Problem:{send_info["message"]}----------')

        # while(send_info["send_status"] == "failed"):
        #     time.sleep(10)
        #     print(f'Problem:{send_info["message"]}')
        #     print("----------Waiting for sending model----------")
        #     send_info = self.send_update(cpt_time)
        
        # print(f'----------Client:{self.client_index}===Baseindex:{self.baseindex}==={send_info["message"]}----------')

    def work(self):
        # 获取客户端本地数据
        self.get_data()
        for i in range(self.epoch):
            # print(f'----------Client:{self.client_index}===Baseindex:{self.baseindex}===start----------')
            self.work_step()
            self.baseindex = self.baseindex + 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', '--server_address', default='127.0.0.1:5000', help='Address of server')
    parser.add_argument('-c', '--client_index', default=0,type=int, help='Index of client')
    parser.add_argument('-e', '--Epoch', default=10,type=int, help='Number of epoch')
    args = parser.parse_args()
    # 为客户端分配一个网络地址，即矿工——miner
    # (self, server_address, layer, client_index=0, Epoch=10, drop_out=0.2, batch_size=32, learning_rate=0.01, epochs=10)
    client = Client(server_address=args.server_address,layer=[784, 256, 256 ,10],client_index=args.client_index,Epoch=args.Epoch,drop_out=0.2, batch_size=32, learning_rate=0.01, epochs=2)
    client.work()
    
