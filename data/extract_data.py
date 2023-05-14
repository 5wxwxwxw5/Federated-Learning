import tensorflow as tf
import numpy as np
import pickle

# 获取mnist数据集
def get_mnist():
    from tensorflow.core.example.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    dataset = dict()
    dataset["train_images"] = mnist.train.images
    dataset["train_labels"] = mnist.train.labels
    dataset["test_images"] = mnist.test.images
    dataset["test_labels"] = mnist.test.labels
    return dataset

# 以二进制形式保存数据集
def save_data(dataset,name="./data/mnist.d"):
    with open(name,"wb") as f:
        pickle.dump(dataset,f)

# 导入数据集
def load_data(name="./data/mnist.d"):
    with open(name,"rb") as f:
        return pickle.load(f)

# 获取数据集的信息
def get_dataset_details(dataset):
    for k in dataset.keys():
        print(k,dataset[k].shape)

# 将数据集拆分（待改进）
def split_dataset(dataset,split_count):
    datasets = []
    split_data_length = len(dataset["train_images"])//split_count
    for i in range(split_count):
        d = dict()
        d["test_images"] = dataset["test_images"][:]
        d["test_labels"] = dataset["test_labels"][:]
        d["train_images"] = dataset["train_images"][i*split_data_length:(i+1)*split_data_length]
        d["train_labels"] = dataset["train_labels"][i*split_data_length:(i+1)*split_data_length]
        datasets.append(d)
    return datasets


if __name__ == '__main__':
    save_data(get_mnist())
    dataset = load_data()
    get_dataset_details(dataset)
    split_num = 4
    for n,d in enumerate(split_dataset(dataset, split_num)):
        # 最后一个数据子集为服务端的训练数据
        if n == split_num - 1:
            save_data(d,"./data/server.d")
            dk = load_data("./data/server.d")
            get_dataset_details(dk)
            print()
        else:
            save_data(d,"./data/federated_data_"+str(n)+".d")
            dk = load_data("./data/federated_data_"+str(n)+".d")
            get_dataset_details(dk)
            print()
