import codecs
import glob
import os
import pickle
import threading
from flask import jsonify,request
from flask import Flask
from federated_learner import Federated_learner
from data.extract_data import save_data, load_data
from argparse import ArgumentParser

app = Flask(__name__)

# 服务端参数
server = {
    # 服务器有两种工作状态，分别为 "receiving" 和 "aggregation"
    "status": "receiving",
    "FLmodel": None,
    "update_limit": None,
    "grads_num": 0,
    "current_index": 0,
    "address": None
}


def delete_prev_grads():
    files = glob.glob('grads/*.d')
    for f in files:
        os.remove(f)

def aggregate_model():
    print(f"----------Baseindex:{server['current_index']}----------Start Model Aggregation----------")
    server["status"] = "aggregation"
    server["FLmodel"].compute_global_model()
    server["FLmodel"].train_model()
    server["FLmodel"].save_model()

    server["status"] = "receiving"
    server["grads_num"] = 0
    server["current_index"] = server["current_index"] + 1

    delete_prev_grads()


# 返回服务端的工作状态
@app.route('/FL/status', methods=['GET'])
def get_status():
    response = {
        "status": server["status"]
    }
    return jsonify(response)


# 客户端向服务端传递更新梯度
@app.route('/FL/grads', methods=['POST'])
def post_grads():
    # 当前服务器如果处于模型聚合阶段，则不再接受客户端传来的模型梯度
    if server["status"] == "aggregation":
        response = {
            "message": "current server is conducting model aggregation and does not accept gradient data",
            "grads_num": server["grads_num"],
            "send_status": "failed",
            "error_code": 0
        }
        return jsonify(response)
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
    client = request.form.get("client")
    baseindex = int(request.form.get("baseindex"))
    update_grads = pickle.loads(codecs.decode(request.form.get("update_grads").encode(), "base64"))
    datasize = int(request.form.get("datasize"))
    computing_time = request.form.get("computing_time")

    # 收到与当前迭代轮次不相符的数据，返回错误信息
    if baseindex != server["current_index"]:
        response = {
            "message": "the number of iterations doesn't match",
            "grads_num": server["grads_num"],
            "send_status": "failed",
            "error_code": 1
        }
        return jsonify(response)

    path = './grads'
    file_name = f'{client}_{baseindex}.d'
    files = os.listdir(path)

    # 如果当前迭代轮次已经接收到该客户端上传的梯度参数，返回错误信息
    if file_name in files:
        response = {
            "message": "current gradient already exists, please don't upload repeatedly",
            "grads_num": server["grads_num"],
            "send_status": "successful",
            "error_code": 2
        }
        return jsonify(response)
    else:
        grads_data = {
            "update_grads": update_grads,
            "datasize": datasize,
            "computing_time": computing_time
        }

        save_path = f'{path}/{file_name}'
        with open(save_path, "wb") as f:
            pickle.dump(grads_data, f)
        
        server["grads_num"] = server["grads_num"] + 1

        response = {
            "message": "successfully uploaded gradient",
            "grads_num": server["grads_num"],
            "send_status": "successful",
            "error_code": 4
        }

        """
        改进点——使用线程执行模型聚合任务
        """

        # 如果已经接受到足够的梯度，则服务器进行模型聚合操作
        if server["grads_num"] == server["update_limit"]:
            thread = threading.Thread(target=aggregate_model)    
            thread.start()
            
        return jsonify(response)


# 客户端从服务端获取更新后的全局模型
@app.route('/FL/model', methods=['GET'])
def get_model():
    # 当前服务器还未初始化模型
    if server["current_index"] == 0:
        response = {
            "global_model": None,
            "accuracy": None,
            "baseindex": server["current_index"],
            "message": "global model has not been initialized yet",
            "error_code": 0
        }
        return jsonify(response)

    model_name = f'global_model{server["current_index"] - 1}.d'
    data = load_data(f'./global_model/{model_name}')
    """
        模型数据的内容
        data = {
            "loss": ...,
            "metric": ...,
            "test_accuracy": ...,
            "parameter": ...,
            "baseindex": ...
        }
    """
    response = {
        "global_model": codecs.encode(pickle.dumps(data["parameter"]), "base64").decode(),
        "baseindex": server["current_index"],
        "message": " global model has been updated",
        "accuracy": {
            "loss": codecs.encode(pickle.dumps(data["loss"]), "base64").decode(),
            "metric": codecs.encode(pickle.dumps(data["metric"]), "base64").decode(),
            "test_accuracy": codecs.encode(pickle.dumps(data["test_accuracy"]), "base64").decode()
        },
        "error_code": 1
    }
    return jsonify(response)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='IP address of this server')
    parser.add_argument('-u', '--update_limit', default=1,type=int, help='number of client')
    args = parser.parse_args()
    # 服务器的网络地址
    address = "{host}:{port}".format(host=args.host,port=args.port)
    server["address"] = address
    server["update_limit"] = args.update_limit

    # 模型初始化
    print(f"----------Baseindex:{server['current_index']}----------Server initialization----------")
    server["FLmodel"] = Federated_learner([784, 256, 256 ,10], drop_out=0.2, path='./data/server.d', batch_size=32, learning_rate=0.01, epochs=2)
    server["FLmodel"].get_data()
    server["FLmodel"].train_model()
    server["FLmodel"].save_model()
    server["current_index"] = server["current_index"] + 1

    app.run(host=args.host,port=args.port)