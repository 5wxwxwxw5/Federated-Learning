# 安装依赖包
目前本项目使用的python包与对应的版本如下：
```
tensorflow                   2.12.0
numpy                        1.23.5
Flask                        2.3.2
requests                     2.30.0
```

# 预处理
模型训练前，首先需要生成实验所需的数据，执行 ./data/extract_data.py文件，即可在该文件夹下生成服务器与客户端的训练数据
```sh
python ./data/extract_data.py
```

# 开始训练
- 启动服务器
```sh
python server.py -p 5000 -i 127.0.0.1 -u 3
```
- 启动客户端（本例中一共启动三个客户端程序，编号分别为0、1、2）
```sh
python client.py -s 127.0.0.1:5000 -c 0 -e 5
python client.py -s 127.0.0.1:5000 -c 1 -e 5
python client.py -s 127.0.0.1:5000 -c 2 -e 5
```

# 保存模型
- 模型训练的每一轮迭代过程中，服务器聚合的全局模型都会以二进制的形式写入 ./global_model 文件夹中
- 客户端上传的梯度参数也会以二进制的形式保存在 ./grads 文件夹中，在每一轮迭代开始之前，服务器都会将该文件夹内旧的梯度文件删除
