"""
LNet:定义了LNet类,继承于torch.nn.Module类,表示LNet类是一个神经网络模型
__init__:构造函数,作用是创建类对象的时候自动调用执行
super().__init__:调用父类构造函数
self.fc1 = nn.Linear(28*28,256):给当前对象添加属性,表示当前对象有一个变量是fc1
self.fq1= nn.Linear:创建一个Linear类的对象赋值给fc1(调用构造函数)
nn.Linear(256,128):输入是256,输出是128,数学公式 y = W*x + b,即W的大小是128*256,b的大小是128*1
nn.Linear(in_features, out_features)
nn.Linear 默认对最后一维进行线性操作
torch.flatten(x, start_dim=1): torch.flatten()是调用torch模块里的flatten函数(模块调用】函数调用 
torch.flatten(x, start_dim=1) 把张量拉平 保留batch 比如(32,1,28,28) -> (32,784)
start_dim=1 因为dim0 = batch,所以从dim1开始展开
"""
import torch
import torch.nn as nn

class LanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


















