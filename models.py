import torch
import torch.nn as nn

#自定义一个全连接神经网络MLP
class LNnet(nn.Module):
    def __init__(self):
        super(LNnet, self).__init__()
        #定义全连接层
        self.fc1 = nn.Linear(28*28,256) # 输入是28*28的灰度图像，输出是256个神经元
        self.fc2 = nn.Linear(256,128) # 第二层，全连接层，输入256个神经元，输出128个神经元
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10) # 最后一层全连接层，输入64个神经元，输出10个类别

    def forward(self,x):
        x = torch.flatten(x,start_dim = 1) # 图像展平为一维的
        x = torch.relu(self.fc1(x)) # 输到relu，增加非线性能力
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # 输出层不用激活函数 CEL（CrossEncryptedLoss）损失在内部会进行softmax操作进行概率映射
        # x = torch.softmax(x, dim = 1) # 使用softmax函数激活函数1代表对类别维度进行softmax
        return x
