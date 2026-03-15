"""
tensor是PyTorch里对多维数组,多维列表的表示.
Tensor可以说是PyTorch里最重要的概念,PyTorch把对数据的存储和操作都封装在Tensor里。
PyTorch里的模型训练的输入输出数据,模型的参数,都是用Tensor来表示的。
Tensor在操作方面和NumPy的ndarray是非常类似的。不同的是Tensor还实现了像GPU计算加速自动求导等PyTorch的核心功能。
"""
import torch
import numpy as np
"""
tensor的创建
"""
t1 = torch.tensor([1,2,3])
# t1 = torch.tensor((1,2,3),dtype=torch.float32)
# print(t1)
t2 = torch.tensor([[1,2,3],
                   [4,5,6]])
# print(t2)
t3 = torch.tensor([[[1,2,3],
                    [3,4,5]],
                    [[5,6,8],
                    [7,8,9]]])
# print(t3)


# 可以通过numpy的ndarray来创建一个tensor
arr = np.array([1,2,3])
t_np = torch.tensor(arr)
# print(t_np)


# bool类型tensor
x = torch.tensor([1,2,3,4,5])
mask = x > 2
# print(mask)
# tensor([False, False,  True,  True,  True])
filtered_x = x[mask]
# print(filtered_x)
# tensor([3, 4, 5])
x[mask] = 0
# print(x)
# tensor([1, 2, 0, 0, 0])


# 可以指定创建tensor的设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
t_gpu = torch.tensor([1,2,3],device="mps")
# print(device)
# mps


# 也可以创建一个指定值/随机值填充的tensor，可以指定形状
shape = (2,3)
# shape1 = (2,3,4)
rand_tensor = torch.rand(shape)
# rand_tensor1 = torch.rand(shape1)
# print(rand_tensor)
# print(rand_tensor1.shape)
# tensor([[0.6991, 0.8872, 0.0367],
#         [0.9351, 0.8367, 0.4005]]) 从[0,1]均匀抽样
randn_tensor = torch.randn(shape)
# print(randn_tensor)
# tensor([[-0.5137,  1.0212,  1.2631],
#         [ 0.2590, -0.3722,  0.2284]]) 标准正态分布抽样
ones_tensor = torch.ones(shape)
# print(ones_tensor)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]]) 值全为1
zeros_tensor = torch.zeros(shape)
# print(zeros_tensor)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]]) 值全为0

"""
tensor的属性
"""
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# tensor = torch.rand(3,4)
# print(f"tensor的形状为{tensor.shape}")
# print(f"tensor中的数据类型为{tensor.dtype}")
# print(f"tensor的设备为{tensor.device}")
# print(f"Using device: {device}")

"""
tensor的形状变换
"""
x = torch.randn(4,4)
# print(x)
x = x.reshape(2,8)
# print(x)
# tensor([[-1.0419,  0.4950,  0.4103, -0.3659],
#         [-0.1713, -0.3125, -1.9098, -2.1756],
#         [-0.7434, -1.7448,  1.2545,  0.6472],
#         [-2.4538,  1.2823, -0.6612,  0.1176]])
# tensor([[-1.0419,  0.4950,  0.4103, -0.3659, -0.1713, -0.3125, -1.9098, -2.1756],
#         [-0.7434, -1.7448,  1.2545,  0.6472, -2.4538,  1.2823, -0.6612,  0.1176]])
x = torch.tensor([[1,2,3],[4,5,6]])
x_reshape = x.reshape(3,2) # 按照元素顺序重新组织顺序
x_transpose = x.permute(1,0) # 交换第一个维度和第0个维度
x_t = x.t() # 对于二维tensor，可以使用tensor.t()方法进行转置操作
# print(x_t)
# print("reshape:",x_reshape)
# print("x_transpose:",x_transpose)
# reshape: tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
# x_transpose: tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
x = torch.tensor([[1,2,3],[4,5,6]])
x_0 = x.unsqueeze(0) # 扩展维度，扩展第0维度
# print(x_0)
x_1 = x.squeeze(dim=0) # 缩减维度
# print(x_1)

"""
数学运算
"""
a = torch.ones((2,3))
b = torch.ones((2,3))
# print(a + b)  # 加法
# print(a - b)  # 减法
# print(a * b)  # 逐元素乘法
# print(a / b)  # 逐元素除法
# print(a @ b.t())  # 矩阵乘法

"""
统计函数
tensor.mean(dim=0) 比方说一个二维张量 dim=0的维度是行 意思是是针对行这个维度进行求均值
tensor.mean(dim=1) 实际上tensor指定的这个统计维度是指要消灭这个维度 即dim=0是对于每列在不同行上求均值
如果希望维度不变 不消灭统计的维度 可以指定参数keepdim=true
"""
t = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
# print(t)
# print(t.mean(dim=0))
# print(t.mean(dim=0).shape)
# print(t.mean(dim=1))
# print(t.mean(dim=1).shape)
# print(t.mean(dim=0,keepdim=True))
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])
# tensor([3., 4.])
# torch.Size([2])
# tensor([1.5000, 3.5000, 5.5000])
# torch.Size([3])
# tensor([[3., 4.]])

"""
索引切片
类似python里的序列数据
"""
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x)
# print(x[0, 1])  # 访问第一行第二个元素
# print(x[:, 1])  # 访问第二列
# print(x[1, :])  # 访问第二行
# print(x[:, :2])  # 访问前两列

"""
广播机制
原则来说tensor的逐元素计算要求张量的形状一致
但实际中通过调整张量的索引方式Strided Memory Access
可以在python内部虚拟扩展出一个形状一致的张量进行运算
广播机制的一般原则：
(1)维度对齐 检查两个张量的形状 维度个数不同 在短的那个前面补1
(2)扩展维度 在维度值为1的维度上 通过虚拟复制让两个tensor维度值相等
(3)按位运算
需要特别注意的是:扩展维度时会对两个tensor的每个维度的维度值进行检查
如果在某个维度上两个tensor的维度值不同
那么必须有一个tensor在这个维度的维度值是1
否则广播就会失败整个计算就失败,
"""
#例1
t1 = torch.randn((3,2))
# print(t1)
t2 = t1 + 1 # 广播机制
# print(t2)
#例2
t1 = torch.ones(3,2)
t2 = torch.ones(2)
t3 = t1 + t2
# t4 = torch.ones(3)
# t5 = t1 + t4 #报错
# print(t1,t2)
# print(t3) 

"""
利用gpu加速计算
"""
# # torch.backends.mps.is_available()
import time

# 确保 GPU 可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵
size = 10000  # 矩阵大小
A_cpu = torch.rand(size, size) # 默认在CPU上创建tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.mps.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

# print(f"CPU time: {cpu_time:.6f} sec")
# if torch.backends.mps.is_available():
#     print(f"GPU time: {gpu_time:.6f} sec")
# else:
#     print("GPU not available, skipping GPU test.")

# (learn-pytorch) lanxiox@lanxioxdeMacBook-Air learn_pytorch % python -u "/Users/lanxiox/learn_pytorch/02 t
# ensor.py"
# Using device: mps
# CPU time: 1.389801 sec
# GPU time: 1.215700 sec

"""
Tensor在不同设备上的计算原则
在 PyTorch 中，将 Tensor 和 Model 移动到 CUDA设备的原则如下:

一. Tensor 放入 CUDA

使用 tensor.to('cuda') 或 tensor.cuda() 可以将一个张量移动到 GPU 上。此时，该张量的数据存储和计算都会在 GPU 上进行。

二. Model 放入 CUDA

使用 model.to('cuda') 或 model.cuda() 可以将一个模型移动到 GPU 上。这实际上是将模型内部的所有可学习参数（即 Parameter 本质上是 Tensor 移动到 GPU 上。

三. 计算设备一致性

Tensor进行计算时 参与计算的所有Tensor必须位于同一设备上。否则nPyTorch 会抛出错误。

四. 计算结果的设备归属

运算结果的Tensor会位于参与计算Tensor所在的设备上。

一般情况下 我们利用GPU训练模型 会把Input Tensor Label Tenosr和Model移动到GPU上 则整个模型的训练期间的计算都会在GPU上。
因为前向传播时 Input Tensor和模型内部参数Tensor进行计算 得到模型Output Tensor也在GPU上。Output Tensor和 LabelTensor 都在GPU上计算得到的Loss 梯度也在GPU上。

一句话总结就是：模型和张量需要显式移动到目标设备上 所有参与同一计算的张量必须位于相同设备，计算结果也会保留在该设备上。
"""