"""
1、回答dataset类是怎么取到每个样本数据的
(1) 从神经网络流程角度,Dataset类 负责：保存数据, 它定义了 index → sample 的映射规则
                    DataLoader类 负责：生成 index, 按 index 调用 dataset[index], 组装 batch
(2) 从调用流程角度,
DataLoader
   ↓
Sampler 生成 index: 创建sample生成索引序列
   ↓
BatchSampler 生成 index list: 将index索引序列分割成batch_size个batch
   ↓
DataLoader 拿到 index: 调用迭代器,取一个 batch 的 index,调用 dataset[i],拼成 batch再返回return collate(batch)
   ↓                     collate函数把元祖序列转换成tensor[(x1,y1),(x2,y2),(x3,y3)] -> tensor([x1,x2,x3]), tensor([y1,y2,y3])
   ↓
dataset[index]   ← 调用 __getitem__
   ↓
返回 sample
   ↓
collate_fn 拼 batch
(3) dataset类怎么取样本数据的
    使用__getitem__(self, index) __getitem__ 方法, 返回的是一个样本, 而不是一个 batch
(4)总的来说, Dataset 负责保存数据并实现 __getitem__(index)，定义 index → sample 的映射规则；
DataLoader 负责通过 Sampler 生成索引，然后调用 dataset[index] 取出每个 sample 并把多个 sample 组成 batch 返回。
"""

"""
进程和线程
进程：资源分配 为了隔离的执行环境 资源:内存、文件描述符表、环境变量、信号处理表 
线程: cpu调度的基本单位的 资源:寄存器、栈、程序计数器 统一进程的线程共享进程的资源
"""