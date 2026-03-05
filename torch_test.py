import torch

print("torch版本",torch.__version__)
print("MPS加速是否可用",torch.backends.mps.is_available())
print("MPS是否已构建:", torch.backends.mps.is_built())
