import torch

# 检查是否支持MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
else:
    device = torch.device('cpu')
    print("MPS not available, using CPU")

train_episodes = 200  # 训练500个回合，按需调整
test_episodes = 5  # 测试10个回合

RETRAIN = False
