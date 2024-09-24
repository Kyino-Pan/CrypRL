import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import device


class DQNAgent(nn.Module):
    def __init__(self, input_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # 三个全连接层分别用于不同的输出
        self.quantFc = nn.Linear(128, 6)  # 输出 6 个数量级
        self.blockFc = nn.Linear(128, 6)  # 输出 6 个阈值
        self.dirFc = nn.Linear(128, 1)  # 输出 1 个方向 (买/卖)

    def forward(self, x):
        # 通过前两个全连接层进行特征提取
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 三个分支的输出
        direction = torch.sigmoid(self.dirFc(x))  # 用 sigmoid 将输出限制在 [0, 1] 范围，决定买卖方向
        quantity = torch.sigmoid(self.quantFc(x))  # 用 sigmoid 将数量级输出为 [0, 1]
        thresholds = torch.sigmoid(self.blockFc(x))  # 用 sigmoid 输出阈值范围 [0, 1]
        # 返回 3 个结果
        return direction, quantity, thresholds


class DQN:
    def __init__(self, state_dim):
        self.model = DQNAgent(state_dim).to(device)
        self.target_model = DQNAgent(state_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99  # 折扣因子

    def update(self, state, action, reward, next_state, done):
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.unsqueeze(-1).to(device)  # 13维动作
        reward = reward.to(device)

        # 从模型中获取 direction, quantity, 和 thresholds 输出
        direction, quantity, thresholds = self.model(state)

        # 将模型输出放到 MPS 设备上（确保一致）
        direction = direction.to(device)
        quantity = quantity.to(device)
        thresholds = thresholds.to(device)

        # 将 direction 二值化：>0.5 表示买入（1），<=0.5 表示卖出（0）
        direction_binary = (direction > 0.5).int()  # 方向：转换为 0 或 1

        # 将数量级和阈值进行二值化处理：>0.5 表示 1，<=0.5 表示 0
        quantity_binary = (quantity > 0.5).int()  # 数量级
        thresholds_binary = (thresholds > 0.5).int()  # 阈值

        # 将 direction, quantity, 和 thresholds 组合成 13 维的动作向量
        action_combined = torch.cat([direction_binary, quantity_binary, thresholds_binary], dim=1)

        # action_combined 的形状为 [batch_size, 13]，可以直接用于 gather 操作
        action_combined = action_combined.to(device)

        # 使用 unsqueeze 扩展维度以适应 gather 的需求
        action_combined = action_combined.unsqueeze(-1).to(device)

        # 确保 action_combined 是 int64 类型用于 gather 操作
        action_combined = action_combined.long()
        action_combined = action_combined.reshape(action_combined.shape[0],-1)
        opt = action_to_opt(action_combined)
        opt = torch.Tensor(opt)

        # 选择与实际动作对应的 Q 值
        q_value = q_values.gather(1, opt.long().unsqueeze(-1)).squeeze(-1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_direction, next_quantity, next_thresholds = self.target_model(next_state)
            next_q_values = torch.cat([next_direction, next_quantity, next_thresholds], dim=1)

            max_next_q_value = next_q_values.max(1)[0]  # 选择下一状态中 Q 值最大的动作
            target_q_value = reward + (1 - done) * self.gamma * max_next_q_value

        # 计算损失并优化模型
        loss = self.loss_fn(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def action_to_opt(action_batch):
    """
    输入: action_batch 形状为 (batch_size, 13)，或 (13,)
    输出: 每个 action 生成的 trade_operation，范围为 [-64, 64]
    """
    # 检查 action 是否为 PyTorch 的 Tensor，如果是，则转换为 NumPy 数组
    if isinstance(action_batch, torch.Tensor):
        action_batch = action_batch.cpu().numpy()  # 将 Tensor 转为 NumPy 数组
    # 确保 action_batch 是二维的 (batch_size, 13)
    if len(action_batch.shape) == 1:
        action_batch = action_batch.reshape(1, -1)  # (13,) 转换为 (1, 13)

    # 提取 action_type, quantities 和 thresholds
    action_type = action_batch[:, 0]  # 二进制的 0 或 1，表示卖出(0) 或 买入(1)
    quantities = np.array([2 ** i for i in range(6)]) * action_batch[:, 1:7]  # 6 个值表示不同数量级
    thresholds = np.array([2 ** i for i in range(6)]) * action_batch[:, 7:13]  # 6 个值用于阈值控制

    # 计算 action_quantity 和 action_threshold
    action_quantity = np.sum(quantities, axis=1)  # 每个样本的交易数量 (batch_size,)
    action_threshold = np.sum(thresholds, axis=1)  # 每个样本的阈值 (batch_size,)

    # 初始化 trade_operation，默认是观望 (0)
    trade_operation = np.zeros(action_batch.shape[0], dtype=int)

    # 买入情况: action_type == 1, 并且 action_quantity > 0
    buy_condition = (action_type == 1) & (action_quantity > 0) & (action_quantity <= action_threshold)
    trade_operation[buy_condition] = np.minimum(64, action_quantity[buy_condition].astype(int))  # 将数量限制在 [1, 64]

    # 卖出情况: action_type == 0, 并且 action_quantity > 0
    sell_condition = (action_type == 0) & (action_quantity > 0) & (action_quantity <= action_threshold)
    trade_operation[sell_condition] = -np.minimum(64, action_quantity[sell_condition].astype(int))  # 将数量限制在 [-64, -1]

    return trade_operation
