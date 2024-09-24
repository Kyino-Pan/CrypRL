import torch
import torch.nn as nn
import torch.optim as optim

from config import device


class DQNAgent(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.model = DQNAgent(state_dim, action_dim).to(device)  # 将模型移到设备上
        self.target_model = DQNAgent(state_dim, action_dim).to(device)  # 同样地，将目标模型移到设备上
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99  # 折扣因子

    def update(self, state, action, reward, next_state, done):
        # 将输入的数据移到设备上
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        reward = reward.squeeze(-1)  # 将 reward 从 (64, 1) 转换为 (64)
        done = done.squeeze(-1)  # 将 done 从 (64, 1) 转换为 (64)
        with torch.no_grad():
            next_q_values = self.target_model(next_state)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * max_next_q_value
        loss = self.loss_fn(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
