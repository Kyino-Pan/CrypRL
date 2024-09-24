
from save_load import *
def train(env, agent, num_episodes=1000, batch_size=64):
    replay_buffer = []
    results_dir = create_results_dir()
    start_epoch, total_reward = load_checkpoint(results_dir, agent.model, agent.optimizer)
    for episode in range(start_epoch, num_episodes):
        import torch
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)  # 将状态移到设备上
        done = False
        total_reward = 0
        while not done:
            # ε-贪婪策略选择动作
            import random
            import torch

            # ε-贪婪策略选择动作
            if random.random() < 0.05:  # ε
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state = state.to(device)  # 确保 state 在 MPS 设备上
                    direction, quantity, thresholds = agent.model(state)
                    # 确保输出都在 MPS 设备上
                    direction = direction.to(device)
                    quantity = quantity.to(device)
                    thresholds = thresholds.to(device)
                    # direction 是 0-1 的输出，用来决定是买入还是卖出，您可以将其二进制化
                    direction = 1 if direction.item() > 0.5 else 0  # 二进制化：>0.5 表示买入，<=0.5 表示卖出

                    # 将数量级和阈值进行处理，这里假设将它们也转换成二进制的0/1输出
                    quantity = (quantity > 0.5).int()  # 数量级：转换为二进制
                    thresholds = (thresholds > 0.5).int()  # 阈值：转换为二进制

                    # 最终的动作组合：将方向，数量级和阈值组合为一个13位的动作
                    action = torch.cat([torch.tensor([direction], device=device), quantity, thresholds],
                                       dim=0).cpu().numpy()
            next_state, reward, done, opt = env.step(action)
            opTensor = torch.tensor(opt, dtype=torch.int32).to(device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)  # 将next_state移到设备上
            reward = torch.tensor([reward], dtype=torch.float32).to(device)  # 将reward移到设备上
            done = torch.tensor([done], dtype=torch.float32).to(device)  # 将done移到设备上

            replay_buffer.append((state, opTensor, reward, next_state, done))

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                agent.update(
                    torch.stack(states),
                    torch.tensor(actions).to(device),
                    torch.stack(rewards),
                    torch.stack(next_states),
                    torch.stack(dones)
                )

            state = next_state
            total_reward += reward.item()

        print(f'Episode {episode + 1}, Total Reward: {total_reward}')

        # 3. 保存训练完成的模型
        save_model(results_dir, agent.model,epoch=episode)

        # 4. 保存策略图表
        save_plot(results_dir, env, epoch=episode)  # 保存图表

def test(env, agent, num_episodes=10):
    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)  # 将状态移到设备上
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action = agent.model(state).argmax().item()
            next_state, reward, done, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)  # 将next_state移到设备上
            total_reward += reward

        print(f'Test Episode {episode + 1}, Total Reward: {total_reward}')
