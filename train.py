import random
from config import device
from save_load import *


def train(env, agent, num_episodes=1000, batch_size=64):
    replay_buffer = []
    results_dir = create_results_dir()
    start_epoch, total_reward = load_checkpoint(results_dir, agent.model, agent.optimizer)
    for episode in range(start_epoch, num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)  # 将状态移到设备上
        done = False
        total_reward = 0
        while not done:
            # ε-贪婪策略选择动作
            if random.random() < 0.1:  # ε
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = agent.model(state).argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)  # 将next_state移到设备上
            reward = torch.tensor([reward], dtype=torch.float32).to(device)  # 将reward移到设备上
            done = torch.tensor([done], dtype=torch.float32).to(device)  # 将done移到设备上

            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                agent.update(
                    torch.stack(states),
                    torch.tensor(actions, dtype=torch.int64).to(device),
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
