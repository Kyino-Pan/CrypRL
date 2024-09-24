from agent import DQN
from config import *
from env import TradingEnv
from read import load_kline_data
from train import train, test


def run():
    # 加载数据
    data = load_kline_data('~/data/BinanceHero/KLineData/kline_data_2024_BTCUSDT.csv')  # 替换为实际的K线数据文件路径

    # 初始化环境
    env = TradingEnv(data)

    # 获取状态空间和动作空间的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 初始化智能体
    agent = DQN(state_dim, action_dim)

    # 训练智能体
    print("开始训练...")
    train(env, agent, num_episodes=train_episodes)
    # 测试智能体
    print("训练完成，开始测试...")
    test(env, agent, num_episodes=test_episodes)


if __name__ == "__main__":
    run()
