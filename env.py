import gym
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):
    def __init__(self, data, max_actions=500):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_step = 0  # 操作计数器
        self.max_actions = max_actions  # 最大操作次数
        self.initial_balance = 1000000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.data['Close Price'].values[self.current_step]

        # Action space: 0 - hold (观望), 1 - buy (买入), 2 - sell (卖出)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.buy_signals = []
        self.sell_signals = []

    def reset(self):
        self.current_step = 0
        self.action_step = 0  # 重置操作计数器
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.data['Close Price'].values[self.current_step]
        self.buy_signals = []
        self.sell_signals = []
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step][
            ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'normalized_close']].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_price = self.data['Close Price'].values[self.current_step]
        reward = 0

        # 记录买入和卖出的时机
        if action == 1:  # Buy
            if self.balance >= self.current_price:  # 确保有足够资金
                self.shares_held += 1
                self.balance -= self.current_price
                self.buy_signals.append((self.data['Open Time'].values[self.current_step], self.current_price))
                self.action_step += 1  # 计入操作次数
            else:
                reward -= 10  # 没有足够的资金，给负奖励
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.current_price
                self.shares_held -= 1
                reward = self.balance + self.shares_held * self.current_price - self.initial_balance  # 奖励基于余额
                self.sell_signals.append((self.data['Open Time'].values[self.current_step], self.current_price))
                self.action_step += 1  # 计入操作次数
            else:
                reward -= 10  # 没有股票可卖，给负奖励
        elif action == 0:  # Hold (观望)
            reward -= 0.1  # 轻微负奖励，鼓励智能体寻找合适的交易机会

        # 每个时间步长都会增加，不论是否执行操作
        self.current_step += 1

        # 结束条件：超过最大步数或者最大操作次数
        done = (self.current_step >= len(self.data) - 1) or (self.action_step >= self.max_actions)

        # 返回观察值、奖励、是否完成、以及调试信息
        return self._get_observation(), reward, done, {}

    def render(self):
        print(
            f'Step: {self.current_step}, Action Step: {self.action_step}, Price: {self.current_price}, Shares Held: {self.shares_held}, Balance: {self.balance}')

    def plot_trades(self):
        # 将时间转换为Matplotlib日期格式
        time_data = pd.to_datetime(self.data['Open Time'])
        close_prices = self.data['Close Price'].values

        # 设置非常宽的图像，调整figsize
        fig, ax = plt.subplots(figsize=(100, 6))  # 图片宽度为100，高度为6

        # 绘制K线图
        ax.plot(time_data, close_prices, label='Close Price', color='blue')

        # 标注买入信号
        if self.buy_signals:
            buy_times, buy_prices = zip(*self.buy_signals)
            ax.scatter(pd.to_datetime(buy_times), buy_prices, marker='^', color='green', label='Buy Signal', s=100)

        # 标注卖出信号
        if self.sell_signals:
            sell_times, sell_prices = zip(*self.sell_signals)
            ax.scatter(pd.to_datetime(sell_times), sell_prices, marker='v', color='red', label='Sell Signal', s=100)

        # 格式化时间轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)

        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        plt.legend()
        plt.title('Trading Strategy Visualization')
        plt.show()
        return plt


