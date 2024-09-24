import gym
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
from gym import spaces
import config  # 包含 TX_TAX 定义
from agent import action_to_opt


class TradingEnv(gym.Env):
    def __init__(self, data, max_actions=500):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_step = 0  # 操作计数器
        self.max_actions = max_actions  # 最大操作次数
        self.initial_balance = 1000000  # 1M
        self.balance = self.initial_balance
        self.shares_held = 1.0
        self.current_price = self.data['Close Price'].values[self.current_step]

        # 初始化各个操作的计数器
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # 动作空间：13个离散值(0 或 1)，1个用于买卖，6个用于控制数量级，6个用于阈值
        self.action_space = spaces.MultiBinary(13)
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
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step][
            ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'normalized_close']].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_price = self.data['Close Price'].values[self.current_step]
        reward = 0
        trade_opt = action_to_opt(action)[0]
        # 检查各个数量是否大于阈值，只有大于时才执行操作
        if trade_opt > 0:  # Buy
            buy_amount = min(trade_opt, self.balance / (self.current_price * (1 + config.TX_TAX)))  # 包含手续费
            if buy_amount > 0:
                self.shares_held += buy_amount
                self.balance -= buy_amount * self.current_price * (1 + config.TX_TAX)  # 减去手续费
                self.buy_signals.append(
                    (self.data['Open Time'].values[self.current_step], self.current_price, buy_amount))
                self.buy_count += 1
                self.action_step += 1  # 计入操作次数
            else:
                reward -= 10  # 没有足够的资金，给负奖励
        elif trade_opt < 0:  # Sell
            sell_amount = min(trade_opt, self.shares_held)  # 不能卖超过持有的货币数量
            if sell_amount > 0:
                self.shares_held -= sell_amount
                self.balance += sell_amount * self.current_price * (1 - config.TX_TAX)  # 减去手续费
                self.sell_signals.append(
                    (self.data['Open Time'].values[self.current_step], self.current_price, sell_amount))
                self.sell_count += 1
                self.action_step += 1  # 计入操作次数
            else:
                reward -= 10  # 没有货币可卖，给负奖励
        else:  # Hold (观望)
            reward -= 0.1  # 轻微负奖励，鼓励智能体寻找合适的交易机会
            self.hold_count += 1
            trade_operation = 0  # 观望，输出为 0

        # 每个时间步长都会增加，不论是否执行操作
        self.current_step += 1

        # 结束条件：超过最大步数或者最大操作次数
        done = (self.current_step >= len(self.data) - 1) or (self.action_step >= self.max_actions)

        # 返回观察值、奖励、是否完成、以及调试信息（包括 trade_operation）
        return self._get_observation(), reward, done, trade_opt

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
            buy_times, buy_prices, buy_amounts = zip(*self.buy_signals)
            ax.scatter(pd.to_datetime(buy_times), buy_prices, marker='^', color='green', label='Buy Signal',
                       s=np.array(buy_amounts) * 10)

        # 标注卖出信号
        if self.sell_signals:
            sell_times, sell_prices, sell_amounts = zip(*self.sell_signals)
            ax.scatter(pd.to_datetime(sell_times), sell_prices, marker='v', color='red', label='Sell Signal',
                       s=np.array(sell_amounts) * 10)

        # 格式化时间轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)

        # 输出买入、卖出、观望次数的比例
        total_actions = self.buy_count + self.sell_count + self.hold_count
        buy_ratio = (self.buy_count / total_actions) * 100 if total_actions > 0 else 0
        sell_ratio = (self.sell_count / total_actions) * 100 if total_actions > 0 else 0
        hold_ratio = (self.hold_count / total_actions) * 100 if total_actions > 0 else 0
        print(f"Buy: {buy_ratio:.2f}%, Sell: {sell_ratio:.2f}%, Hold: {hold_ratio:.2f}%")

        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        plt.legend()
        plt.title('Trading Strategy Visualization')
        plt.show()
        return plt
