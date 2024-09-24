import pandas as pd
import numpy as np


def load_kline_data(filepath):
    # 读取CSV文件，跳过第一行的列标题，并将Open Time解析为整数类型
    data = pd.read_csv(filepath,header=0,
                       names=['Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume'])
    # 将Open Time列转换为可读时间格式（跳过这个步骤如果不需要）
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms', errors='coerce')

    # 归一化处理收盘价格
    data['normalized_close'] = (data['Close Price'] - data['Close Price'].mean()) / data['Close Price'].std()

    # 返回所需的列
    return data[['Open Time','Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'normalized_close']]
