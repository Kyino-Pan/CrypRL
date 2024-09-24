import os
import json
from datetime import datetime
from config import *

def create_results_dir():
    # 获取当前时间
    now = datetime.now()
    time_str = now.strftime("train_%d_%H_%M_%S")  # 格式为 train_dd_hh_mm_ss

    # 创建训练结果目录
    results_dir = os.path.join('./train_results/', time_str)
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def save_hyperparameters(results_dir, hyperparams, epoch):
    # 为每一轮创建一个子目录 epoch_X
    epoch_dir = os.path.join(results_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # 保存超参数为 hyper.json
    hyper_file = os.path.join(epoch_dir, 'hyper.json')
    with open(hyper_file, 'w') as f:
        json.dump(hyperparams, f, indent=4)


def save_model(results_dir, model, epoch):
    # 为每一轮创建一个子目录 epoch_X
    epoch_dir = os.path.join(results_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # 保存模型为 model.pth
    model_file = os.path.join(epoch_dir, 'model.pth')
    torch.save(model.state_dict(), model_file)


def save_plot(results_dir, env, epoch):
    # 为每一轮创建一个子目录 epoch_X
    epoch_dir = os.path.join(results_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    plot = env.plot_trades()
    plot_file = os.path.join(epoch_dir, 'trading_strategy.png')
    plot.savefig(plot_file)
    plot.close()


def save_checkpoint(results_dir, model, optimizer, epoch, total_reward):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_reward': total_reward  # 累计奖励等其他信息
    }

    checkpoint_file = os.path.join(results_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(results_dir, model, optimizer):
    checkpoint_file = os.path.join(results_dir, 'checkpoint.pth')
    if RETRAIN:
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            total_reward = checkpoint['total_reward']
            print(f"Checkpoint loaded: resuming from epoch {epoch}")
            return epoch, total_reward
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0  # 从头开始


