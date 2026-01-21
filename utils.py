from matplotlib.ticker import MaxNLocator
from torch import Tensor
import numpy as np
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import logging
import math
import torch.nn as nn
import colorsys
import torch.nn.functional as F
import random


# 完整的运行以及调用流程
# （1）训练阶段
#    1.数据加载：通过 dataset.py 调用 rgb2uvl 将输入RGB转为UVL格式。 应用 RandomColor 增强光照多样性。
#    2.模型预测：U-Net输入UVL图像，预测UV通道。
#    3.损失计算:使用 criterion_l1_loss() 计算预测UV与GT的L1损失。
#    4.评估：通过 get_MAE 计算角度误差，Evaluator 汇总指标。
#  （2）测试阶段
#    1.后处理：调用 apply_wb 将预测UV转换为RGB图像。使用 post_processing.py 进行物理约束优化（如噪声去除）。
#    2.结果可视化：plot_illum 绘制光照估计结果对比图（类似论文图6）。

EPS = 1e-8 # 其中 ε=1e-8 用于避免 log(0)。

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强与预处理

# 随机光照扰动 (RandomColor) ， 在训练时随机扰动光源色度，增强数据多样性 ， 论文对应：第3.1节的"random data augmentation"。

class RandomColor:
    def __init__(self, sat_min, sat_max, val_min, val_max, hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self, h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def threshold_test(self, hue_list, hue):
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in illum_count:
            while True:
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(self.sat_min, self.sat_max)
                value = np.random.uniform(self.val_min, self.val_max)
                chroma_rgb = np.array(self.hsv2rgb(hue, saturation, value), dtype='float32')
                chroma_rgb /= chroma_rgb[1]    # 归一化到G通道

                if self.threshold_test(hue_list, hue):
                    hue_list.append(hue)
                    ret_chroma[int(i) - 1] = chroma_rgb
                    break

        return np.array(ret_chroma)

# 白平衡应用 (apply_wb) ，将预测的UV值转换回RGB空间，实现白平衡校正。 用于生成最终校正后的图像（图6中的"Ours"列）。  最终白平衡校正
def apply_wb(org_img, pred, pred_type):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """
    pred_rgb = torch.zeros_like(org_img)  # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:, 1, :, :] = org_img[:, 1, :, :]
        pred_rgb[:, 0, :, :] = org_img[:, 0, :, :] / (pred[:, 0, :, :] + 1e-8)  # R_wb = R / illum_R
        pred_rgb[:, 2, :, :] = org_img[:, 2, :, :] / (pred[:, 2, :, :] + 1e-8)  # B_wb = B / illum_B
    elif pred_type == "uv":
        # 从 UV 重建 RGB
        pred_rgb[:, 1, :, :] = org_img[:, 1, :, :]   # G 通道不变
        pred_rgb[:, 0, :, :] = org_img[:, 1, :, :] * torch.exp(pred[:, 0, :, :])  # R = G * (R/G)
        pred_rgb[:, 2, :, :] = org_img[:, 1, :, :] * torch.exp(pred[:, 1, :, :])  # B = G * (B/G)

    return pred_rgb

#  评估与损失函数

#  角度误差计算 (get_MAE) , 角度误差评估标准
def get_MAE(pred, gt, tensor_type='uv', camera=None, mode='uv'):
    """
    pred : (b,c,w,h)
    gt : (b,c,w,h)
    """
    if tensor_type == "rgb":
        if camera == 'galaxy':
            pred = torch.clamp(pred, 0, 1023)
            gt = torch.clamp(gt, 0, 1023)
        elif camera == 'sony' or camera == 'nikon':
            pred = torch.clamp(pred, 0, 16383)
            gt = torch.clamp(gt, 0, 16383)

    cos_similarity = F.cosine_similarity(pred + 1e-4, gt + 1e-4, dim=1)
    cos_similarity = torch.clamp(cos_similarity, -0.999999, 0.999999)
    rad = torch.acos(cos_similarity)
    ang_error = torch.rad2deg(rad)

    # if mask is not None:
    #     ang_error = ang_error[torch.squeeze(mask, 1) != 0]
    mean_angular_error = ang_error.mean()
    return mean_angular_error

def criterion_loss():
    return nn.MSELoss(reduction='mean')

# 实现论文提出的L1损失优化（第4.2节），相比L2对噪声更鲁棒。 抑制低比特噪声影响 .  L1 损失对噪声更鲁棒（论文 Section 3.2）
def criterion_l1_loss():
    return nn.L1Loss(reduction='mean')   # 最小化预测UV 与GT UV的差异


def print_metrics(current_metrics: dict, best_metrics: dict):
    message = (f"\n{'*' * 50}\n"
               f" Mean ......... : {current_metrics['mean']:.4f} (Best: {best_metrics['mean']:.4f})\n"
               f" Median ....... : {current_metrics['median']:.4f} (Best: {best_metrics['median']:.4f})\n"
               f" Trimean ...... : {current_metrics['trimean']:.4f} (Best: {best_metrics['trimean']:.4f})\n"
               f" Best 25% ..... : {current_metrics['bst25']:.4f} (Best: {best_metrics['bst25']:.4f})\n"
               f" Worst 25% .... : {current_metrics['wst25']:.4f} (Best: {best_metrics['wst25']:.4f})\n"
               f" Worst 5% ..... : {current_metrics['wst5']:.4f} (Best: {best_metrics['wst5']:.4f})\n"
               f"{'*' * 50}"
               )
    logging.info(message)

#  训练辅助工具

# (1) 评估指标跟踪 (Evaluator), 统计训练过程中的均值、中位数、最佳25%等误差指标。 表1中的性能指标。
class Evaluator:

    def __init__(self):
        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst5"]
        self.__metrics = {}
        self.__best_metrics = {m: 100.0 for m in monitored_metrics}
        self.__errors = []

    def add_error(self, error: float):
        self.__errors.append(error)

    def reset_errors(self):
        self.__errors = []

    def get_errors(self) -> list:
        return self.__errors

    def get_metrics(self) -> dict:
        return self.__metrics

    def get_best_metrics(self) -> dict:
        return self.__best_metrics

    def compute_metrics(self) -> dict:
        self.__errors = sorted(self.__errors)
        self.__metrics = {
            "mean": np.mean(self.__errors),
            "median": self.__g(0.5),
            "trimean": 0.25 * (self.__g(0.25) + 2 * self.__g(0.5) + self.__g(0.75)),
            "bst25": np.mean(self.__errors[:int(0.25 * len(self.__errors))]),
            "wst25": np.mean(self.__errors[int(0.75 * len(self.__errors)):]),
            "wst5": self.__g(0.95)
        }
        return self.__metrics

    def update_best_metrics(self) -> dict:
        self.__best_metrics["mean"] = self.__metrics["mean"]
        self.__best_metrics["median"] = self.__metrics["median"]
        self.__best_metrics["trimean"] = self.__metrics["trimean"]
        self.__best_metrics["bst25"] = self.__metrics["bst25"]
        self.__best_metrics["wst25"] = self.__metrics["wst25"]
        self.__best_metrics["wst5"] = self.__metrics["wst5"]
        return self.__best_metrics

    def __g(self, f: float) -> float:
        return np.percentile(self.__errors, f * 100)

# 比特深度转换 (convert_14bit2others_bits) ， 将14-bit图像模拟为低比特深度（如10-bit），用于分析比特深度影响。 论文对应：第2.1节的比特深度分析实验。 对用论文中的公式（1）
def convert_14bit2others_bits(data, bit_depth=10):
    "The default bit-depth is 14bit from Nikon in LSMI dataset"

    data = data.astype('float32')
    data = (data / (2 ** 14 - 1)) * (2 ** bit_depth - 1)
    data = np.round(data)

    return data.astype('float32')

# (2) 损失跟踪 (LossTracker) , 记录损失函数的移动平均值，用于监控训练稳定性。
class LossTracker(object):

    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_loss(self) -> float:
        return self.avg

# 将 RGB 图像转换到对数色度空间（UVL），这是论文方法的核心预处理步骤。  对数色度空间提升低比特鲁棒性.
# 输入图像预处理（RGB - UVL）
def rgb2uvl(img_rgb):
    epsilon = 1e-8

    # 下面这是新加的
    img_rgb = np.clip(img_rgb, epsilon, None)  # 显式截断负值和零

    img_uvl = np.zeros_like(img_rgb, dtype='float32')
    img_uvl[:, :, 2] = np.log(img_rgb[:, :, 1] + epsilon)                        # L = log(G)  ，  L 通道：对数亮度（基于绿色通道 G）
    # U/V 通道：对数色度（R/G 和 B/G 的比值）。
    img_uvl[:, :, 0] = np.log(img_rgb[:, :, 0] + epsilon) - img_uvl[:, :, 2]     # U = log(R/G)
    img_uvl[:, :, 1] = np.log(img_rgb[:, :, 2] + epsilon) - img_uvl[:, :, 2]     # V = log(B/G)

    return img_uvl


def plot_illum(pred_map=None, gt_map=None):
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:, 0], pred_map[:, 1], 'ro')
    if gt_map is not None:
        plt.plot(gt_map[:, 0], gt_map[:, 1], 'bx')

    minx, miny = min(gt_map[:, 0]), min(gt_map[:, 1])
    maxx, maxy = max(gt_map[:, 0]), max(gt_map[:, 1])
    lenx = (maxx - minx) / 2
    leny = (maxy - miny) / 2
    add_len = max(lenx, leny) + 0.3

    center_x = (maxx + minx) / 2
    center_y = (maxy + miny) / 2

    plt.xlim(center_x - add_len, center_x + add_len)
    plt.ylim(center_y - add_len, center_y + add_len)

    # make square
    plt.gca().set_aspect('equal', adjustable='box')

    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)


def mix_chroma(mixmap, chroma_list, illum_count):
    ret = np.stack((np.zeros_like(mixmap[:, :, 0], dtype=float),) * 3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i]) - 1
        mixmap_3ch = np.stack((mixmap[:, :, i],) * 3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])

    return ret


def hwc_to_chw(img: np.ndarray):
    """ Converts an image from height * width * channel to (channel * height * width)"""
    return img.transpose(2, 0, 1)


def chw_to_hwx(x: Tensor) -> Tensor:
    """ Converts a Tensor to an Image """
    img = x.cpu().numpy()
    img = img.transpose(0, 2, 3, 1)[0, :, :, :]
    return img


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def log_sys(args):
    dt = datetime.now()
    path_to_log = os.path.join('./log', args.data_name,
                               f'fold_{args.fold_num}_'
                               f'-{dt.day}-{dt.hour}-{dt.minute}')

    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics_log = os.path.join(path_to_log, 'error.csv')
    vis_log_tr = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'train')
    vis_log_acc = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'acc')
    os.makedirs(vis_log_tr, exist_ok=True)
    os.makedirs(vis_log_acc, exist_ok=True)

    param_info = {'lr': args.lr, 'batch_size': args.batch_size,
                  'fold_num': args.fold_num, 'data_name': args.data_name,
                  'time_file': f'{dt.day}-{dt.hour}-{dt.minute}',
                  'seed': f'{args.seed}'}

    return SummaryWriter(vis_log_tr), SummaryWriter(vis_log_acc), \
           path_to_log, path_to_metrics_log, param_info


def save_log(log_dir):
    log_filename = datetime.datetime.now().strftime("%H-%M-") + f"results.txt"
    log_file = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(file_handler)


# Plot statistics at each checkpoint.
def plot_per_check(stats_dir, title, measurements, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.pdf'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(stats_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def set_seed(seed: int):
    """
    Set random seed for all relevant libraries to ensure reproducibility.
    """
    # 1) 环境变量，固定 PYTHON hash 随机化
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2) Python、NumPy、Torch CPU、Torch CUDA
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 3) CuDNN 确定性，禁用 benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle)

