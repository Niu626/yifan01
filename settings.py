import argparse

IMG_RESIZE = 256  # 图片尺寸

def str2bool(v):
    """
    将字符串转换为bool值，用于argparse
    支持: True/False, true/false, 1/0, yes/no, on/off
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
USING_L1_LOSS = True  # 表示在模型训练中选择使用L1损失函数。L1损失函数，也称为平均绝对误差（MAE），是深度学习中常用的一种损失函数，用于衡量模型预测结果与真实标签之间的平均绝对误差。

EPOCHS = 2000  # 增加训练轮数，因为Transformer需要更多时间收敛

LEARNING_RATE = 1e-4  # 提高初始学习率，因为使用OneCycleLR会自动调整

RELOAD_CHECKPOINT = False  # 这个参数的使用False代表从头开始训练，若为True则代表要在LSMI-U的预训练模型上进行微调，即启用预训练模型权重加载。 对应论文中的 4.2

PATH_TO_PTH_CHECKPOINT = ""
DATA_PATH = 'LSMI_dataset/nikon_512'

BS = 4  # 增大batch size，因为Transformer模型参数量大，需要更大的batch size来稳定训练

# Transformer-UNet 特定参数
D_MODEL = 512  # Transformer的隐藏维度
NHEAD = 8  # 注意力头数
NUM_LAYERS = 6  # Transformer层数

# OneCycleLR相关参数
WARMUP_RATIO = 0.3    # 预热阶段比例
DIV_FACTOR = 25       # 初始学习率除数
FINAL_DIV_FACTOR = 1e4  # 最终学习率除数

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l1_loss', type=str2bool, nargs='?', const=True,
                        default=USING_L1_LOSS)  # 这里的 l1_loss 为True时代表使用L1损失。 若是为False时 ，代表使用L2损失。  这里的 USING_L1_LOSS 的默认值为True，表示默认情况下使用 L1 损失

    parser.add_argument('--angular_loss_weight', type=float, default=0.1,
                        help='权重：将 MAE(角度误差) 加入到 UV L1/MSE 损失中的系数')
    
    # 网络结构优化开关（用于消融实验）
    parser.add_argument('--use_local_global_attention', type=str2bool, nargs='?', const=True,
                        default=True,
                        help='是否使用局部-全局混合注意力机制（True/yes/1=使用，False/no/0=使用原始全局注意力）')

    # model hyer-parameters
    parser.add_argument('--use_pretrain', type=str2bool, nargs='?', const=True,
                        default=RELOAD_CHECKPOINT)
    parser.add_argument('--model_path', type=str, default=PATH_TO_PTH_CHECKPOINT if RELOAD_CHECKPOINT else None)
    parser.add_argument('--model_type', type=str, default='Transformer_UNet')  # 修改默认模型类型
    parser.add_argument('--img_ch', type=int, default=3)  # 输入的是UVL三通道
    parser.add_argument('--output_ch', type=int, default=2)  # 经过U-Net预测，输出的是UV俩通道

    # Transformer-UNet 特定参数
    parser.add_argument('--d_model', type=int, default=D_MODEL)
    parser.add_argument('--nhead', type=int, default=NHEAD)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)

    # OneCycleLR参数
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO)
    parser.add_argument('--div_factor', type=float, default=DIV_FACTOR)
    parser.add_argument('--final_div_factor', type=float, default=FINAL_DIV_FACTOR)

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 添加权重衰减
    parser.add_argument('--beta1', type=float, default=0.9)  # 调整Adam优化器的beta1
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=666)  # 固定随机种子

    # dataset & loader config
    parser.add_argument('--trdir', type=str, default=DATA_PATH)
    parser.add_argument('--camera', type=str, default='galaxy')
    parser.add_argument('--image_size', type=int, default=IMG_RESIZE)
    parser.add_argument('--image_pool', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--input_type', type=str, default='uvl', choices=['rgb', 'uvl'])
    parser.add_argument('--output_type', type=str, default='uv', choices=['illumination', 'uv', 'mixmap'])
    parser.add_argument('--uncalculable', type=int, default=None)
    parser.add_argument('--mask_black', type=int, default=None)
    parser.add_argument('--mask_highlight', type=int, default=None)
    parser.add_argument('--mask_uncalculable', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=8)  # 减少worker数量，避免内存占用过高

    # data augmentation config
    parser.add_argument('--random_crop', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--illum_augmentation', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--sat_min', type=float, default=0.2)
    parser.add_argument('--sat_max', type=float, default=0.8)
    parser.add_argument('--val_min', type=float, default=1.0)
    parser.add_argument('--val_max', type=float, default=1.0)
    parser.add_argument('--hue_threshold', type=float, default=0.2)

    # path config
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=20,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[0, 1],
                        help='0 for single-GPU, 1 for multi-GPU')  # 使用1个GPU
    parser.add_argument('--save_result', type=str, default='no')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--patience', type=int, default=200)  # 增加早停耐心值
    parser.add_argument('--change_log', type=str)
    config = parser.parse_args()

    return config
