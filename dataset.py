
# 文件实现了用于"低比特深度图像的鲁棒像素级光照估计算法"的数据加载和处理功能。

# 多光源支持：可以处理包含1-3个不同光源的场景
# 数据增强：包括随机裁剪和颜色变换
# UVL颜色空间转换：将RGB图像转换为论文中提出的UVL表示
# 混合光照处理：通过混合图(mixture map)处理多光源场景
# 掩码支持：标记无效或不可计算的像素区域

import os.path   # 处理文件路径

import cv2   # 图像读取和颜色空间转换
import json  # 解析元数据文件

import matplotlib.pyplot as plt             # 调试时可视化图像（实际代码中未使用）
import torchvision.transforms.functional as TF   # TF（PyTorch图像变换工具）：实现裁剪、缩放等操作
from torch.utils import data   # PyTorch数据集基类
from torchvision import transforms  # 数据预处理流水线
from torchvision.transforms import RandomResizedCrop  # 随机裁剪的辅助工具

from settings import IMG_RESIZE  # 从配置文件导入的图像目标尺寸
from utils import *  # 自定义工具函数（如 rgb2uvl）


class LSMI(data.Dataset):
    def __init__(self, root, split,
                 input_type='uvl', output_type='uv',
                 illum_augmentation=None, transform=None):
        self.root = root  # root：数据集根目录路径
        self.split = split  # split：数据集分割类型（训练集、验证集、测试集）
        self.input_type = input_type  # input_type：指定输入图像格式（RGB或UNL）
        self.output_type = output_type   # 使用UV色度空间表示作为监督信号 ， 论文方法既支持直接预测光照图("illumination")，也支持预测对数色度("uv")，后者对低比特深度图像更鲁棒。
        self.random_color = illum_augmentation  # illum_augmentation: 光照增强方法
        self.transform = transform  # transform: 数据变换方法

        self.image_list = sorted([f for f in os.listdir(os.path.join(root, split))
                                  if f.endswith(".tiff")  # 列出 root/split 目录下所有 .tiff 文件
                                  and len(os.path.splitext(f)[0].split("_")[-1]) in [1, 2, 3]
                                     and 'gt' not in f])  # 排除包含 gt 的文件（GT文件单独处理）

        meta_file = os.path.join(self.root, 'meta.json')  # 加载 meta.json 文件，用于获取每个场景的光源色度数据。
        with open(meta_file, encoding='utf-8-sig') as meta_json:
            self.meta_data = json.load(meta_json)

        logging.info("[Data]\t" + str(self.__len__()) + " " + split + " images are loaded from " + root)

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information 元信息
        input_***       : input image (uvl or rgb) 输入图像(uvl或rgb)
        gt_***          : GT (None or illumination or chromaticity) GT(光照或色度)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels  用于不确定光照(黑色像素)或饱和像素的掩码
        """

        # parse file's name 解析文件名
        filename = os.path.splitext(self.image_list[idx])[0]
        img_file = filename + ".tiff"
        mixture_map = filename + ".npy"
        place, illu_count = filename.split('_')   # 从文件名中提取场景名（place）和光源数量（illu_count）。

        #  1. 准备元信息
        # 初始化包含光照色度的数组（最多支持3个光源）。
        ret_dict = {"illu_chroma": np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])}

        # 从元数据中读取每个光源的色度值，填充到 ret_dict。
        for illu_no in illu_count:
            illu_chroma = self.meta_data[place]["Light" + illu_no]
            ret_dict["illu_chroma"][int(illu_no) - 1] = illu_chroma
        ret_dict["img_file"] = img_file
        ret_dict["place"] = place
        ret_dict["illu_count"] = illu_count

        # 2. 准备输入和输出GT
        # 加载混合图和3通道RGB tiff图像
        # 加载输入图像（BGR → RGB）
        input_path = os.path.join(self.root, self.split, img_file)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')  # 加载 BGR 格式
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)    # 转 RGB

        # 下面这是新加的
        input_rgb = np.clip(input_rgb, a_min=EPS, a_max=None)  # 确保所有像素值 ≥ EPS


        #  处理混合图
        if len(illu_count) != 1:
            mixture_map = np.load(os.path.join(self.root, self.split, mixture_map)).astype('float32')
        else:
            mixture_map = np.ones_like(input_rgb[:, :, 0:1])
        # mixture_map包含-1表示ZERO_MASK，即LSMI的G通道近似无法计算的像素。
        # 如果使用像素级增强，我们必须将负值替换为0
        incalculable_masked_mixture_map = np.where(mixture_map == -1, 0, mixture_map)

        # 随机数据增强
        # 训练时随机扰动光照色度，增强数据多样性。
        if self.random_color and self.split == 'train':
            augment_chroma = self.random_color(illu_count)
            ret_dict["illu_chroma"] *= augment_chroma
            tint_map = mix_chroma(incalculable_masked_mixture_map, augment_chroma, illu_count)
            input_rgb = input_rgb * tint_map

        ret_dict["input_rgb"] = input_rgb

        # 原始 RGB → UVL 转换（rgb2uvl）是模型输入的标准格式。  对应论文公式 (2)
        # 调用 rgb2uvl 转换输入图像 ， 转换后的 input_uvl 会作为 U-Net 的输入
        ret_dict["input_uvl"] = rgb2uvl(input_rgb)   # 转换后的 UVL 格式

        # 准备输出张量
        illu_map = mix_chroma(incalculable_masked_mixture_map, ret_dict["illu_chroma"], illu_count)

        ret_dict["gt_illu"] = np.delete(illu_map, 1, axis=2)

        # 加载GT图像并转换为RGB格式。
        # 对 GT 图像也做同样转换（如果 output_type="uv"）
        output_bgr = cv2.imread(os.path.join(self.root, self.split, filename + "_gt.tiff"),
                                cv2.IMREAD_UNCHANGED).astype('float32')     # 加载 GT
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)   # 白平衡后的RGB

        ret_dict["gt_rgb"] = output_rgb
        output_uvl = rgb2uvl(output_rgb)
        ret_dict["gt_uv"] = np.delete(output_uvl, 2, axis=2)  # 只保留 UV 通道

        # 3. 准备掩码
        # 训练时加载掩码（标记无效区域），测试时用全1掩码。
        if self.split == 'train':
            mask = cv2.imread(os.path.join(self.root, self.split, place + '_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = mask[:, :, None].astype('float32')
        else:
            mask = np.ones_like(input_rgb[:, :, 0:1], dtype='float32')

        ret_dict["mask"] = mask
        # 4. 应用变换
        if self.transform is not None:
            ret_dict = self.transform(ret_dict)

        return ret_dict

    def __len__(self):
        return len(self.image_list)


# 数据变换类
class PairedRandomCrop:
    def __init__(self, size=(256, 256), scale=(0.3, 1.0), ratio=(1., 1.)):
        self.size = size  # 输出尺寸
        self.scale = scale  # 缩放范围
        self.ratio = ratio   # 宽高比范围
    # 对输入和GT同步执行随机裁剪和缩放，保持空间对齐。
    def __call__(self, ret_dict):
        i, j, h, w = RandomResizedCrop.get_params(img=ret_dict['input_rgb'], scale=self.scale, ratio=self.ratio)
        ret_dict['input_rgb'] = TF.resized_crop(ret_dict['input_rgb'], i, j, h, w, self.size)
        ret_dict['input_uvl'] = TF.resized_crop(ret_dict['input_uvl'], i, j, h, w, self.size)
        ret_dict['gt_illu'] = TF.resized_crop(ret_dict['gt_illu'], i, j, h, w, self.size)
        ret_dict['gt_rgb'] = TF.resized_crop(ret_dict['gt_rgb'], i, j, h, w, self.size)
        ret_dict['gt_uv'] = TF.resized_crop(ret_dict['gt_uv'], i, j, h, w, self.size)
        ret_dict['mask'] = TF.resized_crop(ret_dict['mask'], i, j, h, w, self.size)

        return ret_dict


class Resize:
    def __init__(self, size=(256, 256)):
        self.size = size  # 目标尺寸

    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = TF.resize(ret_dict['input_rgb'], self.size)
        ret_dict['input_uvl'] = TF.resize(ret_dict['input_uvl'], self.size)
        ret_dict['gt_illu'] = TF.resize(ret_dict['gt_illu'], self.size)
        ret_dict['gt_rgb'] = TF.resize(ret_dict['gt_rgb'], self.size)
        ret_dict['gt_uv'] = TF.resize(ret_dict['gt_uv'], self.size)
        ret_dict['mask'] = TF.resize(ret_dict['mask'], self.size)

        return ret_dict

# 转换为张量类 ToTensor
class ToTensor:
    # 将NumPy数组转为PyTorch张量，并调整维度为 C×H×W。
    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = torch.from_numpy(ret_dict['input_rgb'].transpose((2, 0, 1)))
        ret_dict['input_uvl'] = torch.from_numpy(ret_dict['input_uvl'].transpose((2, 0, 1)))
        ret_dict['gt_illu'] = torch.from_numpy(ret_dict['gt_illu'].transpose((2, 0, 1)))
        ret_dict['gt_rgb'] = torch.from_numpy(ret_dict['gt_rgb'].transpose((2, 0, 1)))
        ret_dict['gt_uv'] = torch.from_numpy(ret_dict['gt_uv'].transpose((2, 0, 1)))
        ret_dict['mask'] = torch.from_numpy(ret_dict['mask'].transpose((2, 0, 1)))

        return ret_dict

# 随机颜色变换类 RandomColor
class RandomColor:
    def __init__(self, sat_min, sat_max, val_min, val_max, hue_threshold):
        self.sat_min = sat_min  # 饱和度最小值
        self.sat_max = sat_max  # 饱和度最大值
        self.val_min = val_min  # 亮度最小值
        self.val_max = val_max  # 亮度最大值
        self.hue_threshold = hue_threshold  # 色调阈值
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
                chroma_rgb /= chroma_rgb[1]

                if self.threshold_test(hue_list, hue):
                    hue_list.append(hue)
                    ret_chroma[int(i) - 1] = chroma_rgb
                    break

        return np.array(ret_chroma)

# 预处理流水线
# 训练数据增强流水线 aug_crop
def aug_crop():
    # 张量转换 → 随机裁剪
    tsfm = transforms.Compose([ToTensor(),
                               PairedRandomCrop(size=(IMG_RESIZE, IMG_RESIZE), scale=(0.3, 1.0),
                                                ratio=(1., 1.))])
    return tsfm

# 验证数据调整大小流水线 val_resize
def val_resize():
    # 张量转换 → 固定尺寸缩放
    tsfm = transforms.Compose([ToTensor(), Resize(size=(IMG_RESIZE, IMG_RESIZE))])

    return tsfm