# -*- coding: utf-8 -*-
"""
海洋环境监测系统配置文件
配置数据集类别和相关参数
"""

# 数据集类别配置
# 类别索引从0开始
DATASET_CLASSES = {
    0: 'Mask',          # 口罩/面具
    1: 'can',           # 易拉罐
    2: 'cellphone',     # 手机
    3: 'electronics',   # 电子产品
    4: 'gbottle',       # 玻璃瓶
    5: 'glove',         # 手套
    6: 'metal',         # 金属
    7: 'misc',          # 其他杂物
    8: 'net',           # 网
    9: 'pbag',          # 塑料袋
    10: 'pbottle',      # 塑料瓶
    11: 'plastic',      # 塑料
    12: 'rod',          # 杆/棒
    13: 'sunglasses',   # 太阳镜
    14: 'tire'          # 轮胎
}

# 类别名称列表（按索引顺序）
CLASS_NAMES = ['Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal', 'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire']

# 重点关注类别（用于触发警报，主要是塑料类污染物）
FOCUS_CLASSES = ['pbag', 'pbottle', 'plastic', 'net']

# 危险类别（兼容旧版本，与FOCUS_CLASSES相同）
DANGEROUS_CLASSES = ['pbag', 'pbottle', 'plastic', 'net']

# 类别颜色配置（BGR格式，用于绘制边界框）
CLASS_COLORS = {
    'Mask': (255, 0, 0),          # 蓝色
    'can': (0, 255, 255),        # 黄色
    'cellphone': (255, 255, 0),  # 青色
    'electronics': (128, 0, 128), # 紫色
    'gbottle': (0, 255, 0),      # 绿色
    'glove': (255, 165, 0),      # 橙色
    'metal': (192, 192, 192),    # 银色
    'misc': (128, 128, 128),     # 灰色
    'net': (0, 0, 255),          # 红色（重点关注）
    'pbag': (0, 0, 255),         # 红色（重点关注）
    'pbottle': (0, 0, 255),      # 红色（重点关注）
    'plastic': (0, 0, 255),      # 红色（重点关注）
    'rod': (255, 192, 203),      # 粉色
    'sunglasses': (0, 128, 255), # 深蓝色
    'tire': (0, 0, 128)          # 深红色
}

# 类别中文名称（用于显示）
CLASS_NAMES_CN = {
    'Mask': '口罩/面具',
    'can': '易拉罐',
    'cellphone': '手机',
    'electronics': '电子产品',
    'gbottle': '玻璃瓶',
    'glove': '手套',
    'metal': '金属',
    'misc': '其他杂物',
    'net': '网',
    'pbag': '塑料袋',
    'pbottle': '塑料瓶',
    'plastic': '塑料',
    'rod': '杆/棒',
    'sunglasses': '太阳镜',
    'tire': '轮胎'
}

# 检测参数配置
DETECTION_CONFIG = {
    'conf_threshold': 0.25,  # 置信度阈值
    'iou_threshold': 0.45,   # IOU阈值
    'img_size': 640          # 输入图像尺寸
}

