import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from samnet1 import SAMNET
from data import test_dataset  # 自定义的数据加载模块

# 设置设备
torch.cuda.set_device(0)

# 解析输入参数
class opt:
    test_model = "/data/lxy-workspace/samsod/checkpoints/shuangliu12notrans-nokan-gate/SAMNET_best.pth"  # 模型路径
    test_data_root = "./VT/"  # 测试数据根目录
    maps_path = "./results/"  # 结果保存路径
    testsize = 512  # 测试图像尺寸


# 加载模型
print("Loading SAMNET model...")
model = SAMNET(checkpoint_path=opt.test_model)

model.cuda()
model.eval()

# 测试集
test_sets = ["VT5000/Test","VT1000","VT821"]

for dataset in test_sets:
    save_path = os.path.join(opt.maps_path, dataset)
    save_path_mask = os.path.join(save_path, "mask")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_mask):
        os.makedirs(save_path_mask)

    dataset_path = os.path.join(opt.test_data_root, dataset)
    test_loader = test_dataset(dataset_path, opt.testsize)

    # 开始测试
    for i in range(test_loader.size):
        vis_image, inf_image, gt, (H, W), name = test_loader.load_data()
        vis_image, inf_image = vis_image.cuda(), inf_image.cuda()
        shape = (W, H)

        # 前向传播
        mask,_,_,pred = model(vis_image, inf_image)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()  # 将预测结果转为 numpy 格式
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 归一化到 [0, 1]
        mask = mask.sigmoid().data.cpu().numpy().squeeze()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # 保存结果
        result_path = os.path.join(save_path, name)
        cv2.imwrite(result_path, (pred * 255).astype(np.uint8))
        print(f"Saved result to: {result_path}")

        # 保存掩膜
        mask_path = os.path.join(save_path_mask, name)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"Saved mask to: {mask_path}")

print("Test Done!")