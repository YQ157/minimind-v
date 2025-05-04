import cv2
import numpy as np
import os
import random

def data_augmentation(input_image_path, output_dir, num_images=100):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始图像
    img = cv2.imread(input_image_path)

    # 定义数据增强操作的范围
    rotation_range = (-30, 30)  # 旋转角度范围
    scale_range = (0.8, 1.2)    # 缩放范围
    shift_range = (-20, 20)     # 平移范围（像素）
    brightness_range = (-50, 50) # 亮度调整范围
    contrast_range = (0.5, 1.5) # 对比度调整范围

    for i in range(num_images):
        # 创建原始图像的副本进行增强
        augmented_img = img.copy()

        # 随机旋转
        angle = random.uniform(rotation_range[0], rotation_range[1])
        h, w = augmented_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, (w, h))

        # 随机缩放
        scale = random.uniform(scale_range[0], scale_range[1])
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)
        augmented_img = cv2.resize(augmented_img, (scaled_w, scaled_h))

        # 随机平移
        shift_x = random.randint(shift_range[0], shift_range[1])
        shift_y = random.randint(shift_range[0], shift_range[1])
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented_img = cv2.warpAffine(augmented_img, translation_matrix, (w, h))

        # 随机调整亮度和对比度
        brightness = random.randint(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        augmented_img = cv2.convertScaleAbs(augmented_img, alpha=contrast, beta=brightness)

        # 保存增强后的图像
        output_path = os.path.join(output_dir, f"augmented_{i}.jpg")
        cv2.imwrite(output_path, augmented_img)

# 使用示例
input_image_path = "your_input_image.jpg"  # 替换为你的图像路径
output_dir = "augmented_images"
data_augmentation(input_image_path, output_dir)