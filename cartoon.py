# -*- coding = utf-8 -*-
"""
# @Time : 2023/9/22 20:18
# @Author : FriK_log_ff 374591069
# @File : newmyway.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import cv2
import numpy as np
import os


# 读取文件
def read_file(filename: str) -> np.ndarray:
    try:
        img = cv2.imread(filename)
        if img is None:
            raise ValueError("Invalid file path or file format.")
        return img
    except:
        raise ValueError("Invalid file path or file format.")


# 显示图片
def display_image(img: np.ndarray, window_name: str) -> None:
    cv2.imshow(window_name, img)
    cv2.waitKey()


# 边缘掩膜
def edge_mask(image: np.ndarray, line_size: int, blur_value: int) -> np.ndarray:
    if not isinstance(line_size, int) or not isinstance(blur_value, int) or line_size < 1 or blur_value < 1:
        raise ValueError("Invalid value for 'line_size' or 'blur_value' parameter. Must be a positive integer.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


# 颜色量化
def color_quantization(image: np.ndarray, num_colors: int) -> np.ndarray:
    if not isinstance(num_colors, int) or num_colors < 1:
        raise ValueError("Invalid value for 'num_colors' parameter. Must be a positive integer.")

    # 转换图片
    data = np.float32(image).reshape((-1, 3))

    # 设置KMeans聚类参数
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 执行KMeans聚类
    _, labels, centers = cv2.kmeans(data, num_colors, None, kmeans_criteria, 10, flags)
    centers = np.uint8(centers)
    processed_image = centers[labels.flatten()]
    processed_image = processed_image.reshape(image.shape)

    # 应用颜色增强
    hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * 1.5  # 增强饱和度
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


# 上传文件
def cartoonize_single_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file or file format.")

        image = resize_crop(image)

        # 设置边缘掩膜参数并应用
        line_size = 7
        blur_value = 7
        edges = edge_mask(image, line_size, blur_value)

        # 执行颜色量化
        num_colors = 9
        processed_image = color_quantization(image, num_colors)

        # 应用双边滤波
        blurred = cv2.bilateralFilter(processed_image, d=9, sigmaColor=200, sigmaSpace=200)

        # 应用掩膜
        cartoonized_image = cv2.bitwise_and(blurred, blurred, mask=edges)

        return cartoonized_image
    except Exception as e:
        print(f"Failed to cartoonize {image_path}: {e}")
        return None


# if __name__ == '__main__':
#     image_path = "C:\\Users\\86155\\Desktop\\image2.png"  # 修改成你要处理的图片路径
#     cartoonized_image = cartoonize_single_image(image_path)
#     if cartoonized_image is not None:
#         display_image(cartoonized_image, "Cartoonized Image")