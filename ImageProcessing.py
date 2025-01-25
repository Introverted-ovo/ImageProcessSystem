import sys
import cv2
import numpy as np
import dlib
import math
from IPython.external.qt_for_kernel import QtCore
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QTransform, QPainter, QColor, qGreen, qRed, qRgb, qBlue, qGray
from PyQt5.QtWidgets import QFileDialog, QLabel, QMessageBox, QInputDialog
from matplotlib import pyplot as plt

from CropLabel import CropLabel
from MyLabel import MyLabel
import region_grow
import jiaquanpinjie,zhijiepinjie
import cartoon
import sumiao


class ImageProcessor:
    def __init__(self):
        self.file_path = None
        self.last_pos = QPoint()
        self.count = 0

    def open_image(self, label):
        # 打开文件对话框以选择图片
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")

        if self.file_path:
            # 使用 OpenCV 读取图片
            image = cv2.imread(self.file_path)

            # 将颜色通道顺序转换为 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 将 OpenCV 图像转换为 PyQt5 支持的格式
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将 QImage 转换为 QPixmap，并在 QLabel 中显示
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def camera_open_image(self, label):
        cap = cv2.VideoCapture(0)
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()

        # 设置帧的宽度和高度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            # 读取摄像头的帧
            ret, frame = cap.read()
            # 显示结果
            cv2.imshow('camera capture', frame)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite("captured_image.jpg", frame)
                break
        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()
        self.file_path = "captured_image.jpg"
        # 将 QImage 转换为 QPixmap，并在 QLabel 中显示
        label.setPixmap(QPixmap(self.file_path))
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def save_image(self,label):
        # 获取 QLabel 中显示的 pixmap
        pixmap = label.pixmap()

        if pixmap is not None:
            # 将 QPixmap 转换为 OpenCV 支持的格式（BGR）
            image = pixmap.toImage()
            image = image.convertToFormat(QImage.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            bgr_image = np.array(ptr).reshape(height, width, 3)  # BGR image

            # 将 BGR 图像转换为 RGB 格式
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # 保存图像
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(None, "Save Image File", "", "Image Files (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, rgb_image)


    def program_exit(self):
        # 创建一个消息框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("退出")
        msg_box.setText("你确定退出吗?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # 检查用户的选择
        response = msg_box.exec_()

        if response == QMessageBox.Yes:
            sys.exit()  # 退出程序
        else:
            pass  # 如果选择 "否"，不做任何事情


    def rotate(self, angle, label):
        # 从 label 中获取当前图片
        pixmap = label.pixmap()
        if pixmap is not None:

            # 创建QTransform对象来执行旋转
            transform = QTransform()
            transform.rotate(angle)

            # 应用变换并更新标签上的图片
            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            label.setPixmap(rotated_pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")

            # 检查用户的选择
            response = msg_box.exec_()

            print("No image found in QLabel.")

    def transform_image(self,label,i):
        pixmap = label.pixmap()
        if pixmap is not None:
            # Convert QPixmap to NumPy array
            image_np = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            width, height = image_np.width(), image_np.height()
            ptr = image_np.bits()
            ptr.setsize(height * width * 3)
            image_opencv = np.array(ptr).reshape(height, width, 3).copy()
            # Convert RGB to BGR (if needed)
            image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
            flipped_image = cv2.flip(image_opencv, i)  # Flip horizontally
            flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
            height, width, channel = flipped_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(flipped_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def translate_scale_start(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            text = label.text()
            layout = label.layout()
            parent = label.parent()
            label.hide()

            self.new_label = MyLabel()
            self.new_label.setObjectName("new1")
            self.new_label.setText(text)
            self.new_label.setPixmap(pixmap)
            if layout:
                self.new_label.setLayout(layout)
            #        parent.layout().removeWidget(label)
            parent.layout().addWidget(self.new_label)
            self.new_label.setCursor(Qt.CrossCursor)  # 设置光标类型
            self.new_label.show()

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")

            # 检查用户的选择
            response = msg_box.exec_()

            print("No image found in QLabel.")

    def translate_scale_end(self,label):
         self.new_label.setParent(None)
         self.new_label.deleteLater()
         label.show()


    def image_crop(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # 读取图片
            self.count = 1
            text = label.text()
            layout = label.layout()
            parent = label.parent()
            label.hide()

            self.new_label1 = CropLabel(label)
            self.new_label1.setObjectName("new2")
            self.new_label1.setText(text)
            self.new_label1.setPixmap(pixmap)
            if layout:
                self.new_label1.setLayout(layout)
            #        parent.layout().removeWidget(label)
            parent.layout().addWidget(self.new_label1)
            self.new_label1.setCursor(Qt.CrossCursor)  # 设置光标类型
            self.new_label1.show()
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

        self.file_path = "cropped_image.jpg"


    def scale_histogram(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # Convert QPixmap to NumPy array
            image_np = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            width, height = image_np.width(), image_np.height()
            ptr = image_np.bits()
            ptr.setsize(height * width * 3)
            image_opencv = np.array(ptr).reshape(height, width, 3).copy()

            # Convert RGB to BGR (if needed)
            image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)

            # Now you can use image_opencv with OpenCV
            # For example, you can calculate histogram
            histogram = []
            for i in range(3):  # for each channel (BGR)
                channel_hist = cv2.calcHist([image_opencv], [i], None, [256], [0, 256], accumulate=False)
                histogram.append(channel_hist)

            # Display histogram
            plt.figure()
            plt.title('Histogram')
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.plot(histogram[i], color=color)
            plt.xlim([0, 256])
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.show()

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")





    def rgb_gray(self,label):
        # 从 label 中获取当前图片
        pixmap = label.pixmap()
        if pixmap is not None:
            # 将 QPixmap 转换为 OpenCV 格式的图像
            image = pixmap.toImage()
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # ARGB8888 格式

            # 将图像从 ARGB 转换为灰度图像
            gray_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

            # 将灰度图像转换为 QPixmap
            gray_pixmap = QPixmap.fromImage(
                QImage(gray_image.data, gray_image.shape[1], gray_image.shape[0], gray_image.strides[0],
                       QImage.Format_Grayscale8))

            # 在 label 中显示灰度图像
            label.setPixmap(gray_pixmap)
            label.setAlignment(Qt.AlignCenter)   #居中显示

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")

            # 检查用户的选择
            response = msg_box.exec_()

            print("No image found in QLabel.")


    def rgb_hsi(self,label):
        # 从 label 中获取当前图片
        pixmap = label.pixmap()
        if pixmap is not None:
            # 读取图片
            image = cv2.imread(self.file_path)
            hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            cv2.imwrite("hsi_image.png", hsi_image)
            pixmap = QPixmap("hsi_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def show_gaussian(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(1)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def add_gaussian_noise(self,label,ui):
        # 读取图片
        image = cv2.imread(self.file_path)
        row, col, ch = image.shape
        mean = ui.spinBox.value()
        sigma = ui.spinBox_3.value()
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        cv2.imwrite("noisy_img.png", noisy)
        pixmap = QPixmap("noisy_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def show_pepper(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(2)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def add_pepper_noise(self, label,ui):
        image = cv2.imread(self.file_path)
        # 设置添加椒盐噪声的数目比例
        s_vs_p = ui.doubleSpinBox.value()
        # 设置添加噪声图像像素的数目
        amount = ui.doubleSpinBox_3.value()
        noisy_img = np.copy(image)
        # 添加salt噪声
        num_salt = np.ceil(amount * image.size * s_vs_p)
        # 设置添加噪声的坐标位置
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = [255, 255, 255]
        # 添加pepper噪声
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        # 设置添加噪声的坐标位置
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = [0, 0, 0]
        cv2.imwrite("noisy_img.png", noisy_img)
        pixmap = QPixmap("noisy_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def add_bs_noise(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # Convert QPixmap to NumPy array
            image_np = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            width, height = image_np.width(), image_np.height()
            ptr = image_np.bits()
            ptr.setsize(height * width * 3)
            image_opencv = np.array(ptr).reshape(height, width, 3).copy()

            # 计算图像像素的分布范围
            vals = len(np.unique(image_opencv))
            vals = 2 ** np.ceil(np.log2(vals))
            # 给图片添加泊松噪声
            noisy_img = np.random.poisson(image_opencv * vals) / float(vals)

            image = noisy_img[:, :, ::-1]  # 解决处理后颜色发生偏差

            cv2.imwrite("noisy_img.png", image)

            pixmap = QPixmap("noisy_img.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")

            # 检查用户的选择
            response = msg_box.exec_()

            print("No image found in QLabel.")


    def add_speckle_noise(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # Convert QPixmap to NumPy array
            image_np = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            width, height = image_np.width(), image_np.height()
            ptr = image_np.bits()
            ptr.setsize(height * width * 3)
            image_opencv = np.array(ptr).reshape(height, width, 3).copy()

            row, col, ch = image_opencv.shape
            # 随机生成一个服从分布的噪声
            gauss = np.random.randn(row, col, ch)
            # 给图片添加speckle噪声
            noisy_img = image_opencv + image_opencv * gauss
            # 归一化图像的像素值
            noisy_img = np.clip(noisy_img, a_min=0, a_max=255)

            image = noisy_img[:, :, ::-1]  # 解决处理后颜色发生偏差

            cv2.imwrite("noisy_img.png", image)

            pixmap = QPixmap("noisy_img.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")

            # 检查用户的选择
            response = msg_box.exec_()

            print("No image found in QLabel.")


    def show_average_noise(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(4)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def denoise_average(self, label,ui):
        image = cv2.imread(self.file_path)
        result = cv2.blur(image, (ui.spinBox_4.value(), ui.spinBox_5.value()))
        cv2.imwrite("denoise_img.png", result)
        pixmap = QPixmap("denoise_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_median_noise(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(3)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def denoise_median(self, label,ui):
        image = cv2.imread(self.file_path)
        result = cv2.medianBlur(image, ui.spinBox_6.value())
        cv2.imwrite("denoise_img.png", result)
        pixmap = QPixmap("denoise_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def show_gaussian_noise(self, ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(5)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def denoise_gaussian(self, label, ui):
        image = cv2.imread(self.file_path)
        result = cv2.GaussianBlur(image, (ui.spinBox_7.value(), ui.spinBox_8.value()), ui.spinBox_9.value())
        cv2.imwrite("denoise_img.png", result)
        pixmap = QPixmap("denoise_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_bilateral_noise(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(6)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def denoise_bilateral(self, label,ui):
        image = cv2.imread(self.file_path)
        result = cv2.bilateralFilter(image, ui.spinBox_12.value(), ui.spinBox_10.value(), ui.spinBox_11.value())
        cv2.imwrite("denoise_img.png", result)
        pixmap = QPixmap("denoise_img.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def zhijiepinjie0(self,label):
        file_dialog = QFileDialog()
        img1_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "",
                                                        "Image Files (*.png *.jpg *.bmp)")
        img2_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "",
                                                        "Image Files (*.png *.jpg *.bmp)")
        stitched_image = zhijiepinjie.match_images(img1_path, img2_path)
        cv2.imshow('Stitched Image', stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("stitched_image.png", stitched_image)
        self.file_path = "stitched_image.png"
        pixmap = QPixmap("stitched_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def jiaquanpinjie0(self, label):
        file_dialog = QFileDialog()
        img1_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "",
                                                        "Image Files (*.png *.jpg *.bmp)")
        img2_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "",
                                                        "Image Files (*.png *.jpg *.bmp)")
        stitched_image = jiaquanpinjie.image_pinjie(img1_path, img2_path)
        cv2.imshow('Stitched Image', stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("stitched_image.png", stitched_image)
        self.file_path = "stitched_image.png"
        pixmap = QPixmap("stitched_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示




    def histogram_equalization(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization
            equalized_image = cv2.equalizeHist(gray_image)
            # Convert grayscale back to RGB
            equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("equalization_img.png", equalized_image_rgb)
            pixmap = QPixmap("equalization_img.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def increase_image(self, brightness_slider,contrast_slider,saturation_slider,sharpness_slider, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            brightness_value = brightness_slider.value()
            contrast_value = contrast_slider.value()
            saturation_value = saturation_slider.value()
            sharpness_value = sharpness_slider.value()
            adjusted_image = self.adjust_imgae(brightness_value, contrast_value, saturation_value, sharpness_value,
                                               image)
            cv2.imwrite("adjusted_image.png", adjusted_image)
            pixmap = QPixmap("adjusted_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def adjust_imgae(self,brightness,contrast,saturation,sharpness,image):
        image1 = cv2.addWeighted(image, 1, image, 0, brightness)
        image2 = cv2.convertScaleAbs(image1, alpha=1 + contrast / 100)
        image3 = cv2.GaussianBlur(image2, (sharpness * 2 + 1, sharpness * 2 + 1), 0)

        # 将图像从 BGR 色彩空间转换为 HSV 色彩空间
        hsv_image = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)
        # 调整饱和度
        hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], saturation)
        # 将图像从 HSV 色彩空间转换回 BGR 色彩空间
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return adjusted_image


    def pintu(self, label):
        file_dialog = QFileDialog()
        file_path1, _ = file_dialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        file_dialog = QFileDialog()
        file_path2, _ = file_dialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        file_dialog = QFileDialog()
        file_path3, _ = file_dialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        # 加载图像
        image1 = cv2.imread(file_path1)
        image1_resized = cv2.resize(image1, (600, 200))
        image2 = cv2.imread(file_path2)
        image2_resized = cv2.resize(image2, (600, 200))
        image3 = cv2.imread(file_path3)
        image3_resized = cv2.resize(image3, (600, 200))
        # 拼接图片，垂直拼接
        merged_image_vertical = np.vstack((image1_resized, image2_resized, image3_resized))
        # 保存拼接后的图像
        cv2.imwrite("pintu.jpg", merged_image_vertical)
        pixmap = QPixmap("pintu.jpg")
        label.setPixmap(pixmap)
        # label.setScaledContents(True)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_erode(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(8)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def erode_image(self,label,ui):
        image = cv2.imread(self.file_path)
        # 执行腐蚀变换
        kernel = np.ones((ui.spinBox_2.value(), ui.spinBox_2.value()), np.uint8)
        # kernel = ui.spinBox_2.value()
        eroded_image = cv2.erode(image, kernel, iterations=1)
        cv2.imwrite("eroded_image.png", eroded_image)
        pixmap = QPixmap("eroded_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示




    def show_dilate(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(9)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def dilate_image(self,label,ui):
        image = cv2.imread(self.file_path)
        # 执行腐蚀变换
        kernel = np.ones((ui.spinBox_13.value(), ui.spinBox_13.value()), np.uint8)
        # kernel = ui.spinBox_13.value()
        dilate_image = cv2.dilate(image, kernel, iterations=1)
        cv2.imwrite("dilate_image.png", dilate_image)
        pixmap = QPixmap("dilate_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def show_opened(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(10)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def opened_image(self,label,ui):
        image = cv2.imread(self.file_path)
        # 执行腐蚀变换
        kernel = np.ones((ui.spinBox_14.value(), ui.spinBox_14.value()), np.uint8)
        #kernel = ui.spinBox_14.value()
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("opened_image.png", opened_image)
        pixmap = QPixmap("opened_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_closed(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(11)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def closed_image(self,label,ui):
        image = cv2.imread(self.file_path)
        # 执行闭操作
        kernel = np.ones((ui.spinBox_15.value(), ui.spinBox_15.value()), np.uint8)
        # kernel = ui.spinBox_15.value()
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("closed_image.png", closed_image)
        pixmap = QPixmap("closed_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_gradient(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(12)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def gradient_image(self,label,ui):
        image = cv2.imread(self.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 执行梯度操作
        kernel = np.ones((ui.spinBox_16.value(), ui.spinBox_16.value()), np.uint8)
        # kernel = ui.spinBox_16.value()
        gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite("closed_image.png", gradient_image)
        pixmap = QPixmap("closed_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_tophat(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(13)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def tophat_image(self,label,ui):
        image = cv2.imread(self.file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用一个小型矩形核进行顶帽操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ui.spinBox_17.value(), ui.spinBox_17.value()))
        # kernel = ui.spinBox_17.value()
        top_hat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)

        cv2.imwrite("tophat_image.png", top_hat)
        pixmap = QPixmap("tophat_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_blackhat(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(14)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def blackhat_image(self,label,ui):
        image = cv2.imread(self.file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用一个小型矩形核进行顶帽操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ui.spinBox_18.value(), ui.spinBox_18.value()))
        # kernel = ui.spinBox_18.value()
        black_hat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

        cv2.imwrite("black_hat_image.png", black_hat)
        pixmap = QPixmap("black_hat_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示

    def wei_cai_se(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            src = cv2.imread(self.file_path)
            cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("input", src)
            dst = cv2.applyColorMap(src, cv2.COLORMAP_COOL)
            cv2.imshow("output", dst)

            # 伪色彩
            image = cv2.imread(self.file_path)
            color_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            cv2.imshow("image", image)
            cv2.imshow("color_image", color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite("color_image.png", color_image)
            pixmap = QPixmap("color_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示

            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def edge_dectect(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            edges = cv2.Canny(image_rgb, 50, 150)
            cv2.imwrite("edge_image.png", edges)
            pixmap = QPixmap("edge_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def gontour_dectect(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(image_gray, 175, 200, 0)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            dst = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
            cv2.imwrite("gontour_image.png", dst)
            pixmap = QPixmap("gontour_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")



    def LSD_line_dectect(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lsd = cv2.createLineSegmentDetector(0)

            dlines = lsd.detect(image_gray)
            for dline in dlines[0]:
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                cv2.line(image,(x0,y0),(x1,y1),(0,255,0),1,cv2.LINE_AA)
            cv2.imwrite("line_image.png", image)
            pixmap = QPixmap("line_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")




    def Circles_dectect(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 124, param1=100, param2=60, minRadius=50,
                                       maxRadius=200)
            arr1 = np.zeros([0, 2], dtype=int)  # 创建一个0行, 2列的空数组
            if circles is not None:
                circles = np.uint16(np.around(circles))  # 4舍5入, 然后转为uint16
                for i in circles[0, :]:
                    arr1 = np.append(arr1, (i[0], i[1]))  # arr1是圆心坐标的np数组
                    # print(arr1)
                    cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 3)  # 轮廓
                    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 0), 6)  # 圆心

            cv2.imwrite("circle_image.png", image)
            pixmap = QPixmap("circle_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def image_add1(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("circle_image.png", image)
            pixmap = QPixmap("circle_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")




    def show_edge_dectect2(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(15)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def edge_dectect2(self,label,ui):
        image = cv2.imread(self.file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(image_rgb, ui.spinBox_20.value(), ui.spinBox_21.value())
        cv2.imwrite("edge_image.png", edges)
        pixmap = QPixmap("edge_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示


    def show_threshold(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(16)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    def threshold(self,label,ui):
        image = cv2.imread(self.file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, ui.spinBox_22.value(), ui.spinBox_23.value(), cv2.THRESH_BINARY)
        cv2.imwrite("threshold_image.png", threshold_image)
        pixmap = QPixmap("threshold_image.png")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def region_growing(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            region = region_grow.Region_Grow(img ,self.file_path)
            cv2.imwrite("region_growing_image.png", region)
            pixmap = QPixmap("region_growing_image.png")
            ui.label.setPixmap(pixmap)
            ui.label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")



    def border_img(self,label,i):
        pixmap = label.pixmap()
        if pixmap is not None:
            base_image = cv2.imread(self.file_path)
            if i == 1:
                border_image = cv2.imread("biankuang1.png")
            elif i == 2:
                border_image = cv2.imread("biankuang2.png")
            elif i == 3:
                border_image = cv2.imread("biankuang3.png")
            elif i == 4:
                border_image = cv2.imread("biankuang4.png")
            # 获取背景图像的尺寸
            height, width, _ = base_image.shape
            # 将logo调整为与背景图像相同大小
            resized_logo = cv2.resize(border_image, (width, height))
            rows, cols, channels = resized_logo.shape  # 获取图像2的属性
            roi = base_image[0:rows, 0:cols]  # 选择roi范围
            logo2gray = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
            ret, mask = cv2.threshold(logo2gray, 200, 255, cv2.THRESH_BINARY)  # 设置阈值，大于175的置为255，小于175的置为0    logo黑色
            mask_inv = cv2.bitwise_not(mask)  # 非运算  logo白色
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask)  # 删除了ROI中的logo区域(mask的logo为黑色，故and后该区域被舍去)---》上图中的图一
            logo_fg = cv2.bitwise_and(resized_logo, resized_logo,
                                      mask=mask_inv)  # 删除了logo中的空白区域(mask_inv的logo为百度色，故and后该区域被保留)----》上图中的图二
            dst = cv2.add(img1_bg, logo_fg)  # 两个像插销一样，一个被镂空，一个被保留，两者相加，刚刚好
            base_image[0:rows, 0:cols] = dst  # 将贴图后的区域图，覆盖到原图
            cv2.imwrite("border_image.png", base_image)
            pixmap = QPixmap("border_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    # 浮雕效果
    def apply_emboss_effect(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[0:2]
            # 定义空白图像，存放图像浮雕处理之后的图片
            img1 = np.zeros((h, w), dtype=gray.dtype)
            # 通过对原始图像进行遍历，通过浮雕公式修改像素值，然后进行浮雕处理
            for i in range(h):
                for j in range(w - 1):
                    # 前一个像素值
                    a = gray[i, j]
                    # 后一个像素值
                    b = gray[i, j + 1]
                    # 新的像素值,防止像素溢出
                    img1[i, j] = min(max((int(a) - int(b) + 128), 0), 255)
            cv2.imwrite("emboss_image.png", img1)
            pixmap = QPixmap("emboss_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def tutoujing(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            rows, cols, c = img.shape
            center_x, center_y = rows / 2, cols / 2
            radius = math.sqrt(rows ** 2 + cols ** 2) / 2
            new_img = np.zeros_like(img)  # 创建一个与原始图像相同大小的新图像，用于存储结果
            for i in range(rows):
                for j in range(cols):
                    dis = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    if dis <= radius:
                        new_i = int(np.round(center_x + (i - center_x) / radius * dis))
                        new_j = int(np.round(center_y + (j - center_y) / radius * dis))
                        new_img[i, j] = img[new_i, new_j]
            cv2.imwrite("tutoujing.png", new_img)
            pixmap = QPixmap("tutoujing.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def sumiao(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            src_image_name = self.file_path
            dst_image_name = 'sketch_example.jpg'
            sumiao.rgb_to_sketch(src_image_name, dst_image_name)
            pixmap = QPixmap("sketch_example.jpg")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def maoboli(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            src = cv2.imread(self.file_path)
            dst = np.zeros_like(src)
            rows, cols, _ = src.shape
            offsets = 5
            for y in range(rows - offsets):
                for x in range(cols - offsets):
                    random_num = np.random.randint(0, offsets)
                    dst[y, x] = src[y + random_num, x + random_num]
            cv2.imshow('src', src)
            cv2.imshow('dst', dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("maoboli.png", dst)
            pixmap = QPixmap("maoboli.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def huaijiu(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            # 获取图像属性
            h, w = img.shape[0:2]
            # 定义空白图像，存放图像怀旧处理之后的图片
            img1 = np.zeros((h, w, 3), dtype=img.dtype)
            # 通过对原始图像进行遍历，通过怀旧公式修改像素值，然后进行怀旧处理
            for i in range(h):
                for j in range(w):
                    B = 0.131 * img[i, j, 0] + 0.534 * img[i, j, 1] + 0.272 * img[i, j, 2]
                    G = 0.168 * img[i, j, 0] + 0.686 * img[i, j, 1] + 0.349 * img[i, j, 2]
                    R = 0.189 * img[i, j, 0] + 0.769 * img[i, j, 1] + 0.393 * img[i, j, 2]
                    # 防止图像溢出
                    if B > 255:
                        B = 255
                    if G > 255:
                        G = 255
                    if R > 255:
                        R = 255
                    img1[i, j] = [int(B), int(G), int(R)]  # B\G\R三通道都设置为怀旧值
            cv2.imwrite("huanjiu.png", img1)
            pixmap = QPixmap("huanjiu.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def cartoon(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            cartoonized_image = cartoon.cartoonize_single_image(self.file_path)
            cv2.imwrite("cartoonized_image.png", cartoonized_image)
            pixmap = QPixmap("cartoonized_image.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")



    def liunian(self, label, weight):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            # 获取图像属性
            h, w = img.shape[0:2]
            # 定义空白图像，存放图像流年处理之后的图片
            img1 = np.zeros((h, w, 3), dtype=img.dtype)
            # 通过对原始图像进行遍历，通过流年公式修改B通道的像素值，然后进行流连处理
            for i in range(h):
                for j in range(w):
                    B = int(np.sqrt(img[i, j, 0]) * weight)  # 对通道B的像素值开方，然后乘以权重
                    if B > 255:
                        B = 255
                    img1[i, j] = [B, img[i, j, 1], img[i, j, 2]]
            cv2.imwrite("liunian.png", img1)
            pixmap = QPixmap("liunian.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def youqi(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            # 对原图像进行扩充，处理黑边
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            # 获取图像属性
            h, w = img.shape[0:2]
            # 定义空白图像，存放图像油漆处理之后的图片
            img1 = np.zeros((h, w, 3), dtype=img.dtype)
            # 定义卷积模板
            kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
            # 通过对原始图像进行遍历，通过油漆公式修改像素值，然后进行油漆处理
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    # 周围邻域与模板进行卷积并防止像素溢出
                    B = min(max(0, np.sum(img[i - 1:i + 1 + 1, j - 1:j + 1 + 1, 0] * kernel)), 255)
                    G = min(max(0, np.sum(img[i - 1:i + 1 + 1, j - 1:j + 1 + 1, 1] * kernel)), 255)
                    R = min(max(0, np.sum(img[i - 1:i + 1 + 1, j - 1:j + 1 + 1, 2] * kernel)), 255)
                    img1[i, j] = [B, G, R]
            # img1 = img1[(0 + 1):(h - 1), (0 + 1):(h - 1)]  # 裁剪恢复原图
            cv2.imwrite("liunian.png", img1)
            pixmap = QPixmap("liunian.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")



    def guangzhao(self, label, lightIntensity):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            # 获取图像属性
            h, w = img.shape[0:2]
            # 定义空白图像，存放图像光照处理之后的图片
            img1 = np.zeros((h, w, 3), dtype=img.dtype)
            # 确定中心点的位置
            x, y = int(h / 2), int(w / 2)
            # 确定半径
            r = min(x, y)
            # 通过对原始图像进行遍历，通过光照公式修改像素值，然后进行光照处理
            for i in range(h):
                for j in range(w):
                    # 计算像素点i，j到中心点的距离的平方
                    distance = (x - i) ** 2 + (y - j) ** 2
                    # 比较距离与半径的大小，当距离大于半径，不做处理，当距离小于等于半径，设置为光照强度值
                    if distance > r ** 2:
                        img1[i, j] = img[i, j]
                    else:
                        result = int(lightIntensity * (1.0 - np.sqrt(distance) / r))  # 通过距离来计算光照强度值
                        # 光照特效处理后的图像三通道值与255取最小，防止溢出(0和结果中选择最大的，结果和255中选择最小的)
                        B = min(max(0, img[i, j, 0] + result), 255)
                        G = min(max(0, img[i, j, 1] + result), 255)
                        R = min(max(0, img[i, j, 2] + result), 255)
                        img1[i, j] = [B, G, R]  # B\G\R三通道都设置为光照强度值
            cv2.imwrite("liunian.png", img1)
            pixmap = QPixmap("liunian.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")





    def shuicai(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            result = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
            cv2.imwrite("shuicai.png", result)
            pixmap = QPixmap("shuicai.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def youhua(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            img = cv2.imread(self.file_path)
            res = cv2.xphoto.oilPainting(img, 7, 1)
            cv2.imwrite("youhua.png", res)
            pixmap = QPixmap("youhua.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def caiqian(self, label):
        pixmap = label.pixmap()
        if pixmap is not None:
            src = cv2.imread(self.file_path)
            dst_gray, dst_color = cv2.pencilSketch(src, sigma_s=5, sigma_r=0.1, shade_factor=0.03)
            cv2.imshow('gray_pencil', dst_gray)
            cv2.imshow('color_pencil', dst_color)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imwrite("caiqian.png", dst_color)
            pixmap = QPixmap("caiqian.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def oldphoto(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # 读取图像
            image = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
            # 将图像从 BGR 转换为 LAB
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            # 将 LAB 图像拆分为单独的通道
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            # 对每个通道应用自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            a_channel = clahe.apply(a_channel)
            b_channel = clahe.apply(b_channel)
            # 将处理后的通道重新组合为 LAB 图像
            lab_image = cv2.merge((l_channel, a_channel, b_channel))
            # 将图像从 LAB 转换回 BGR
            result_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
            cv2.imwrite("oldphoto.png",result_image)
            pixmap = QPixmap("oldphoto.png")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    def show_supper(self,ui):
        pixmap = ui.label.pixmap()
        if pixmap is not None:
            ui.stackedWidget.setCurrentIndex(7)
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")
    # 超分辨率
    def supper(self,label,ui):
        # 使用OpenCV加载图像
        image = cv2.imread(self.file_path)
        # 将图像进行超分辨率处理
        # 这里使用双立方插值法将图像放大两倍
        upscale_factor = ui.doubleSpinBox_2.value()
        super_res_image = cv2.resize(
            image, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC
        )
        # 将结果保存并显示
        cv2.imwrite("high_res_image.jpg", super_res_image)
        pixmap = QPixmap("high_res_image.jpg")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # 图像居中显示



    def beautify(self,mopi_slider,r_slider,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            image = cv2.imread(self.file_path)
            mopi_value = mopi_slider.value()
            r = r_slider.value()
            image_dst = self.mopi(image,mopi_value)
            image_dst = self.big_eye(image_dst, r, 1)
            cv2.imwrite("swite_skin_image.jpg", image_dst)
            pixmap = QPixmap("swite_skin_image.jpg")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")

    def YCrCb_ellipse_model(self,img):
        skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)  # 绘制椭圆弧线
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
        (Y, Cr, Cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
        skin = np.zeros(Cr.shape, dtype=np.uint8)  # 掩膜
        (x, y) = Cr.shape
        for i in range(0, x):
            for j in range(0, y):
                if skinCrCbHist[Cr[i][j], Cb[i][j]] > 0:  # 若不在椭圆区间中
                    skin[i][j] = 255
        res = cv2.bitwise_and(img, img, mask=skin)
        return skin, res

    def guided_filter(self,I, p, win_size, eps):
        assert I.any() <= 1 and p.any() <= 1
        mean_I = cv2.blur(I, (win_size, win_size))
        mean_p = cv2.blur(p, (win_size, win_size))

        corr_I = cv2.blur(I * I, (win_size, win_size))
        corr_Ip = cv2.blur(I * p, (win_size, win_size))

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.blur(a, (win_size, win_size))
        mean_b = cv2.blur(b, (win_size, win_size))

        q = mean_a * I + mean_b

        return q
    def mopi(self, img ,value):
        skin, _ = self.YCrCb_ellipse_model(img)  # 获得皮肤的掩膜数组
        # 进行一次开运算
        kernel = np.ones((value, value), dtype=np.uint8)
        skin = cv2.erode(skin, kernel=kernel)
        skin = cv2.dilate(skin, kernel=kernel)
        img1 = self.guided_filter(img / 255.0, img / 255.0, 10, eps=0.001) * 255
        img1 = np.array(img1, dtype=np.uint8)
        img1 = cv2.bitwise_and(img1, img1, mask=skin)  # 将皮肤与背景分离
        skin = cv2.bitwise_not(skin)
        img1 = cv2.add(img1, cv2.bitwise_and(img, img, mask=skin))  # 磨皮后的结果与背景叠加

        return img1


    def get_face_key_point(self, img):
        # 初始化人脸检测器和关键点检测器
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 替换为你的关键点检测器模型路径

        # 将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用人脸检测器检测图像中的人脸
        faces = detector(gray)

        # 如果未检测到人脸，则返回None
        if len(faces) == 0:
            return None

        # 获取第一个检测到的人脸的关键点
        face_landmarks = predictor(gray, faces[0])
        left_eye_pos = ((face_landmarks.part(36).x + face_landmarks.part(39).x) // 2,
                           (face_landmarks.part(36).y + face_landmarks.part(39).y) // 2)
        right_eye_pos = ((face_landmarks.part(42).x + face_landmarks.part(45).x) // 2,
                            (face_landmarks.part(42).y + face_landmarks.part(45).y) // 2)

        return left_eye_pos, right_eye_pos
    def bilinear_interpolation(self, img, vector_u, c):
        ux, uy = vector_u
        x1, x2 = int(ux), int(ux + 1)
        y1, y2 = int(uy), int(uy + 1)

        # f_x_y1 = (x2-ux)/(x2-x1)*img[x1][y1]+(ux-x1)/(x2-x1)*img[x2][y1]
        # f_x_y2 = (x2 - ux) / (x2 - x1) * img[x1][y2] + (ux - x1) / (x2 - x1) * img[x2][y2]

        f_x_y1 = (x2 - ux) / (x2 - x1) * img[y1][x1][c] + (ux - x1) / (x2 - x1) * img[y1][x2][c]
        f_x_y2 = (x2 - ux) / (x2 - x1) * img[y2][x1][c] + (ux - x1) / (x2 - x1) * img[y2][x2][c]

        f_x_y = (y2 - uy) / (y2 - y1) * f_x_y1 + (uy - y1) / (y2 - y1) * f_x_y2
        return int(f_x_y)
    def Local_scaling_warps(self, img, cx, cy, r_max, a):
        img1 = np.copy(img)
        for y in range(cy - r_max, cy + r_max + 1):
            d = int(math.sqrt(r_max ** 2 - (y - cy) ** 2))
            x0 = cx - d
            x1 = cx + d
            for x in range(x0, x1 + 1):
                r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)  # 求出当前位置的半径
                for c in range(3):
                    vector_c = np.array([cx, cy])
                    vector_r = np.array([x, y]) - vector_c
                    f_s = (1 - ((r / r_max - 1) ** 2) * a)
                    vector_u = vector_c + f_s * vector_r  # 原坐标
                    img1[y][x][c] = self.bilinear_interpolation(img, vector_u, c)
        return img1
    def big_eye(self, img, r_max, a, left_eye_pos=None, right_eye_pos=None):
        img0 = img.copy()
        if left_eye_pos == None or right_eye_pos == None:
            left_eye_pos, right_eye_pos = self.get_face_key_point(img)
        img0 = cv2.circle(img0, left_eye_pos, radius=10, color=(0, 0, 255))
        img0 = cv2.circle(img0, right_eye_pos, radius=10, color=(0, 0, 255))
        img = self.Local_scaling_warps(img, left_eye_pos[0], left_eye_pos[1], r_max=r_max, a=a)
        img = self.Local_scaling_warps(img, right_eye_pos[0], right_eye_pos[1], r_max=r_max, a=a)
        return img

    # '''
    # 方法： Interactive Image Warping 局部平移算法
    # '''
    #
    # # startX, startY,left_landmark[0, 0], left_landmark[0, 1]
    # # endPt[0, 0], endPt[0, 1]
    # # r_left,r_right为半径
    # def localTranslationWarp(self, srcImg, startX, startY, endX, endY, radius):
    #     ddradius = float(radius * radius)
    #     copyImg = np.zeros(srcImg.shape, np.uint8)
    #     copyImg = srcImg.copy()
    #
    #     # 计算公式中的|m-c|^2
    #     ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    #     H, W, C = srcImg.shape
    #     for i in range(W):
    #         for j in range(H):
    #             # 计算该点是否在形变圆的范围之内
    #             # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
    #             if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
    #                 continue
    #
    #             distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
    #
    #             if (distance < ddradius):
    #                 # 计算出（i,j）坐标的原坐标
    #                 # 计算公式中右边平方号里的部分
    #                 ratio = (ddradius - distance) / (ddradius - distance + ddmc)
    #                 ratio = ratio * ratio
    #
    #                 # 映射原位置
    #                 UX = i - ratio * (endX - startX)
    #                 UY = j - ratio * (endY - startY)
    #
    #                 # 根据双线性插值法得到UX，UY的值
    #                 value = self.BilinearInsert(srcImg, UX, UY)
    #                 # 改变当前 i ，j的值
    #                 copyImg[j, i] = value
    #
    #     return copyImg
    #
    # def landmark_dec_dlib_fun(self, img_src,detector,predictor):
    #     # 转灰度图
    #     img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('grayimg', img_gray)
    #     land_marks = []
    #
    #     rects = detector(img_gray, 0)
    #     print(rects)
    #     for i in range(len(rects)):
    #         land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
    #         for idx, point in enumerate(land_marks_node):
    #             # 68点坐标
    #             pos = (point[0, 0], point[0, 1])
    #             print(idx, pos)
    #             # 利用cv2.circle给每个特征点画一个圈，共68个
    #             cv2.circle(img_src, pos, 5, color=(0, 255, 0))
    #             # 利用cv2.putText输出1-68
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             cv2.putText(img_src, str(idx + 1), pos, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #         land_marks.append(land_marks_node)
    #
    #     return land_marks
    #
    # def BilinearInsert(self, src, ux, uy):
    #     w, h, c = src.shape
    #     if c == 3:
    #         x1 = int(ux)
    #         x2 = x1 + 1
    #         y1 = int(uy)
    #         y2 = y1 + 1
    #
    #         part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
    #         part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
    #         part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
    #         part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))
    #
    #         insertValue = part1 + part2 + part3 + part4
    #
    #         return insertValue.astype(np.int8)
    #
    # def face_thin_auto(self, src, detector, predictor):
    #     # 68个关键点二维数组
    #     landmarks = self.landmark_dec_dlib_fun(src, detector, predictor)
    #
    #     # 如果未检测到人脸关键点，就不进行瘦脸
    #     if len(landmarks) == 0:
    #         return
    #
    #     for landmarks_node in landmarks:
    #         # 第4个点左
    #         left_landmark = landmarks_node[3]
    #         # 第6个点左
    #         left_landmark_down = landmarks_node[5]
    #         # 第14个点右
    #         right_landmark = landmarks_node[13]
    #         # 第16个点右
    #         right_landmark_down = landmarks_node[15]
    #         # 第31个点鼻尖
    #         endPt = landmarks_node[30]
    #
    #         # 计算第4个点到第6个点的距离作为瘦脸距离
    #         r_left = math.sqrt(
    #             (left_landmark[0, 0] - left_landmark_down[0, 0]) * (
    #                     left_landmark[0, 0] - left_landmark_down[0, 0]) +
    #             (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))
    #
    #         # 计算第14个点到第16个点的距离作为瘦脸距离
    #         r_right = math.sqrt(
    #             (right_landmark[0, 0] - right_landmark_down[0, 0]) * (
    #                         right_landmark[0, 0] - right_landmark_down[0, 0]) +
    #             (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))
    #
    #         # 瘦左边脸
    #         thin_image = self.localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
    #                                           r_left)
    #         # 瘦右边脸
    #         thin_image = self.localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
    #                                           endPt[0, 1], r_right)
    #         return thin_image
    #
    # def shoulian(self, label):
    #     image = cv2.imread("swite_skin_image.jpg")
    #     detector = dlib.get_frontal_face_detector()
    #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #     image_dst = self.face_thin_auto(image, detector, predictor)
    #     cv2.imwrite("swite_skin_image.jpg", image_dst)
    #     pixmap = QPixmap("swite_skin_image.jpg")
    #     label.setPixmap(pixmap)
    #     label.setAlignment(Qt.AlignCenter)  # 图像居中显示

    def tp_FaceTec(self,label):
        pixmap = label.pixmap()
        if pixmap is not None:
            # 加载人脸检测的分类器
            face_cascade = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            # 读取图像
            image = cv2.imread(self.file_path)
            # 转换为灰度图像
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 使用分类器进行人脸检测
            faces = face_cascade(gray_image)
            # 在检测到的人脸上绘制矩形框
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 使用68点关键点检测器检测关键点
                landmarks = predictor(gray_image, face)

                # 遍历每个关键点，并在图像上绘制
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            # 将结果保存并显示
            cv2.imwrite("tp_FaceTec_image.jpg", image)
            pixmap = QPixmap("tp_FaceTec_image.jpg")
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # 图像居中显示
            print("Successfully read image from QLabel.")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有图片！")
            # 检查用户的选择
            response = msg_box.exec_()
            print("No image found in QLabel.")


    # 人脸识别（视频版）
    def sp_FaceTec(self):
        # 加载人脸检测器
        face_cascade = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 打开摄像头
        cap = cv2.VideoCapture(0)

        # 设置帧的宽度和高度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            # 读取摄像头的帧
            ret, frame = cap.read()
            # 将帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 在灰度图像中检测人脸
            faces = face_cascade(gray)
            # 标记检测到的人脸
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 使用68点关键点检测器检测关键点
                landmarks = predictor(gray, face)

                # 遍历每个关键点，并在图像上绘制
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 显示结果
            cv2.imshow('Face Detection', frame)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放摄像头资源
        cap.release()
        cv2.destroyAllWindows()

