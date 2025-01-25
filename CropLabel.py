import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage


class CropLabel(QLabel):
    def __init__(self, label , parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.point1 = None
        self.point2 = None
        self.rectangle = None
        self.pixmap = label.pixmap()


    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.point1 is None:
                self.point1 = event.pos()
            elif self.point2 is None:
                self.point2 = event.pos()
                self.rectangle = QRect(self.point1, self.point2)
                self.update()
                # 定义区域
                x1_ = self.get_point1_x()
                y1_ = self.get_point1_y()
                x2_ = self.get_point2_x()
                y2_ = self.get_point2_y()
                self.point1 = None
                self.point2 = None
                image = self.pixmap.toImage()
                # 裁剪图片
                cropped_image = image.copy(QRect(x1_, y1_, x2_ - x1_, y2_ - y1_))
                # 转换为 RGB 格式
                cropped_image_rgb = cropped_image.convertToFormat(QImage.Format_RGB888)
                # 保存裁剪后的图片
                cropped_image_rgb.save("cropped_image.jpg", format="JPEG")
                print("success")


    def paintEvent(self, event):
        super().paintEvent(event)
        if self.point1:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
            painter.drawPoint(self.point1)
        if self.point2:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
            painter.drawPoint(self.point2)
        if self.rectangle:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(self.rectangle)

    def get_point1_x(self):
        if self.point1:
            return self.point1.x()
        else:
            return None

    def get_point1_y(self):
        if self.point1:
            return self.point1.y()
        else:
            return None

    def get_point2_x(self):
        if self.point2:
            return self.point2.x()
        else:
            return None

    def get_point2_y(self):
        if self.point2:
            return self.point2.y()
        else:
            return None
