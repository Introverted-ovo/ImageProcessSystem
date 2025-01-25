from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QLabel, QApplication


class MyLabel(QLabel):
    SCALE_MIN_VALUE = 0.1
    SCALE_MAX_VALUE = 10.0

    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setPixmap(QPixmap("image.jpg"))
        self.m_scaleValue = 1.0
        self.m_rectPixmap = QRectF(self.pixmap().rect())
        self.m_drawPoint = QPointF(0.0, 0.0)
        self.m_pressed = False
        self.m_lastPos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_pressed = True
            self.m_lastPos = event.pos()

    def mouseMoveEvent(self, event):
        if self.m_pressed:
            delta = event.pos() - self.m_lastPos
            self.m_lastPos = event.pos()
            self.m_drawPoint += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_pressed = False

    def wheelEvent(self, event):
        oldScale = self.m_scaleValue
        if event.angleDelta().y() > 0:
            self.m_scaleValue *= 1.1
        else:
            self.m_scaleValue *= 0.9

        if self.m_scaleValue > self.SCALE_MAX_VALUE:
            self.m_scaleValue = self.SCALE_MAX_VALUE

        if self.m_scaleValue < self.SCALE_MIN_VALUE:
            self.m_scaleValue = self.SCALE_MIN_VALUE

        if self.m_rectPixmap.contains(event.pos()):
            x = self.m_drawPoint.x() - (event.pos().x() - self.m_drawPoint.x()) / self.m_rectPixmap.width() * (
                        self.width() * (self.m_scaleValue - oldScale))
            y = self.m_drawPoint.y() - (event.pos().y() - self.m_drawPoint.y()) / self.m_rectPixmap.height() * (
                        self.height() * (self.m_scaleValue - oldScale))
            self.m_drawPoint = QPointF(x, y)
        else:
            x = self.m_drawPoint.x() - (self.width() * (self.m_scaleValue - oldScale)) / 2
            y = self.m_drawPoint.y() - (self.height() * (self.m_scaleValue - oldScale)) / 2
            self.m_drawPoint = QPointF(x, y)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.scale(self.m_scaleValue, self.m_scaleValue)
        painter.drawPixmap(self.m_drawPoint, self.pixmap())

    def resizeEvent(self, event):
        super(MyLabel, self).resizeEvent(event)
        self.m_rectPixmap = QRectF(self.pixmap().rect())
        self.m_rectPixmap.setWidth(self.m_rectPixmap.width() * self.m_scaleValue)
        self.m_rectPixmap.setHeight(self.m_rectPixmap.height() * self.m_scaleValue)
        self.m_drawPoint = QPointF((self.width() - self.m_rectPixmap.width()) / 2,
                                   (self.height() - self.m_rectPixmap.height()) / 2)


