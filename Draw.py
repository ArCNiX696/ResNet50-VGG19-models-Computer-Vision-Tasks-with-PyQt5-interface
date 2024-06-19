from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QFileDialog
import numpy as np
import cv2


class InteractiveCanvas(QtWidgets.QGraphicsView): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))# Assing a QGraphicsScene in InteractiveCanvas
        self.canvas = QtGui.QPixmap(361,281)  
        self.canvas.fill(QtCore.Qt.black)  # black background
        self.scene().addPixmap(self.canvas) #add the canvas to the scene
        self.drawing = False #Control when is drawing
        self.last_point = QtCore.QPoint()
        self.pen_color = QtCore.Qt.white  # pen color
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)#desactivate scroll bars
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)#desactivate scroll bars

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton: #activated when mouse left is clicked 
            self.drawing = True #activates draw
            self.last_point = event.pos() #Obtain the coors to start drawing

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.drawing:#Condition
            painter = QtGui.QPainter(self.canvas) #Create a painter object to paint
            painter.setPen(QtGui.QPen(self.pen_color, 18, QtCore.Qt.SolidLine))#Pen color,width,style
            painter.drawLine(self.last_point, event.pos()) #Draw the line till last state
            painter.end()
            self.last_point = event.pos()
            self.scene().clear() 
            self.scene().addPixmap(self.canvas) #Update and visualize what is drawn

    def mouseReleaseEvent(self, event):#Verifies if left button is released
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            self.saveCanvas()

    def saveCanvas(self):
        file_path = './Datasets/VGG19_inference/Created.jpg'
        self.canvas.save(file_path)

    def displayImage(self):
        image_path='./Graphics/VGG19/Acc&Loss.png'

        # Load the image using OpenCV
        cv_image = cv2.imread(image_path)

        # Resize the image to the desired size
        resized_image = cv2.resize(cv_image, (361, 281))

        # Convert from BGR (OpenCV) to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, _ = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Convert QImage to QPixmap and display it
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.canvas = pixmap
        self.scene().clear()
        self.scene().addPixmap(self.canvas)
        self.update()  # Update the widget

    def clearCanvas(self):
        self.canvas.fill(QtCore.Qt.black) #reset the windows
        self.scene().clear()
        self.scene().addPixmap(self.canvas)


    