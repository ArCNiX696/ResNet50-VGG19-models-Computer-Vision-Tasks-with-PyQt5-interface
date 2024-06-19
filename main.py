from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
import cv2 as cv
import os
from Background_sub import *
from Optical_flow import *
from PCA_Tiger import *
from VGG19 import VGG19Tears
from Draw import InteractiveCanvas
import keyboard
from Resnet_show import ShowClasses
from ResNet import *
#Temporal
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image


class Ui_MainWindow(object):
    def __init__(self):
        self.Bgs=BackgroundSub()
        self.Opt=Optical()
        self.pca=PCATiger()
        self.canvas = InteractiveCanvas()
        self.Vgg19=VGG19Tears()
        self.show=ShowClasses()
        self.Resnet=ResNet50ScareClaw()
    
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1102, 872)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #Load Image button
        self.loadImg_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadImg_pushButton.setGeometry(QtCore.QRect(10, 310, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.loadImg_pushButton.setFont(font)
        self.loadImg_pushButton.setObjectName("pushButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(180, 110, 281, 141))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.Backg_pushButton = QtWidgets.QPushButton(self.groupBox)
        self.Backg_pushButton.setGeometry(QtCore.QRect(20, 50, 241, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Backg_pushButton.setFont(font)
        self.Backg_pushButton.setObjectName("pushButton_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(180, 300, 291, 201))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")

        #Optical reprocessing
        self.Opt_pre_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.Opt_pre_pushButton.setGeometry(QtCore.QRect(20, 40, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Opt_pre_pushButton.setFont(font)
        self.Opt_pre_pushButton.setObjectName("pushButton_4")

        #Optical Tracking
        self.Opt_trak_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.Opt_trak_pushButton.setGeometry(QtCore.QRect(20, 120, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Opt_trak_pushButton.setFont(font)
        self.Opt_trak_pushButton.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(180, 520, 291, 141))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")

        #Dimension Reduction button
        self.PCA_pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.PCA_pushButton.setGeometry(QtCore.QRect(20, 50, 241, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.PCA_pushButton.setFont(font)
        self.PCA_pushButton.setObjectName("pushButton_6")

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(480, 110, 611, 341))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")

        #Windows 1
        self.canvas.setParent(self.groupBox_4)
        self.canvas.setParent(self.groupBox_4)  # Establece el grupo correcto como padre
        self.canvas.setGeometry(QtCore.QRect(230, 40, 361, 281))  # Ajusta la geometría según sea necesario
        self.canvas.setObjectName("canvas")
        self.canvas.setStyleSheet("background-color: black;")
        #self.graphicsView_2 = QtWidgets.QGraphicsView(self.groupBox_4)
        #self.graphicsView_2.setGeometry(QtCore.QRect(230, 40, 361, 281))
        #self.graphicsView_2.setObjectName("graphicsView_2")
        #self.canvas.setObjectName("canvas")
        #self.graphicsView_2.setStyleSheet("background-color: black;")

        #4.1 Show Model structure
        self.Mdl_str_pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.Mdl_str_pushButton.setGeometry(QtCore.QRect(10, 40, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Mdl_str_pushButton.setFont(font)
        self.Mdl_str_pushButton.setObjectName("pushButton_7")

        #4.1 Predict
        self.Predict_pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.Predict_pushButton.setGeometry(QtCore.QRect(10, 200, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Predict_pushButton.setFont(font)
        self.Predict_pushButton.setObjectName("pushButton_8")
        
        #Acc&Loss display
        self.Graphic_pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.Graphic_pushButton.setGeometry(QtCore.QRect(10, 120, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Graphic_pushButton.setFont(font)
        self.Graphic_pushButton.setObjectName("pushButton_9")

        #Reset Window
        self.Reset_pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.Reset_pushButton.setGeometry(QtCore.QRect(10, 280, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Reset_pushButton.setFont(font)
        self.Reset_pushButton.setObjectName("pushButton_10")


        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(480, 460, 611, 361))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")

        #Windows 2
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_5)
        self.graphicsView.setGeometry(QtCore.QRect(220, 30, 371, 311))
        self.graphicsView.setObjectName("graphicsView")

        self.Load_img_pb_5 = QtWidgets.QPushButton(self.groupBox_5)
        self.Load_img_pb_5.setGeometry(QtCore.QRect(10, 30, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Load_img_pb_5.setFont(font)
        self.Load_img_pb_5.setObjectName("pushButton_11")

        #Show images
        self.Show_pushButton = QtWidgets.QPushButton(self.groupBox_5)
        self.Show_pushButton.setGeometry(QtCore.QRect(10, 100, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Show_pushButton.setFont(font)
        self.Show_pushButton.setObjectName("pushButton_12")

        #ResNet Model Structure
        self.Structure_pushButton = QtWidgets.QPushButton(self.groupBox_5)
        self.Structure_pushButton.setGeometry(QtCore.QRect(10, 170, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Structure_pushButton.setFont(font)
        self.Structure_pushButton.setObjectName("pushButton_13")

        #Show Acc Comparison
        self.Acc_comp_pushButton = QtWidgets.QPushButton(self.groupBox_5)
        self.Acc_comp_pushButton.setGeometry(QtCore.QRect(10, 240, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Acc_comp_pushButton.setFont(font)
        self.Acc_comp_pushButton.setObjectName("pushButton_14")

        #ResNet Inference 
        self.Inference_pushButton = QtWidgets.QPushButton(self.groupBox_5)
        self.Inference_pushButton.setGeometry(QtCore.QRect(10, 300, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Inference_pushButton.setFont(font)
        self.Inference_pushButton.setObjectName("pushButton_15")
        self.LoadVid_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoadVid_pushButton.setGeometry(QtCore.QRect(10, 390, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.LoadVid_pushButton.setFont(font)
        self.LoadVid_pushButton.setObjectName("pushButton_2")

        #Error windows
        self.ErrorMsg = QtWidgets.QTextEdit(self.centralwidget)
        self.ErrorMsg.setGeometry(QtCore.QRect(30, 10, 1041, 81))
        self.ErrorMsg.setObjectName("textEdit")
        font = QtGui.QFont("Microsoft YaHei", 10)  
        font.setPointSize(20)  
        self.ErrorMsg.setFont(font)
        self.ErrorMsg.setStyleSheet("QTextEdit { color: #8B0000; }")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1102, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadImg_pushButton.setText(_translate("MainWindow", "Load Image"))
        self.groupBox.setTitle(_translate("MainWindow", "1.Background Subtraction"))
        self.Backg_pushButton.setText(_translate("MainWindow", "1.1 Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.Opt_pre_pushButton.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.Opt_trak_pushButton.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3.PCA"))
        self.PCA_pushButton.setText(_translate("MainWindow", "3.1 Dimension Reduction"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4.MINIST Classifier Using VGG19"))
        self.Mdl_str_pushButton.setText(_translate("MainWindow", "4.1 Show Model Structure"))
        self.Predict_pushButton.setText(_translate("MainWindow", "4.3 Predict"))
        self.Graphic_pushButton.setText(_translate("MainWindow", "4.2 Show Accuaracy And Loss"))
        self.Reset_pushButton.setText(_translate("MainWindow", "4.4 Reset"))
        self.groupBox_5.setTitle(_translate("MainWindow", "5.ResNet50"))
        self.Load_img_pb_5.setText(_translate("MainWindow", "5.1 Load Image"))
        self.Show_pushButton.setText(_translate("MainWindow", "5.2 Show Images"))
        self.Structure_pushButton.setText(_translate("MainWindow", "5.3 Show Model Structure"))
        self.Acc_comp_pushButton.setText(_translate("MainWindow", "5.4 Show Comparison"))
        self.Inference_pushButton.setText(_translate("MainWindow", "5.5 Inference"))
        self.LoadVid_pushButton.setText(_translate("MainWindow", "Load Video"))


        #Activation Buttons
        self.LoadVid_pushButton.clicked.connect(self.Load_video)
        self.loadImg_pushButton.clicked.connect(self.Load_image)
        self.Backg_pushButton.clicked.connect(self.Background_act)
        self.Opt_pre_pushButton.clicked.connect(self.Optical_pre_act)
        self.Opt_trak_pushButton.clicked.connect(self.Optical_track_act)
        self.PCA_pushButton.clicked.connect(self.PCA_act)
        self.Mdl_str_pushButton.clicked.connect(self.Vgg19_summary_act)
        self.Graphic_pushButton.clicked.connect(self.Vgg19_graphic)
        self.Predict_pushButton.clicked.connect(self.Inference_act)
        self.Reset_pushButton.clicked.connect(self.Graffiti_reset)
        self.Load_img_pb_5.clicked.connect(self.ResNet_LoadImg)
        self.Show_pushButton.clicked.connect(self.Resnet_show) 
        self.Structure_pushButton.clicked.connect(self.Resnet_structure)
        self.Acc_comp_pushButton.clicked.connect(self.Resnet_comparison)
        self.Inference_pushButton.clicked.connect(self.ResNet_inference)       

    #Task 1 and 2 load video for Background Subtraction and tracking
    def Load_video(self):

        try:
            self.video_path=QFileDialog.getOpenFileName()[0]
            #self.video_path=QFileDialog.getOpenFileName(None, "Select Video", "", "Video Files (*.mp4 *.avi)")[0]

                
            if self.video_path:
                self.vid=cv.VideoCapture(self.video_path)
                self.ErrorMsg.setText("<span style='color: #00008B;'>Video file founded!!</span>")
                QTimer.singleShot(6000, self.ErrorMsg.clear)

                
            
            else:
                errorMessage='Error:Video no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)

    


    def Background_act(self):
        try:

            if self.vid:
                self.Bgs.Background(self.vid)
                self.ErrorMsg.setText(self.Bgs.Endmsg)
                QTimer.singleShot(6000, self.ErrorMsg.clear)

                
            
            else:
                errorMessage='Error:Video no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)


    def Optical_pre_act(self):
        try:

            if self.vid:
                self.ErrorMsg.setText("<span style='color: #00008B;'>Point Traced!!</span>")
                QTimer.singleShot(6000, self.ErrorMsg.clear)
                self.Opt.Preprocess(self.vid)

                if self.Opt.key:
                    self.ErrorMsg.setText(self.Opt.Msg)
                    QTimer.singleShot(10000, self.ErrorMsg.clear)
                

                
            
            else:
                errorMessage='Error:Video no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)


    def Optical_track_act(self):
        try:

            if self.vid:
                self.ErrorMsg.setText("<span style='color: #00008B;'>Tracking...</span>")
                QTimer.singleShot(6000, self.ErrorMsg.clear)
                self.Opt.track_and_display(self.vid)
                

                if self.Opt.key_2:
                    self.ErrorMsg.setText(self.Opt.Msg)
                    QTimer.singleShot(12000, self.ErrorMsg.clear)
                

                
            
            else:
                errorMessage='Error:Video no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)

    
    #Load Image
    def Load_image(self):

        try:
            self.img_path=QFileDialog.getOpenFileName()[0]
                  
            if self.img_path:
                self.ErrorMsg.setText("<span style='color: #00008B;'>Image file founded!!</span>")
                QTimer.singleShot(6000, self.ErrorMsg.clear)
                return self.img_path

                
            
            else:
                errorMessage='Error:Video no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)



    def PCA_act(self):
        try:

            if self.img_path:
                keyboard.add_hotkey("w", self.pca.print_msg)
                self.pca.Pca_func(self.img_path)
                
                

            else:
                errorMessage='Error:Image no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(6000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)


    
    def Graffiti_reset(self):
        self.canvas.clearCanvas()
        self.ErrorMsg.setText("<span style='color: #00008B;'>Window Reset!!</span>")
        QTimer.singleShot(6000, self.ErrorMsg.clear)


    def Inference_act(self):
        self.Vgg19.inference()
        self.ErrorMsg.setText(self.Vgg19.Msg)
        QTimer.singleShot(20000, self.ErrorMsg.clear)


    def Vgg19_summary_act(self):
        self.Vgg19.Model_summary()
        self.ErrorMsg.setText(self.Vgg19.Msg)
        QTimer.singleShot(8000, self.ErrorMsg.clear)

    def Vgg19_graphic(self):
        self.canvas.displayImage()
        self.ErrorMsg.setText("<span style='color: #00008B;'>Displaying Acc&Loss graphic...</span>")
        QTimer.singleShot(6000, self.ErrorMsg.clear)

       
    def Resnet_show(self):
        self.show.show()
        self.ErrorMsg.setText("<span style='color: #00008B;'>Displaying Inference Classes images...</span>")
        QTimer.singleShot(6000, self.ErrorMsg.clear)
        
    #Resnet Load Image
    def ResNet_LoadImg(self):

        try:
            Tk().withdraw()
            self.image_path = askopenfilename()
            
                  
            if self.image_path:
                image = cv.imread(self.image_path)
                self.rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                rgb_image = cv.resize(self.rgb_image, (self.graphicsView.width(), self.graphicsView.height()))

                height, width,_ = rgb_image.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                
                scene = QtWidgets.QGraphicsScene()

                pixmap = QtGui.QPixmap.fromImage(q_image)
                
                #pixmap = pixmap.scaled(self.graphicsView.width(), self.graphicsView.height(), QtCore.Qt.KeepAspectRatio)
                
                scene.addPixmap(pixmap)
                #scene.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Asegurarse de que la escena tiene el tamaño del QPixmap

                self.graphicsView.setScene(scene)
                self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

                self.graphicsView.show() 
                
                self.ErrorMsg.setText("<span style='color: #00008B;'>Image file loaded!!</span>")
                QTimer.singleShot(10000, self.ErrorMsg.clear)
                

                
            
            else:
                errorMessage='Error:Image no founded or corrupted, please check!!'
                self.ErrorMsg.setText(errorMessage)
                QTimer.singleShot(10000, self.ErrorMsg.clear)


        except Exception:
            errorMessage=f"<span style='font-family:Microsoft YaHei; font-size:14pt;'>An error occurred when you upload the image.<br>" \
                        "Check that the image is not corrupted,<br>" \
                        "the folder does not have Chinese chars or special chars!!</span>"
              
            self.ErrorMsg.setText(errorMessage)
    
    def ResNet_inference(self):
        self.Resnet.inference_main(self.rgb_image)
        self.ErrorMsg.setText(self.Resnet.Msg)
        QTimer.singleShot(20000, self.ErrorMsg.clear)



    def Resnet_structure(self):
        self.Resnet.Show_structure()
        self.ErrorMsg.setText(self.Resnet.Msg)
        QTimer.singleShot(8000, self.ErrorMsg.clear)


    def Resnet_comparison(self):
        self.show.show_validation_acc()
        self.ErrorMsg.setText("<span style='color: #00008B;'>Displaying Accuaracy Comparison...</span>")
        QTimer.singleShot(10000, self.ErrorMsg.clear)
        

if __name__== "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
