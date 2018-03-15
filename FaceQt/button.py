import sys
import cv2
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder
import sqlite3
import time
class Life2Coding(QDialog):
    def __init__(self):
        super(Life2Coding,self).__init__()
        loadUi('life1.ui',self)
        self.image=None
        self.pushButton.clicked.connect(self.loadClicked)

    def loadClicked(self):
        name = 'komal'
        self.loadImage('/home/klu/Desktop/FACE-GUI/FaceQt/knn_examples/train/'+name+'/img1.jpg')

    def loadImage(self,fname):
        self.image=cv2.imread(fname)
        self.displayImage(self.image, 1)
    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(outImage))
            self.label.setScaledContents(True)


if __name__ == '__main__':
    app=QApplication(sys.argv)
    window = Life2Coding()
    window.setWindowTitle('Attendence')
    window.show()
sys.exit(app.exec_())
