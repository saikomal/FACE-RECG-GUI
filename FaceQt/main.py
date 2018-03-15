import sys
import cv2
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap


class Life2Coding(QDialog):
    def __init__(self):
        super(Life2Coding, self).__init__()

        loadUi('Attendence.ui', self)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 311)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 241)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    def update_frame(self):
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        ret, self.image = self.capture.read()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #gray = frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)

        )


        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        self.image = cv2.flip(self.image, 1)
        self.displayImage(self.image, 1)

    def stop_webcam(self):
        self.timer.stop()

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
            self.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.imglabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Life2Coding()
    window.setWindowTitle('Attendence')
    window.show()
    sys.exit(app.exec_())
