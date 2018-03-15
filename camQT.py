import sys
import cv2
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
class Life2Coding(QDialog):
	def __init__(self):
		super(Life2Coding,self).__init__()
		loadUi('Attendence.ui',self)

if __name__=='__main__':
	app = QApplication(sys.argv)
	window=Life2Coding()
	window.setWindowTitle('Attendence')
	window.show()
	sys.exit(app.exec_())
