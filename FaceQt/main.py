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
def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.
$

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model of disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified.
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf
knn_clf = train("knn_examples/train")

def predict(im, knn_clf = None, model_save_path ="", DIST_THRESH = .5):
    """
    recognizes faces in given image, based on a trained knn classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """
    #X_img = face_recognition.load_image_file(X_img_path)
    X_img = im
    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]


class Life2Coding(QDialog):


    def __init__(self):
        super(Life2Coding, self).__init__()
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        c=sqlite3.connect("base.db")
        loadUi('Attendence.ui', self)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 311)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 241)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        c=sqlite3.connect("base.db")

        face_names = []
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        ret, self.image = self.capture.read()
        preds = predict(self.image, knn_clf=knn_clf)

        print(preds)
        for i in range(len(preds)):
            sq = "insert into identity(Id) values('" + str(preds[i][0]) + "')"
            face_names.append(preds[i][0])
            c.execute(sq)
        c.commit()
        print(face_names)
        if(len(face_names)!=0 and face_names[0]!='N/A'):
            self.loadImage('/home/klu/Desktop/FACE-GUI/FaceQt/knn_examples/train/'+face_names[0]+'/img1.jpg')
            self.label_10.setText(face_names[0])
            self.label_11.setText("150030711")
            self.label_12.setText("CSE")
            self.label_13.setText("9441884057")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #gramaheshy = frame
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

    def loadImage(self,fname):
        self.image1=cv2.imread(fname)
        self.displayImage1(self.image1,1)

    def displayImage1(self, img, window=1):
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

    def displayImage(self, img, window=1):
        #self.loadImage('logo.jpg')
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
