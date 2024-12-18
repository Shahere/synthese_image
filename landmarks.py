import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from imutils import face_utils 
import sys
import math

DEBUG = True

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Avoir les points d'un visage
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        #pas de visage
        return np.array([])
    
    shape = predictor(gray, faces[0])
    points = []
    for i in range(68):
        x = shape.part(i).x
        y = shape.part(i).y
        points.append((x, y))
    
    #Affiches les 68 landmarks du visage en rouge
    if DEBUG:
        for (i, rect) in enumerate(faces):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
    
    if DEBUG:
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return points

# Il ne faut pas prendre une image en portait et une en paysage, sinon le resize va tout casser et
# le model sera incapable de reperer un visage
source = cv2.imread("./image/savinien3.jpg")
destination = cv2.imread("./image/ewen.jpg")

source = cv2.resize(source, (destination.shape[1], destination.shape[0]))

src_landmarks = get_landmarks(source)
dst_landmarks = get_landmarks(destination)
src_landmarks = np.array(src_landmarks, dtype=np.int32)
dst_landmarks = np.array(dst_landmarks, dtype=np.int32)

while True:
    cv2.imshow("Blended", source)
    k = cv2.waitKey(10) & 0xFF
    if k== 27:
        break
cv2.destroyAllWindows()


