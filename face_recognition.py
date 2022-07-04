#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:10:34 2022

@author: ben
"""

import cv2 as cv
import os

DIR = r'/home/ben/Documents/Courses/OpenCV/faces_train/train'

people = []
for i in os.listdir(DIR):
    people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

#features = np.load('features.npy')
#labels = np.load('lables.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()    
face_recognizer.read('face_trained.yml')

img = cv.imread(r'/home/ben/Documents/Courses/OpenCV/faces_train/validate/mila/images5.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('person', gray)

#detect face

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label is {people[label]} with confidence of {confidence}')
    
    cv.putText(img, str(people[label]), (10,45), cv.FONT_HERSHEY_PLAIN, 2.0,
    (0,255,0), thickness=1)
    cv.putText(img, str(f'{confidence:.1f} % confidence this is'), (10,25), cv.FONT_HERSHEY_PLAIN, 2.0,
    (0,255,0), thickness = 1)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
cv.imshow('Detected Face', img)
cv.imwrite('image.jpeg', img)

cv.waitKey(0)