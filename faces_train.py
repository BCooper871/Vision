#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:37:06 2022

@author: ben
"""

import os 
import cv2 as cv
import numpy as np

DIR = r'/home/ben/Documents/Courses/OpenCV/faces_train/train'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
for i in os.listdir(DIR):
    people.append(i)
    
features = []
labels = []
    
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            
            img_arrary = cv.imread(img_path)
            gray = cv.cvtColor(img_arrary, cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=3)
            
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()
print('training done')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()    

#Training

face_recognizer.train(features, labels)    

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)