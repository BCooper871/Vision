#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:42:15 2022

@author: ben
"""

import cv2 as cv

img = cv.imread('TAIM.jpg') 

#cv.imshow('Tom Hardy', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('grey Tom', gray)

harr_cascade = cv.CascadeClassifier('harr_face.xml')

faces_rect = harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'# of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Dectected Faces', img)

cv.waitKey(0)
