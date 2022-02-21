#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2 as cv
import numpy as np
classifer = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[ ]:


camera = cv.VideoCapture(0)

while True:
    
    _, frame = camera.read()
    frame = cv.flip(frame, 1)
    faces = classifer.detectMultiScale(frame, 1.7, 5)
    
    for (x,y,w,h) in faces:    
        cv.rectangle(frame, (x,y), (x+w, y+h), (170,90,120), 2)
    

    
    cv.imshow("Original", frame)
    key = cv.waitKey(0)
    
    if (key == 27):
        cv.destroyAllWindows()
        break


camera.release()


# In[ ]:




