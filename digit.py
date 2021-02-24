import pandas as pd
import plotly.express as px 
import statistics 
import cv2
import csv 
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml 
from PIL import Image
import PIL.ImageOps
import os, ssl, time

x, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts()) 

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 9, train_size = 7500, test_size = 2500)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0
classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scale, y_train)
y_predict = classifier.predict(x_test_scale)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BG2GRAY)
        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        image_pil = Image.fromarray(roi)
        image_l = image_pil.convert('L')
        image_l_resize = image_l.resize((28, 28), Image.ANTIALIAS) 
        image_invert = PIL.ImageOps.invert(image_l_resize)
        pixel_filter = 20
        min_image_pixel = np.percentile(image_invert, pixel_filter)
        image_scaled = np.clip(min_image_pixel, 0, 255) 
        max_image_pixel = np.max(image_invert) 
        image_array = np.asarray(image_scaled)/max_image_pixel 
        test_sample = np.array(image_array).reshape(1784) 
        test_predict = classifier.predict(test_sample)
        print(test_predict) 
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break 
    except Exception as e :
        pass 

cap.release()
cv2.destroyAllWindows() 