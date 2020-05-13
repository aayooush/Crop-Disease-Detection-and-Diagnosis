from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

import pymysql
conn = pymysql.connect(host="localhost",user="root",passwd="",db="cddd")
print('Databse Connected ...')
cur = conn.cursor()

IMG_W = 224
IMG_H = 224
disease = {0: 'Apple___APPLE SCAB',
 1: 'Apple___BLACK ROT',
 2: 'Apple___CEDAR APPLE RUST',
 3: 'Apple___Healthy',
 4: 'Blueberry___Healthy',
 5: 'Cherry___POWDERY MILDEW',
 6: 'Cherry___Healthy',
 7: 'Corn___COMMON RUST',
 8: 'Corn___LEAF SPOT',
 9: 'Corn___LEAF BLIGHT',
 10: 'Corn___Healthy',
 11: 'Grape___ESCA (BLACK MEASLES)',
 12: 'Grape___LEAF BLIGHT',
 13: 'Grape___BLACK ROT',
 14: 'Grape___Healthy',
 15: 'Orange___CITRUS GREENING',
 16: 'Peach___Bacterial Spot',
 17: 'Peach___Healthy',
 18: 'Bell Pepper___Bacterial Spot',
 19: 'Bell Pepper___Healthy',
 20: 'Potato___Early Blight',
 21: 'Potato___Late Blight',
 22: 'Potato___Healthy',
 23: 'Raspberry___Healthy',
 24: 'Soybean___Healthy',
 25: 'Squash___POWDERY MILDEW',
 26: 'Strawberry___LEAF SCORCH',
 27: 'Strawberry___Healthy',
 28: 'Tomato___BACTERIAL SPOT',
 29: 'Tomato___EARLY BLIGHT',
 30: 'Tomato___LATE BLIGHT',
 31: 'Tomato___LEAF MOLD',
 32: 'Tomato___LEAF SPOT',
 33: 'Tomato___SPIDER MITES',
 34: 'Tomato___TARGET SPOT',
 35: 'Tomato___YELLOW LEAF CURL VIRUS',
 36: 'Tomato___TOMATO MOSAIC VIRUS',
 37: 'Tomato___Healthy'}
 

def normalize(df):    
    return (df - df.min()) / (df.max() - df.min())

def process(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR) 
    b,g,r = cv2.split(im)
    image = cv2.merge([r,g,b])
    res = cv2.resize(image,(IMG_W, IMG_H), interpolation = cv2.INTER_CUBIC)
    return np.array(res, dtype=np.int32)

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/PlantDisease.h5'

result = ""
disease_name = ""

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])
model._make_predict_function()          # Necessary

# model = joblib.load(MODEL_PATH)

print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img = process(img_path)

    input_image = []
    input_image.append(normalize(img))
    input_image = np.array(input_image)
    preds = model.predict(input_image)
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('plant-detector.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        fp = []
        for i in preds:
            fp.append(list(i).index(max(i)))
            
        result1 = disease[fp[0]]
        
        h = result1.split("___")
        if h[1] == "Healthy":
            result = h[0] + "__Healthy Plant"
        else:
            result = h[0] + "__Infected Plant"
        
        global disease_name
        disease_name = h[1]

        return result
    return None
    
    #return render_template('landing.html',username=u,img=imgs,title=ts,gen=GenList,topimg=imgstop,toptit=tstop,gen1=g1tit,gen2=g2tit,gimg1=g1img,gimg2=g2img)

@app.route('/info', methods=['GET'])
def info():
    try:
        d = "select * from diseases where Disease_Name=%s;"
        global disease_name
        cur.execute(d,disease_name)
        disease_data = cur.fetchall()
        # print(disease_data)
        name = disease_data[0][1]
        symp = disease_data[0][2]
        prec = disease_data[0][3]
    except Exception as e:
        print("ERROR=",e)
    return render_template('plant-info.html',n=name,s=symp,p=prec)

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('192.168.0.7',5000),app)
    http_server.serve_forever()
