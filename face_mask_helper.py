from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import os
import config


# this part for masks
def setup_face_mask():
    face_mask_args=config.face_mask_args
    prototxtPath = face_mask_args["prototxtPath"]
    weightsPath = face_mask_args["weightsPath"]
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model(face_mask_args["modelPath"]) 
    face_confidence=face_mask_args["face_confidence"]
    print("Setup face mask")
    return net,model,face_confidence   



