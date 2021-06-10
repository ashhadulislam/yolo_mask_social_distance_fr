from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import os

import face_recognition



def both_mask_face(frame,mask_params,face_params):
    isMaskLess=False
    net,model,face_confidence=mask_params
    known_faces, known_face_names=face_params

    (h, w) = frame.shape[:2]
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
    (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > face_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_for_fd=face[:]


            # this part for mask
            # resize it to 224x224, and preprocess it
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if mask > withoutMask:
                label="Mask"
                color = (0, 255, 0)
                name=""
            else:
                label="No Mask"
                isMaskLess=True
                color = (0, 0, 255)
                name=""
                # since this person is not wearing mask, let us 
                # detect their face
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(face_for_fd)
                face_encodings = face_recognition.face_encodings(face_for_fd, face_locations)
                # let us enforce single face                
                # face_names = []
                # for face_encoding in face_encodings:
                for i in range(len(face_encodings)):
                    # will run just once as we hopefully have
                    # one face
                    if i>0:
                        break
                    face_encoding=face_encodings[0]                
                    # See if the face is a match for the known face(s)
                    match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)


                    # If you had more than 2 faces, you could make this logic a lot prettier
                    # but I kept it simple for the demo
                    print(match)
                    name = "..."
                    for i in range(len(match)):
                        if match[i]:
                            name=known_face_names[i]
                            break
                    # face_names.append(name)
                    

                # so above loop just runs once
                # hopefully tries to identify the person
                # and comes out

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)    
            if name!="..." and name!="":
                cv2.rectangle(frame, (startX, endY - 25), (endX, endY), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (startX, endY - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame,isMaskLess



            
def enqueue_image(frame):

    prepend="/Users/amirulislam/projects/Qatar/Artificial_Intelligence_National_Competition/old_drink_new_bottle/YOLO_pdf_flask/static/surveil/"
    file_num=0
    file_limits=10
    for i in range(file_limits-1):
        old_frame=cv2.imread(prepend+str(i)+".jpg")
        cv2.imwrite(prepend+str(i+1)+".jpg",old_frame)        
    
    file_name=prepend+str(file_num)+".jpg"
    cv2.imwrite(file_name,frame)





