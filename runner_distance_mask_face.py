'''
on every person detected, we apply local binary pattern
to modify the image 
then we apply the different pooling methods

this program we try to apply multiple 
pooling types - avgpool, minpool and maxpool
'''

from datetime import datetime

import numpy as np
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from keras.models import Model
from skimage import feature
import os
import shutil

import time
import face_mask_helper
import face_detection_helper
import combo_mask_face


from math import pow, sqrt

detection_confidence=0.75
safe_dist=150
skipcount=10

def detect_persons(frame,x,y,w,h,net,layer_names,output_layers,classes):
    print("In detect_persons",frame.shape)
    print("x,y,w,h = ",x,y,w,h)

    

    startX=x
    startY=y
    endX=startX+w
    endY=startY+h


    # cv2.rectangle(frame, (startX, startY), (endX, endY), (0,250,50), 2)
        
    roi=frame[startY:endY, startX:endX]
    print("ROI shape = ",roi.shape)


        

    # (h, w) = frame.shape[:2]
    (height, width) = roi.shape[:2]
    channels=roi.shape[2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    # blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
    # blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    
    # prototxt=args["prototxt"]
    # ppl_model=args["model"]

    net.setInput(blob)
    outs = net.forward(output_layers)
    person_class_id=0
    class_ids = []
    confidences = []
    boxes = []

    
    # Focal length
    F = 615


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > detection_confidence:
                if class_id==person_class_id:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)                    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print("Count of people = ",len(indexes))

    count_people=0
    people_dict={}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])+" "+str(round(confidences[i]*100))
            # color = colors[class_ids[i]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            startX, endX, startY, endY=x,x+w,y,y+h
            
            height=round(endY-startY,4)
            # Distance from camera based on triangle similarity
            distance=(165*F)/height

            # Mid point of bounding box
            x_mid = round((startX+endX)/2,4)
            y_mid = round((startY+endY)/2,4)

            # Mid-point of bounding boxes (in cm) based on triangle similarity technique
            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F


            # print(startX, startY, endX, endY)
            # cv2.putText(frame, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)
            people_dict[count_people]={}
            people_dict[count_people]["coords"]=(startX, startY, endX, endY)
            people_dict[count_people]["confidence"]=confidences[i]
            people_dict[count_people]["position"]=(x_mid_cm,y_mid_cm,distance)
            count_people+=1
                
    return frame,count_people,people_dict



# among the persons in the frame, detect the ones that are close by
def detect_close_persons(people_dict):
    print("In detect_close_persons")
    isClose=False
    
    dict_breach_details={}
    list_breach_details=[]

    close_objects = set()
    for i in people_dict.keys():
        for j in people_dict.keys():
            if i < j:
                x_i=people_dict[i]["position"][0]
                y_i=people_dict[i]["position"][1]
                dist_i=people_dict[i]["position"][2]

                x_j=people_dict[j]["position"][0]
                y_j=people_dict[j]["position"][1]
                dist_j=people_dict[j]["position"][2]

                
                dist = sqrt(pow(x_i-x_j,2) + pow(y_i-y_j,2) + pow(dist_i-dist_j,2))
                # print("distance between ",i," and ",j," is ",dist,"\n")

                # Check if distance less than 2 metres or 200 centimetres
                if dist < safe_dist:                    
                    close_objects.add(i)
                    close_objects.add(j)      
                    isClose=True              
                    temp_dict={}
                    # can we convert the distance to feet
                    # dist_feet=int(0.0328084*dist)
                    temp_dict["distance"]=int(dist)
                    temp_dict["employee_1"]=i
                    temp_dict["employee_2"]=j
                    list_breach_details.append(temp_dict)

                else:
                    pass
                    # breach_happened.append(False)  
    dict_breach_details["breach_data"]=list_breach_details
    


              
    return close_objects,dict_breach_details,isClose
 

def setupYOLO():
    # Load Yolo
    print("LOADING YOLO")
    yolocation="res/yolo_details/"
    net = cv2.dnn.readNet(yolocation+"yolov3.weights", yolocation+"yolov3.cfg")
    #save all the names in file o the list classes
    classes = []
    with open(yolocation+"coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    #get layers of the network
    layer_names = net.getLayerNames()
    #Determine the output layer names from the YOLO model 
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLO LOADED")  
    return net,layer_names,output_layers,classes  





def main():

    # setup YOLO model
    net,layer_names,output_layers,classes=setupYOLO()
    face_net,mask_model,face_confidence=face_mask_helper.setup_face_mask()
    known_faces,known_face_names=face_detection_helper.setup_faces()

    



    # cap = cv2.VideoCapture("res/videoplaybackcamcafe.mp4")    
    cap = cv2.VideoCapture(0)
    # any one of the two lines above
    

       
    
    
    
    ret, frame = cap.read()
    h=frame.shape[0]
    w=frame.shape[1]
    x_start=0
    y_start=0


    
    
    count_frames=0
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        count_frames+=1
        if count_frames%skipcount==0:
            
            # input("Press enter for next")
            # frame,count,people_dict=detect_persons(frame,x,y,w,h)            
            frame,count,people_dict=detect_persons(frame,x_start,y_start,w,h,net,layer_names,output_layers,classes)            
            close_objects,dict_breach_details,isClose = detect_close_persons(people_dict)
            
            print("************************************")
            print("Number of people is", count, people_dict)
            print("************************************")
            print("Proximity check, close_objects",close_objects)
            print("\n")
            print("dict_breach_details",dict_breach_details)

            print("************************************")
            print("*Going for face recog & mask detection**")
            print("************************************") 

            mask_params=(face_net,mask_model,face_confidence)
            face_params=(known_faces, known_face_names)
            frame,isMaskLess=combo_mask_face.both_mask_face(frame, mask_params,face_params)
            # parameters are 1 frame
            # 2 for mask
            # 3 for face recog           

            # print("************************************")
            # print("******Going for mask detection******")
            # print("************************************")            
            # frame=face_mask_helper.detect_face_mask(frame,face_net,mask_model,face_confidence)

            # print("************************************")
            # print("******Going for face recognitn******")
            # print("************************************")            
            # frame=face_detection_helper.process_frame_give_names(frame,known_faces, known_face_names)

            
            for i in range(count):


                startX, startY, endX, endY=people_dict[i]["coords"]
                current_confidence=people_dict[i]["confidence"]

                crop_img = frame[startY:endY, startX:endX]
                print("Shape of cropped image is ",crop_img.shape)
                
                            

                
                # this part to mark boxes on people              
                COLOR = (0,255,0)
                if i in close_objects:                    
                    COLOR=(0,0,255)
                # label = "{}: {:.2f}%".format(i, current_confidence * 100)
                label=str(i)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
                        
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
                

            
            if isMaskLess or isClose:
                # add the time
                # cv2.rectangle(frame, (9, 10), (100, 20), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame,str(datetime.now()),(10,20), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),2,cv2.LINE_AA)
                combo_mask_face.enqueue_image(frame)

            # resize the frame for
            # screen record purpose
            resize = cv2.resize(frame, (600, 400)) 
            cv2.imshow('frame',resize)  

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

