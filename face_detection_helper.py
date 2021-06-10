import face_recognition
import cv2
import os



def setup_faces():

    known_faces=[]
    known_face_names=[]
    # Load some sample pictures and learn how to recognize them.
    for face_file in os.listdir("faces"):
        print(face_file)
        if ".DS" in face_file:
            continue
        face_img=face_recognition.load_image_file("faces/"+face_file)
        face_enc=face_recognition.face_encodings(face_img)[0]
        known_faces.append(face_enc)
        known_face_names.append(face_file.split(".")[0])

    return known_faces,known_face_names


