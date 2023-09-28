#Dependancies
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import draw

import requests
import json

#Keypoints Using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model

def postAPI(data):
    url = "http://127.0.0.1:8000/add"
    headers = {'Content-type': 'application/json'}
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response if response.status_code == 200 else None
    except:
        return None

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def json_keypoints(results):
    frame = {
        "landmarks": []
    }
    if results.left_hand_landmarks:
        frame["landmarks"].append({"x": res.x, "y": res.y, "z": res.z} for res in results.left_hand_landmarks.landmark)
    else:
        for _ in range(21):
            frame["landmarks"].append({"x": 0, "y": 0, "z": 0})
    if results.right_hand_landmarks:
        frame["landmarks"].append({"x": res.x, "y": res.y, "z": res.z} for res in results.right_hand_landmarks.landmark)
    else:
        for _ in range(21):
            frame["landmarks"].append({"x": 0, "y": 0, "z": 0})
    if results.pose_landmarks:
        frame["landmarks"].append({"x": res.x, "y": res.y, "z": res.z} for res in results.pose_landmarks.landmark)
    else:
        for _ in range(33):
            frame["landmarks"].append({"x": 0, "y": 0, "z": 0})

    return frame
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    #rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([lh, rh, pose]).tolist()
0
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    err = False
    while not err:
        #action = input("Enter action: ")
        action = "hello"
        #Default collect 30 sequences of data
        for num_sequence in range(5):
            if err: break
            #JSON data to send
            data = {
              "action" : action,
              "frames": []
            }
            for num_frame in range(30):
                if err: break
                # Read feed
                ret, frame = cap.read()
                
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw.styled_landmarks(image, results)
                
                #Wait buffer for each sequence & label
                if num_frame == 0:
                    draw.img_print(image, "STARTING COLLECTION")
                    cv2.waitKey(5000)
                draw.img_print_collect(image, action, num_sequence)
                
                #Extract and save keypoints
                data["frames"].append(json_keypoints(results))

                #Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    err = True
            #Send sequence of frames to API
            if err: break
            if not postAPI(data):
                err = True
    cap.release()
    cv2.destroyAllWindows()

