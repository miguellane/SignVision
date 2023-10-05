#Machine Learning
import cv2
import mediapipe as mp
#Data types
import queue
import json
import requests
#TTS
import pyttsx3
#Project files
import draw




#Keypoints Using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model

class Sequence():
    def __init__(self, action=None):
        self.action = action
        self.frames = queue.Queue(maxsize=30)

    #Add frame to sequence, if sequence is full, send to API
    def queueFrame(self, landmarks):
        frame = self.landmarksToDict(landmarks)
        self.frames.put(frame)

        result = None
        if self.frames.full():
            result = self.postAPI()
            self.frames.get()
        return result

    #Convert mediapipe landmarks to dict, with 0 if no landmarks
    def landmarksToDict(self, landmarks):
        frame = {
            #Landmarks [left 21, right 21, pose 33]
            "landmarks": [{"x": 0, "y": 0, "z": 0}] * 75
        }
        if landmarks.left_hand_landmarks:
            frame["landmarks"][0:21] = [{"x": land.x, "y": land.y, "z": land.z} for land in landmarks.left_hand_landmarks.landmark]
        if landmarks.right_hand_landmarks:
            frame["landmarks"][21:42] = [{"x": land.x, "y": land.y, "z": land.z} for land in landmarks.right_hand_landmarks.landmark]
        if landmarks.pose_landmarks:
            frame["landmarks"][42:75] = [{"x": land.x, "y": land.y, "z": land.z} for land in landmarks.pose_landmarks.landmark]

        return frame

    def postAPI(self):
        fulldict = {
            "action": self.action,
            "frames": list(self.frames.queue)
        }
        url = "http://127.0.0.1:8000/add"
        payload = json.dumps(fulldict)
        headers = {'Content-type': 'application/json'}
        try:
            response = requests.post(url, payload, headers=headers)
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

def collect():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        err = False
        while not err:
            action = input("Enter action: ")
            #Collect 5 sequences
            for num_sequence in range(5):
                if err: break
                seq = Sequence(action)
                #Collect 30 frames of data
                for num_frame in range(30):
                    if err: break
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, landmarks = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw.styled_landmarks(image, landmarks)

                    #Wait buffer for each sequence & label
                    if num_frame == 0:
                        draw.img_print(image, "STARTING COLLECTION")
                        cv2.waitKey(5000)
                    draw.img_print_collect(image, seq.action, num_sequence)

                    #Queue frame and send to API if valid
                    seq.queueFrame(landmarks)

                    #Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        err = True
                        break
        cap.release()
        cv2.destroyAllWindows()

def classify():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            engine = pyttsx3.init()
            seq = Sequence()
            sentence = queue.Queue(maxsize=5)
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, landmarks = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw.styled_landmarks(image, landmarks)

            #Queue frame and send to API if valid
            result = seq.queueFrame(landmarks)
            if result:
                print(result)
                if sentence[-1] != result:
                    if sentence.full():
                        sentence.get()
                    sentence.put(result)
                    engine.say("Hello")
                   
            draw.img_print(image, list(sentence.queue))
            engine.runAndWait()

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    return

mode = input("Enter mode: (classify or collect): ")
if mode == "classify":
    classify()
elif mode == "collect":
    collect()
else:
    print("Invalid mode")