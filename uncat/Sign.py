import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./Data/Models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:

    cap = cv2.VideoCapture(1)
    while True:
        success, numpy_image = cap.read()
        if success:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

            recognition_result = recognizer.recognize(mp_image)
            if recognition_result.gestures:
                print(recognition_result.gestures[0][0])
                cv2.putText(numpy_image, str(recognition_result.gestures[0][0]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
            cv2.imshow("Image", numpy_image)
            cv2.waitKey(10)
    

if __name__ == "__main__":
    main()