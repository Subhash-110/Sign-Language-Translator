import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3  # For text-to-speech
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

model = pickle.load(open('model.pkl', 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize last_prediction_time to prevent immediate speech on startup
last_prediction_time = 0
prediction_interval = 5  # Speak every 5 seconds

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        imageWidth, imageHeight = image.shape[:2]
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                data = []
                for point in mp_hands.HandLandmark:
                    normalizedLandmark = hand_landmarks.landmark[point]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                    data.append(normalizedLandmark.x)
                    data.append(normalizedLandmark.y)
                    data.append(normalizedLandmark.z)

                out = model.predict([data])
                predicted_gesture = out[0]  # Get the predicted gesture string

                current_time = time.time()
                if current_time - last_prediction_time >= prediction_interval:
                    print(predicted_gesture) # print the result in console also
                    engine.say(predicted_gesture) # speak the result
                    engine.runAndWait()
                    last_prediction_time = current_time


                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                image = cv2.putText(image, predicted_gesture, org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
engine.stop()  # Important: Stop the engine when done