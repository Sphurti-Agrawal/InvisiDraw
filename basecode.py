import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup for Canvas
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes for color points
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Colors for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create the paint window
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for a mirrored effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of the index fingertip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert to pixel coordinates
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a circle at the fingertip
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                # Check for color selection or drawing
                if cy <= 65:
                    if 40 <= cx <= 140:  # Clear Button
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0
                        paintWindow[67:, :, :] = 255
                    elif 160 <= cx <= 255:
                        colorIndex = 0  # Blue
                    elif 275 <= cx <= 370:
                        colorIndex = 1  # Green
                    elif 390 <= cx <= 485:
                        colorIndex = 2  # Red
                    elif 505 <= cx <= 600:
                        colorIndex = 3  # Yellow
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft((cx, cy))
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft((cx, cy))
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft((cx, cy))
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft((cx, cy))

        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw lines of all the colors on the canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Add the color buttons to the live frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)

        cv2.putText(frame, "CLEAR ALL", (49, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

        # Show all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        # If the 'q' key is pressed then stop the application
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
