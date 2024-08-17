import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from tkinter import Tk, colorchooser, filedialog

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup for Canvas
points = [deque(maxlen=1024)]
index = 0
selected_color = (255, 0, 0)  # Default color for drawing
drawing_paused = False
brush_thickness = 2  # Default brush thickness
thickness_levels = [2, 4, 6, 8]  # Available brush thickness levels
show_thickness_panel = False  # Whether to show the thickness panel

# Create the paint window
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load button images
color_picker_img = cv2.imread('color_picker_icon.png', cv2.IMREAD_UNCHANGED)
clear_img = cv2.imread('clear_icon.png', cv2.IMREAD_UNCHANGED)
save_img = cv2.imread('save_icon.png', cv2.IMREAD_UNCHANGED)
brush_thickness_img = cv2.imread('brush_thickness_icon.png', cv2.IMREAD_UNCHANGED)

# Resize images to fit the button area
color_picker_img = cv2.resize(color_picker_img, (60, 60))
clear_img = cv2.resize(clear_img, (60, 60))
save_img = cv2.resize(save_img, (60, 60))
brush_thickness_img = cv2.resize(brush_thickness_img, (60, 60))

# Function to detect if a finger is up
def is_finger_up(hand_landmarks, finger_tip_idx, finger_pip_idx):
    return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_pip_idx].y

# Function to overlay images with transparency
def overlay_image(frame, img, pos):
    y1, y2 = pos[1], pos[1] + img.shape[0]
    x1, x2 = pos[0], pos[0] + img.shape[1]

    if img.shape[2] == 4:  # If the image has an alpha channel
        alpha_s = img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    else:  # If the image does not have an alpha channel
        frame[y1:y2, x1:x2] = img

# Initialize MediaPipe Hands with higher detection confidence
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
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
                
                # Get the coordinates of the index fingertip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_cx, index_cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Draw a green circle on the index fingertip
                cv2.circle(frame, (index_cx, index_cy), 10, (0, 255, 0), cv2.FILLED)

                # Check if the index and middle fingers are up
                index_finger_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                middle_finger_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)

                # Update drawing state based on the index and middle fingers
                if index_finger_up and middle_finger_up:
                    if not drawing_paused:  # Transitioning to paused state
                        points.append(deque(maxlen=1024))  # Start a new line segment
                        index += 1
                    drawing_paused = True
                elif index_finger_up and not middle_finger_up:
                    if drawing_paused:  # Transitioning to drawing state
                        points.append(deque(maxlen=1024))  # Start a new line segment
                        index += 1
                    drawing_paused = False

                if not drawing_paused:
                    # Check for button presses
                    if index_cy <= 65:
                        if 40 <= index_cx <= 100:  # Color Picker Button
                            root = Tk()
                            root.withdraw()  # Hide the root window
                            color = colorchooser.askcolor()[0]
                            if color:
                                selected_color = tuple(map(int, color[::-1]))  # Convert to BGR format
                            root.destroy()
                        elif 160 <= index_cx <= 220:  # Clear Button
                            paintWindow[:] = 255
                            points = [deque(maxlen=1024)]
                            index = 0
                        elif 275 <= index_cx <= 335:  # Save Button
                            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
                            if file_path:
                                cv2.imwrite(file_path, paintWindow)
                        elif 390 <= index_cx <= 450:  # Brush Thickness Button
                            show_thickness_panel = not show_thickness_panel
                    elif show_thickness_panel:
                        for i, thickness in enumerate(thickness_levels):
                            if 390 + i * 30 <= index_cx <= 420 + i * 30 and 65 <= index_cy <= 95:
                                brush_thickness = thickness
                                show_thickness_panel = False
                                break
                else:
                    # Ensure points list is long enough
                    if index >= len(points):
                        points.append(deque(maxlen=1024))
                    points[index].appendleft((index_cx, index_cy))
        
        else:
            points.append(deque(maxlen=1024))
            index += 1

        # Draw lines on the canvas and frame
        for i in range(len(points)):
            for j in range(1, len(points[i])):
                if points[i][j - 1] is None or points[i][j] is None:
                    continue
                cv2.line(frame, points[i][j - 1], points[i][j], selected_color, brush_thickness)
                cv2.line(paintWindow, points[i][j - 1], points[i][j], selected_color, brush_thickness)

        # Overlay the custom button images
        overlay_image(frame, color_picker_img, (40, 5))
        overlay_image(frame, clear_img, (160, 5))
        overlay_image(frame, save_img, (275, 5))
        overlay_image(frame, brush_thickness_img, (390, 5))

        # Draw the brush thickness panel if needed
        if show_thickness_panel:
            for i, thickness in enumerate(thickness_levels):
                cv2.rectangle(frame, (390 + i * 30, 65), (420 + i * 30, 95), (200, 200, 200), -1)
                cv2.putText(frame, str(thickness), (400 + i * 30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Show all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        # If the 'q' key is pressed then stop the application
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the camera and all resources
cap.release()
cv2.destroyAll
