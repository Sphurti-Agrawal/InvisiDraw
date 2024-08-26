import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from tkinter import Tk, colorchooser, filedialog, simpledialog

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup for Canvas
points = [deque(maxlen=1024)]
index = 0
selected_color = (255, 0, 0)  # Default color for drawing
drawing_paused = False
current_shape = None
start_pos = None

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
circle_img = cv2.imread('circle_icon.png', cv2.IMREAD_UNCHANGED)
rectangle_img = cv2.imread('rectangle_icon.png', cv2.IMREAD_UNCHANGED)
line_img = cv2.imread('line_icon.png', cv2.IMREAD_UNCHANGED)

# Resize images to fit the button area
color_picker_img = cv2.resize(color_picker_img, (60, 60))
clear_img = cv2.resize(clear_img, (60, 60))
save_img = cv2.resize(save_img, (60, 60))
brush_thickness_img = cv2.resize(brush_thickness_img, (60, 60))
circle_img = cv2.resize(circle_img, (60, 60))
rectangle_img = cv2.resize(rectangle_img, (60, 60))
line_img = cv2.resize(line_img, (60, 60))

# Function to detect if a finger is up
def is_finger_up(hand_landmarks, finger_tip_idx, finger_pip_idx):
    return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_pip_idx].y

# Function to overlay images with transparency

def overlay_image(frame, img, pos):
    y1, y2 = pos[1], pos[1] + img.shape[0]
    x1, x2 = pos[0], pos[0] + img.shape[1]

    # Ensure the overlay area fits within the frame dimensions
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return  # Skip overlay if it doesn't fit

    if img.shape[2] == 4:  # If the image has an alpha channel
        alpha_s = img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    else:  # If the image does not have an alpha channel
        frame[y1:y2, x1:x2] = img


# Function to display the brush thickness menu
def show_brush_thickness_menu():
    root = Tk()
    root.withdraw()  # Hide the root window
    brush_thickness = simpledialog.askinteger("Brush Thickness", "Enter thickness (e.g., 1-10):", minvalue=1, maxvalue=10)
    root.destroy()
    return brush_thickness if brush_thickness else 2

# Brush thickness
brush_thickness = 2  # Default thickness

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
                
                # Get the coordinates of the index and middle fingertips
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_cx, index_cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Draw a green circle on the index fingertip
                cv2.circle(frame, (index_cx, index_cy), 10, (0, 255, 0), cv2.FILLED)

                # Check if the index and middle fingers are up
                index_finger_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                middle_finger_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)

                # Update drawing state based on the fingers
                if index_finger_up and middle_finger_up:
                    if not drawing_paused:  # Pause drawing
                        drawing_paused = True
                        points.append(deque(maxlen=1024))  # Start a new line segment
                        index += 1
                elif index_finger_up and not middle_finger_up:
                    if drawing_paused:  # Resume drawing
                        drawing_paused = False
                        points.append(deque(maxlen=1024))  # Start a new line segment
                        index += 1

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
                        elif 385 <= index_cx <= 445:  # Brush Thickness Button
                            brush_thickness = show_brush_thickness_menu()
                        elif 495 <= index_cx <= 555:  # Circle Button
                            current_shape = 'circle'
                        elif 605 <= index_cx <= 665:  # Rectangle Button
                            current_shape = 'rectangle'
                        elif 715 <= index_cx <= 775:  # Line Button
                            current_shape = 'line'
                    else:
                        # Set the start position for drawing shapes
                        if start_pos is None:
                            start_pos = (index_cx, index_cy)
                        else:
                            if current_shape == 'circle':
                                radius = int(np.linalg.norm(np.array(start_pos) - np.array((index_cx, index_cy))))
                                cv2.circle(frame, start_pos, radius, selected_color, brush_thickness)
                                cv2.circle(paintWindow, start_pos, radius, selected_color, brush_thickness)
                            elif current_shape == 'rectangle':
                                cv2.rectangle(frame, start_pos, (index_cx, index_cy), selected_color, brush_thickness)
                                cv2.rectangle(paintWindow, start_pos, (index_cx, index_cy), selected_color, brush_thickness)
                            elif current_shape == 'line':
                                cv2.line(frame, start_pos, (index_cx, index_cy), selected_color, brush_thickness)
                                cv2.line(paintWindow, start_pos, (index_cx, index_cy), selected_color, brush_thickness)
                            
                            # Reset start position after drawing the shape
                            start_pos = None
                            current_shape = None
                        
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

        # Display buttons
        overlay_image(frame, color_picker_img, (40, 5))   # Color Picker Button
        overlay_image(frame, clear_img, (160, 5))         # Clear Button
        overlay_image(frame, save_img, (275, 5))          # Save Button
        overlay_image(frame, brush_thickness_img, (385, 5))  # Brush Thickness Button
        overlay_image(frame, circle_img, (495, 5))        # Circle Button
        overlay_image(frame, rectangle_img, (605, 5))     # Rectangle Button
        overlay_image(frame, line_img, (715, 5))          # Line Button

        # Display the frame
        cv2.imshow('Paint', paintWindow)
        cv2.imshow('Frame', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
