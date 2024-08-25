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

# Create the paint window
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load button images
def load_and_resize_image(image_path, size=(60, 60)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cv2.resize(img, size)
    else:
        print(f"Error loading {image_path}")
    return img

color_picker_img = load_and_resize_image('color_picker_icon.png')
clear_img = load_and_resize_image('clear_icon.png')
save_img = load_and_resize_image('save_icon.png')
brush_thickness_img = load_and_resize_image('brush_thickness_icon.png')
circle_img = load_and_resize_image('circle_icon.png')
rectangle_img = load_and_resize_image('rectangle_icon.png')
triangle_img = load_and_resize_image('triangle_icon.png')

# Function to detect if a finger is up
def is_finger_up(hand_landmarks, finger_tip_idx, finger_pip_idx):
    return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_pip_idx].y

# Function to overlay images with transparency
def overlay_image(frame, img, pos):
    img_h, img_w = img.shape[:2]
    y1, y2 = pos[1], pos[1] + img_h
    x1, x2 = pos[0], pos[0] + img_w

    # Ensure the overlay area does not exceed the frame dimensions
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        print("Overlay image exceeds frame dimensions")
        return

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

# Function to draw predefined shapes
def draw_shape(paintWindow, shape, position, size=50, color=(0, 0, 0), thickness=2):
    cx, cy = position
    if shape == 'circle':
        cv2.circle(paintWindow, (cx, cy), size // 2, color, thickness)
    elif shape == 'rectangle':
        cv2.rectangle(paintWindow, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), color, thickness)
    elif shape == 'triangle':
        pts = np.array([[cx, cy - size // 2], [cx - size // 2, cy + size // 2], [cx + size // 2, cy + size // 2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(paintWindow, [pts], isClosed=True, color=color, thickness=thickness)

# Brush thickness
brush_thickness = 2  # Default thickness
selected_shape = None  # No shape selected by default

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
                            selected_shape = 'circle'
                        elif 605 <= index_cx <= 665:  # Rectangle Button
                            selected_shape = 'rectangle'
                        elif 715 <= index_cx <= 775:  # Triangle Button
                            selected_shape = 'triangle'
                    else:
                        if selected_shape:
                            draw_shape(paintWindow, selected_shape, (index_cx, index_cy), color=selected_color)
                            selected_shape = None  # Reset after drawing shape
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
        overlay_image(frame, color_picker_img, (40, 5))   # Color Picker Button
        overlay_image(frame, clear_img, (160, 5))         # Clear Button
        overlay_image(frame, save_img, (275, 5))          # Save Button
        overlay_image(frame, brush_thickness_img, (385, 5))  # Brush Thickness Button
        overlay_image(frame, circle_img, (495, 5))        # Circle Button
        overlay_image(frame, rectangle_img, (605, 5))     # Rectangle Button
        overlay_image(frame, triangle_img, (715, 5))      # Triangle Button

        # Display the paint window and the frame
        cv2.imshow("Paint", paintWindow)
        cv2.imshow("Frame", frame)

        # Exit on pressing the 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
