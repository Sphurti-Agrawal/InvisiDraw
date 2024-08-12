import numpy as np
import cv2
from collections import deque

# Default callback function for trackbar
def setValues(x):
    pass

# Create a window for color detection and RGB palette
cv2.namedWindow("Color detectors")
cv2.namedWindow("RGB Palette")

# Create trackbars for adjusting the HSV range for color detection
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValues)

# Create trackbars for RGB color selection
cv2.createTrackbar("Red", "RGB Palette", 255, 255, setValues)
cv2.createTrackbar("Green", "RGB Palette", 0, 255, setValues)
cv2.createTrackbar("Blue", "RGB Palette", 0, 255, setValues)

# Points deque for different colors
points = [deque(maxlen=1024)]

# Index to mark position of pointers in color array
index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Set up the canvas
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Load the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])

    # Get RGB values from trackbars
    r = cv2.getTrackbarPos("Red", "RGB Palette")
    g = cv2.getTrackbarPos("Green", "RGB Palette")
    b = cv2.getTrackbarPos("Blue", "RGB Palette")
    color = (b, g, r)

    # Adding the color palette to the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add instructions for the user
    instructions = [
        "Instructions:",
        "1. Move the colored object in front of the camera.",
        "2. Use the trackbars to set the color detection range.",
        "3. Draw on the screen by moving the object.",
        "4. Select drawing color from the RGB Palette.",
        "5. Click 'CLEAR ALL' to erase the canvas.",
        "6. Press 'q' to quit the application."
    ]

    y0, dy = 470, 20
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(paintWindow, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Mask for the detected color
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    # Find contours in the mask
    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear All
                points = [deque(maxlen=512)]
                index = 0
                paintWindow[67:, :, :] = 255
        else:
            points[index].appendleft(center)
    else:
        points.append(deque(maxlen=512))
        index += 1

    # Draw lines of all colors on the canvas and frame
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], color, 2)
            cv2.line(paintWindow, points[j][k - 1], points[j][k], color, 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("mask", Mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

