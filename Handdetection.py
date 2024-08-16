import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of the index fingertip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert to pixel coordinates
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a circle at the fingertip
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                
                # Use the fingertip coordinates to draw on the air canvas or any other application logic
                
        # Display the frame
        cv2.imshow('Air Canvas', frame)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
