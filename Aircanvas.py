import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from tkinter import Tk , colorchooser , filedialog , simpledialog

#Initialize mediapipe hands and utils
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

#Setup for canvas
canvas=[deque(maxlen=1024)]
index=0
selected_colour=(255,0,0) #Default colour for drawing : Blue