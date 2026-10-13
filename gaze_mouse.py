"""
Professional AI Head-Tracking & Gaze-Controlled Mouse (Modern MediaPipe API Edition)
Author: AI Collaborator
Description: Controls OS cursor movement using face orientation / nose position 
             via MediaPipe Face Landmarker Tasks API and One-Euro Filtering.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui

# --- CONFIGURATION CONSTANTS ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"
CAMERA_INDEX = 0
CLICK_COOLDOWN = 0.5       # Cooldown to prevent accidental multiple clicks
BLINK_THRESHOLD = 0.018    # Vertical eye aspect ratio threshold for blink-to-click

# --- ONE-EURO FILTER FOR BUTTERY-SMOOTH MOTION ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t, cutoff):
        r = 2 * np.pi * cutoff * (t - self.t_prev)
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        
        a_d = self.smoothing_factor(t, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def download_model_if_missing(path, url):
    if not os.path.exists(path):
        print("[INFO] Downloading MediaPipe Face Landmarker model...")
        try:
            urllib.request.urlretrieve(url, path)
            print("[INFO] Model downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            exit(1)

def main():
    pyautogui.FAILSAFE = False
    download_model_if_missing(MODEL_PATH, MODEL_URL)

    # Initialize MediaPipe Face Landmarker Detector in VIDEO mode
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    init_time = time.time()
    filter_x = OneEuroFilter(init_time, screen_w / 2, min_cutoff=0.8, beta=0.02)
    filter_y = OneEuroFilter(init_time, screen_h / 2, min_cutoff=0.8, beta=0.02)

    last_action_time = 0

    print("[INFO] AI Head-Tracking Mouse is running. Press 'q' in video window to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = time.time()
        frame_timestamp_ms = int((current_time - init_time) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        action_text = "HEAD TRACKING"
        action_color = (255, 255, 255)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # Use nose tip (landmark index 1) as head tracking anchor point
            nose = landmarks[1]
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)

            cv2.circle(frame, (nose_x, nose_y), 6, (0, 255, 0), cv2.FILLED)

            # Map head movement to monitor resolution bounds
            margin_x, margin_y = 130, 90
            mapped_x = np.interp(nose_x, (margin_x, w - margin_x), (0, screen_w))
            mapped_y = np.interp(nose_y, (margin_y, h - margin_y), (0, screen_h))

            # Apply One-Euro filter for smooth motion
            smooth_x = filter_x(current_time, mapped_x)
            smooth_y = filter_y(current_time, mapped_y)

            pyautogui.moveTo(smooth_x, smooth_y)

            # Blink Detection via eyelid landmarks (Upper eyelid 159, Lower eyelid 145)
            upper_lid = landmarks[159]
            lower_lid = landmarks[145]
            eye_vertical_dist = abs(upper_lid.y - lower_lid.y)

            if eye_vertical_dist < BLINK_THRESHOLD:
                action_text = "BLINK CLICK"
                action_color = (0, 255, 0)
                if current_time - last_action_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_action_time = current_time

        cv2.putText(frame, f"Action: {action_text}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        cv2.putText(frame, "Tilt head smoothly to navigate cursor", (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("AI Head & Gaze Tracking Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()