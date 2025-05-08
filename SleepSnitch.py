import cv2
import mediapipe as mp
import math
import pygame
import os

# Use pygame only if audio file exists
beep_sound_path = "beep.wav"

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_EAR(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth_points):
    A = euclidean_distance(mouth_points[13], mouth_points[14])  # vertical
    B = euclidean_distance(mouth_points[78], mouth_points[308])  # horizontal
    return A / B

def get_head_tilt_angle(landmarks, w, h):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = (right_eye.x - left_eye.x) * w
    dy = (right_eye.y - left_eye.y) * h
    return math.degrees(math.atan2(dy, dx))

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

# Thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.75
TILT_ANGLE_THRESHOLD = 30
CLOSED_FRAMES_THRESHOLD = 60
YAWN_FRAMES_THRESHOLD = 15

# States
closed_frame_count = 0
yawn_frame_count = 0
buzzer_played = False

# Initialize pygame mixer only if beep file exists
if os.path.exists(beep_sound_path):
    pygame.mixer.init()
    pygame.mixer.music.load(beep_sound_path)

# Webcam
cap = cv2.VideoCapture(0)  # Pi Camera or USB

# Reduce resolution for better FPS on Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w = frame.shape[:2]

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        mouth_points = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH}
        mar = calculate_MAR(mouth_points)
        angle = get_head_tilt_angle(landmarks, w, h)

        # EAR check
        if avg_ear < EAR_THRESHOLD:
            closed_frame_count += 1
        else:
            closed_frame_count = 0
            buzzer_played = False

        # MAR check
        if mar > MAR_THRESHOLD:
            yawn_frame_count += 1
        else:
            yawn_frame_count = 0

        # Display info
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Tilt: {angle:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)

        # Alert
        if closed_frame_count >= CLOSED_FRAMES_THRESHOLD or yawn_frame_count >= YAWN_FRAMES_THRESHOLD or abs(angle) > TILT_ANGLE_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if not buzzer_played and os.path.exists(beep_sound_path):
                try:
                    pygame.mixer.music.play()
                except:
                    print("Failed to play alert.")
                buzzer_played = True

        # Draw landmarks
        for (x, y) in left_eye + right_eye + list(mouth_points.values()):
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

    cv2.imshow("Fatigue Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()