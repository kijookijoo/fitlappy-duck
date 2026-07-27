"""
Phase 1: Pose Landmark Pipeline
Goal: Get MediaPipe body landmarks rendering live on webcam feed.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize MediaPipe PoseLandmarker
model_path = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run pose detection
    detection_result = detector.detect_for_video(mp_image, frame_count)
    frame_count += 1
    
    # Draw landmarks on frame
    annotated_frame = frame.copy()
    
    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            # Draw each landmark as a dot
            for landmark in pose_landmarks:
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw circle at landmark position
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw skeleton connections (optional but helpful for visualization)
            # MediaPipe pose has 33 landmarks, draw lines between connected joints
            connections = [
                # Head
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                # Body
                (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                # Left arm
                (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),
                # Right arm
                (12, 24), (24, 26), (26, 28), (28, 30), (30, 32),
                # Left leg
                (23, 11), (25, 23), (27, 25), (29, 27), (31, 29),
                # Right leg
                (24, 12), (26, 24), (28, 26), (30, 28), (32, 30),
            ]
            
            for start, end in connections:
                if start < len(pose_landmarks) and end < len(pose_landmarks):
                    start_pos = pose_landmarks[start]
                    end_pos = pose_landmarks[end]
                    
                    h, w = frame.shape[:2]
                    x1, y1 = int(start_pos.x * w), int(start_pos.y * h)
                    x2, y2 = int(end_pos.x * w), int(end_pos.y * h)
                    
                    # Only draw if confidence is high enough
                    if start_pos.presence > 0.5 and end_pos.presence > 0.5:
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add FPS counter
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow("Pose Landmarks - Phase 1", annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Phase 1 complete!")
