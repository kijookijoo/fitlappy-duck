"""
Real-time push-up counter using MediaPipe Pose.

Features
--------
- Counts only full-range reps (elbows below a depth threshold, then full lockout).
- Ignores partial reps via confirmed down + up phases with debounce.
- Skeleton overlay using pose connections for easy visual alignment.

Usage
-----
python3 pushup.py                # default camera 0
python3 pushup.py --camera 1     # alternate camera
python3 pushup.py --cpu          # force CPU if GL/GPU fails

Notes
-----
- Assumes the athlete faces the camera on the floor; it relies mainly on the shoulder–elbow angle.
- Tune `--down-angle` and `--up-angle` if your camera angle differs.
"""

import argparse
import os
import time
from collections import deque
import threading
import queue

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from ws_bridge import WSBroadcaster
except ImportError:
    WSBroadcaster = None

# Pre-set env vars to avoid GPU / font cache issues on macOS/headless runs
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")  # force CPU if GL context fails
os.environ.setdefault("MEDIAPIPE_USE_GPU", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe INFO/WARN spam

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

POSE_MODEL_ASSET = "pose_landmarker.task"


def parse_args():
    parser = argparse.ArgumentParser(description="Live push-up counter")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for MediaPipe Tasks backend")
    parser.add_argument("--down-angle", type=float, default=95.0,
                        help="Elbow angle (deg) considered the bottom of a rep (default: 95)")
    parser.add_argument("--up-angle", type=float, default=125.0,
                        help="Elbow angle (deg) considered full lockout (default: 125)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Smoothing window for elbow angle in frames (default: 3)")
    parser.add_argument("--confirm-down", type=int, default=2,
                        help="Frames required below down-angle to register bottom (default: 2)")
    parser.add_argument("--confirm-up", type=int, default=2,
                        help="Frames required above up-angle to register lockout (default: 2)")
    parser.add_argument("--no-window", action="store_true",
                        help="Do not open OpenCV display window (reduces extra popups)")
    parser.add_argument("--landmark-every", type=int, default=2,
                        help="Send landmarks to WS every N frames to reduce bandwidth/latency (default: 2)")
    return parser.parse_args()


def init_pose_detector(force_cpu: bool = False):
    """Create a pose detector. Prefer Tasks API; fall back to Solutions."""
    try:
        delegate = (
            python.BaseOptions.Delegate.CPU
            if force_cpu
            else python.BaseOptions.Delegate.GPU
        )
        base_options = python.BaseOptions(
            model_asset_path=POSE_MODEL_ASSET,
            delegate=delegate,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        return detector, "tasks"
    except Exception as e:  # pragma: no cover - runtime guard
        print(f"⚠️  Tasks PoseLandmarker failed: {e}")
        print("↪️  Falling back to MediaPipe Solutions (CPU-only).")

    if hasattr(mp, "solutions"):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return pose, "solutions"

    raise RuntimeError("No available pose backend. Install `mediapipe==0.10.9`.")


def angle_3pts(a, b, c):
    """Compute angle ABC (in degrees) with B as vertex."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def choose_side(landmarks, use_tasks: bool):
    """
    Pick left or right arm based on presence/visibility scores.
    Returns ("left"|"right", index offsets).
    """
    if use_tasks:
        def vis(idx): return landmarks[idx].presence
    else:
        def vis(idx): return landmarks[idx].visibility

    left_score = sum(vis(i) for i in [11, 13, 15, 23, 25, 27])
    right_score = sum(vis(i) for i in [12, 14, 16, 24, 26, 28])
    return ("left", [11, 13, 15, 23, 25, 27]) if left_score >= right_score else ("right", [12, 14, 16, 24, 26, 28])


def extract_landmarks(result, backend: str):
    """Return list of 33 landmarks for the chosen pose (tasks or solutions)."""
    if backend == "tasks":
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]
    else:
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark


def get_point_xy(lm, w, h):
    return (lm.x * w, lm.y * h)


def main():
    args = parse_args()

    # Audio feedback disabled
    speech_queue: queue.Queue[str | None] = queue.Queue()
    engine = None

    ws = None
    ws_port = 9001
    if WSBroadcaster:
        try:
            ws = WSBroadcaster(host="0.0.0.0", port=ws_port)
            ws.start()
            print(f"🌐 WS broadcaster running on ws://127.0.0.1:{ws_port}")
        except Exception as e:
            print(f"⚠️  Failed to start WS broadcaster: {e}")
            ws = None
    else:
        print("ℹ️  websockets/WSBroadcaster not available; Phaser client won't receive events.")

    def speech_worker():
        return

    speaker_thread = None
    if engine:
        speaker_thread = threading.Thread(target=speech_worker, daemon=True)
        speaker_thread.start()
    else:
        print("ℹ️  speech disabled.")

    def speak(text: str):
        return

    detector, backend = init_pose_detector(force_cpu=args.cpu)
    print(f"✅ Pose backend: {backend}")
    use_tasks = backend == "tasks"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # request higher FPS to catch fast reps

    frame_idx = 0
    reps = 0
    state = "up"  # start in lockout
    reached_bottom = False
    elbow_hist = deque(maxlen=max(1, args.smooth))
    down_frames = 0
    up_frames = 0
    fps = 0.0
    fps_ts = time.time()
    locked_side = None
    lost_frames = 0
    game_over = False

    # Skeleton connections (MediaPipe Pose indices)
    skeleton_edges = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 12),            # shoulders
        (11, 23), (12, 24),  # torso top
        (23, 24),            # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
        (27, 31), (28, 32),  # ankles to feet
        (15, 17), (16, 18),  # wrists to hands
        (15, 19), (16, 20),
    ]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if backend == "tasks":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection_result = detector.detect_for_video(mp_image, frame_idx)
                frame_idx += 1
            else:
                detection_result = detector.process(rgb)

            pose_landmarks = extract_landmarks(detection_result, backend)
            overlay = frame.copy()
            h, w = frame.shape[:2]

            elbow_angle = None
            posture_ok = False

            if pose_landmarks and not game_over:
                side_guess, idxs = choose_side(pose_landmarks, use_tasks)

                # Lock to one side to avoid flip-flop; relock only after several missing frames
                if locked_side is None or lost_frames > 10:
                    locked_side = side_guess
                side = locked_side

                # Pick indices based on locked side
                idxs = [11, 13, 15, 23, 25, 27] if side == "left" else [12, 14, 16, 24, 26, 28]

                shoulder_idx, elbow_idx, wrist_idx, hip_idx, knee_idx, ankle_idx = idxs

                shoulder = get_point_xy(pose_landmarks[shoulder_idx], w, h)
                elbow = get_point_xy(pose_landmarks[elbow_idx], w, h)
                wrist = get_point_xy(pose_landmarks[wrist_idx], w, h)
                elbow_angle = angle_3pts(shoulder, elbow, wrist)

                elbow_hist.append(elbow_angle)
                smooth_elbow = sum(elbow_hist) / len(elbow_hist)

                posture_ok = True  # if landmarks are valid we proceed

                # Down confirmation
                if smooth_elbow <= args.down_angle:
                    down_frames += 1
                else:
                    down_frames = 0

                if smooth_elbow >= args.up_angle:
                    up_frames += 1
                else:
                    up_frames = 0

                if state == "up" and down_frames >= args.confirm_down:
                    state = "down"
                    reached_bottom = True
                    up_frames = 0  # reset top counter
                    speak("down")
                    if ws:
                        ws.broadcast({"event": "down"})

                elif state == "down" and reached_bottom and up_frames >= args.confirm_up:
                    reps += 1
                    state = "up"
                    reached_bottom = False
                    down_frames = 0
                    speak(f"up {reps}")
                    if ws:
                        ws.broadcast({"event": "up", "count": reps})
            else:
                elbow_hist.clear()
                reached_bottom = False
                down_frames = 0
                up_frames = 0
                lost_frames += 1

            if pose_landmarks:
                # Draw skeleton lines
                for start, end in skeleton_edges:
                    if start < len(pose_landmarks) and end < len(pose_landmarks):
                        ls, le = pose_landmarks[start], pose_landmarks[end]
                        if (use_tasks and ls.presence > 0.3 and le.presence > 0.3) or \
                           (not use_tasks and ls.visibility > 0.3 and le.visibility > 0.3):
                            x1, y1 = int(ls.x * w), int(ls.y * h)
                            x2, y2 = int(le.x * w), int(le.y * h)
                            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Debug text
                cv2.putText(overlay, f"Side: {side}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
                cv2.putText(overlay, f"Elbow: {smooth_elbow:.1f} deg", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Broadcast landmarks to client (normalized coords)
                if ws and frame_idx % max(1, args.landmark_every) == 0:
                    lm_payload = [
                        {"x": round(float(lm.x), 4), "y": round(float(lm.y), 4),
                         "z": round(float(lm.z), 4),
                         "visibility": round(float(getattr(lm, "visibility", lm.presence if hasattr(lm, "presence") else 0.0)), 4)}
                        for lm in pose_landmarks
                    ]
                    ws.broadcast({"event": "landmarks", "landmarks": lm_payload})

                lost_frames = 0  # reset missing counter when landmarks seen

            # Rep/state overlay
            cv2.putText(overlay, f"Push-ups: {reps}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(overlay, f"State: {state}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            form_color = (0, 255, 0) if posture_ok else (0, 0, 255)
            cv2.putText(overlay, "Tracking" if posture_ok else "No pose detected",
                        (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, form_color, 2)

            # FPS
            dt = time.time() - fps_ts
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            fps_ts = time.time()
            cv2.putText(overlay, f"FPS: {fps:.1f}", (overlay.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not args.no_window:
                cv2.imshow("Push-up Counter", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if engine:
            speech_queue.put(None)
            speaker_thread.join(timeout=1.0)
        if ws:
            ws.stop()


if __name__ == "__main__":
    main()
