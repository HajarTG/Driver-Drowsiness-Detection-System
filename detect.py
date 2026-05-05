"""
Real-time Driver Drowsiness Detection using a webcam.

Loads a trained model (CNN or MobileNetV2) and performs frame-by-frame
inference. An audible alert is played when sustained drowsiness is detected.

Algorithm
---------
1. Capture frame from webcam.
2. Detect face(s) using Haar Cascades.
3. Detect eye(s) within each face ROI.
4. Crop and preprocess each eye image.
5. Pass through the CNN/MobileNetV2 classifier.
6. Maintain a rolling window of predictions — if the mean drowsiness
   score exceeds ``--threshold`` for at least ``--alert_frames`` consecutive
   frames, trigger an alert.
7. Display the annotated frame with bounding boxes and status overlay.

Usage
-----
Run with the default CNN model::

    python detect.py --model_path saved_models/cnn_final.keras

Run with MobileNetV2 (requires 96×96 input)::

    python detect.py --model_path saved_models/mobilenet_final.keras \\
                     --image_size 96

Increase sensitivity (lower threshold)::

    python detect.py --model_path saved_models/cnn_final.keras \\
                     --threshold 0.4 --alert_frames 10

Press **q** to quit the detection window.
"""

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils.face_detector import FaceEyeDetector

# ---------------------------------------------------------------------------
# Optional audio alert via pygame
# ---------------------------------------------------------------------------
try:
    import pygame

    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
except Exception:
    _PYGAME_AVAILABLE = False


ALERT_SOUND_PATH = Path(__file__).parent / "assets" / "alert.wav"


def _play_alert():
    """Play an alert sound if pygame and the audio file are available."""
    if _PYGAME_AVAILABLE and ALERT_SOUND_PATH.exists():
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(str(ALERT_SOUND_PATH))
                pygame.mixer.music.play()
        except Exception:
            pass
    else:
        # Fallback: terminal bell
        print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class DrowsinessDetector:
    """
    Wraps a trained Keras model for real-time drowsiness detection.

    Args:
        model_path (str): Path to a saved Keras model (``model.save()`` format).
        image_size (int): Square input size expected by the model (e.g. 64 or 96).
        threshold (float): Probability threshold for "Drowsy" classification.
        window_size (int): Rolling window of frames used to smooth predictions.
        alert_frames (int): Consecutive frames above threshold before alerting.

    Example:
        >>> detector = DrowsinessDetector("saved_models/cnn_final.keras")
        >>> detector.run()  # Opens the webcam
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 64,
        threshold: float = 0.5,
        window_size: int = 20,
        alert_frames: int = 15,
        camera_index: int = 0,
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.alert_frames = alert_frames
        self.image_size = image_size
        self.camera_index = camera_index

        self._scores = collections.deque(maxlen=window_size)
        self._alert_count = 0
        self._last_alert_time = 0.0
        self._alert_cooldown = 3.0  # seconds between alerts

        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary(print_fn=lambda x: None)  # Silent summary
        print("Model loaded successfully.")

        self.face_detector = FaceEyeDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Open the webcam and run real-time detection until 'q' is pressed."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(
                f"[ERROR] Cannot open camera index {self.camera_index}. "
                "Ensure a webcam is connected and accessible."
            )
            sys.exit(1)

        # Set camera properties for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\nDetection started. Press 'q' to quit.")
        fps_timer = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame. Retrying …")
                continue

            frame_count += 1
            annotated, drowsy, score = self.process_frame(frame)

            # Compute FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()
            else:
                fps = frame_count / max(elapsed, 1e-6)

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Driver Drowsiness Detection", annotated)

            if drowsy:
                now = time.time()
                if now - self._last_alert_time > self._alert_cooldown:
                    _play_alert()
                    self._last_alert_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame and return the annotated frame.

        Args:
            frame (np.ndarray): BGR frame from OpenCV.

        Returns:
            tuple:
                - **annotated** (np.ndarray): Frame with bounding boxes and overlay.
                - **drowsy** (bool): Whether the driver is currently classified
                  as drowsy (based on rolling window).
                - **score** (float): Mean drowsiness score in the rolling window.
        """
        faces, eyes_per_face = self.face_detector.detect(frame)

        frame_score = 0.0
        if faces and any(eyes_per_face):
            eye_scores = []
            for eyes in eyes_per_face:
                for eye_rect in eyes:
                    roi = self.face_detector.extract_eye_roi(
                        frame, eye_rect, target_size=(self.image_size, self.image_size)
                    )
                    if roi is not None:
                        pred = float(self.model.predict(roi, verbose=0)[0][0])
                        eye_scores.append(pred)
            if eye_scores:
                frame_score = float(np.mean(eye_scores))

        self._scores.append(frame_score)
        mean_score = float(np.mean(self._scores)) if self._scores else 0.0

        # Count consecutive high-score frames
        if mean_score >= self.threshold:
            self._alert_count += 1
        else:
            self._alert_count = max(0, self._alert_count - 1)

        drowsy = self._alert_count >= self.alert_frames

        annotated = self.face_detector.draw_detections(
            frame.copy(), faces, eyes_per_face, drowsy=drowsy, score=mean_score
        )
        return annotated, drowsy, mean_score

    def process_image(self, image_path: str):
        """
        Run drowsiness detection on a single image file (for testing).

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: ``(annotated_frame, drowsy, score)``
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        # Reset rolling window for single-image mode
        self._scores.clear()
        self._alert_count = 0
        for _ in range(self.alert_frames):
            annotated, drowsy, score = self.process_frame(frame)
        return annotated, drowsy, score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time driver drowsiness detection."
    )
    parser.add_argument(
        "--model_path",
        default="saved_models/cnn_final.keras",
        help="Path to a saved Keras model (default: saved_models/cnn_final.keras).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Input image size for the model (default: 64).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drowsiness probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Rolling window size for score smoothing (default: 20).",
    )
    parser.add_argument(
        "--alert_frames",
        type=int,
        default=15,
        help="Consecutive drowsy frames before alerting (default: 15).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0 for built-in webcam).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a single image file for offline testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.model_path).exists():
        print(
            f"[ERROR] Model file not found: {args.model_path}\n"
            "Train a model first with:  python train.py --model cnn --data_dir data/"
        )
        sys.exit(1)

    detector = DrowsinessDetector(
        model_path=args.model_path,
        image_size=args.image_size,
        threshold=args.threshold,
        window_size=args.window_size,
        alert_frames=args.alert_frames,
        camera_index=args.camera,
    )

    if args.image:
        print(f"Processing image: {args.image}")
        annotated, drowsy, score = detector.process_image(args.image)
        status = "DROWSY ⚠️" if drowsy else "Awake ✅"
        print(f"Result: {status} (score={score:.3f})")
        cv2.imwrite("detection_result.jpg", annotated)
        print("Annotated image saved to: detection_result.jpg")
    else:
        detector.run()


if __name__ == "__main__":
    main()
