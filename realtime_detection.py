import customtkinter as ctk
import cv2
import numpy as np
import threading
import time
from keras.models import load_model
from PIL import Image
from playsound import playsound

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best_SleepyDriving_CNN.h5"
ALARM_SOUND = "alarm.wav"
IMG_SIZE = 224
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PADDING = 25

# ===============================
# UI CONFIG
# ===============================
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

COLORS = {
    "safe": "#1b5e20",
    "sleepy": "#b71c1c",
    "bg": "#f1f8e9",
    "panel": "#f5f5f5",
    "muted": "#616161"
}

# ===============================
# APP
# ===============================
class DriverMonitoringApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("🚗 Driver Drowsiness Detection System")
        self.geometry("1000x650")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

        # Load model
        self.model = load_model(MODEL_PATH)

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

        # Camera & state
        self.cap = None
        self.running = False
        self.alarm_on = False

        # Drowsiness logic
        self.sleepy_threshold = 0.7
        self.sleepy_start_time = None
        self.sleepy_duration = 3

        self.build_ui()

    # ===============================
    # UI
    # ===============================
    def build_ui(self):

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="🧠 Real-Time Driver Monitoring",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=15)

        # Video area
        self.video_label = ctk.CTkLabel(
            self, width=700, height=500, fg_color=COLORS["bg"]
        )
        self.video_label.grid(row=1, column=0, padx=20, pady=20)

        # Side panel
        self.panel = ctk.CTkFrame(
            self,
            corner_radius=20,
            fg_color=COLORS["panel"]
        )
        self.panel.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        # Start button
        self.start_btn = ctk.CTkButton(
            self.panel,
            text="▶  Start Monitoring",
            command=self.toggle_camera,
            height=45,
            corner_radius=25,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.start_btn.pack(pady=(25, 30), fill="x", padx=25)

        # Status
        self.status_label = ctk.CTkLabel(
            self.panel,
            text="DRIVER STATUS",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["muted"]
        )
        self.status_label.pack()

        self.status_value = ctk.CTkLabel(
            self.panel,
            text="---",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.status_value.pack(pady=(5, 25))

        # Drowsiness score
        self.prob_label = ctk.CTkLabel(
            self.panel,
            text="DROWSINESS LEVEL",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["muted"]
        )
        self.prob_label.pack()

        self.prob_value = ctk.CTkLabel(
            self.panel,
            text="0.0 %",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.prob_value.pack(pady=(5, 15))

        # Progress bar
        self.progress = ctk.CTkProgressBar(
            self.panel,
            height=12,
            corner_radius=10
        )
        self.progress.pack(fill="x", padx=25, pady=(0, 30))
        self.progress.set(0)

        # Info
        self.info = ctk.CTkLabel(
            self.panel,
            text=(
                "System Information\n\n"
                "• Model: CNN (Custom)\n"
                "• Drowsiness Threshold: 0.7\n"
                "• Alarm Trigger: 3 seconds"
            ),
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color="#424242"
        )
        self.info.pack(padx=25, anchor="w")

    # ===============================
    # CAMERA CONTROL
    # ===============================
    def toggle_camera(self):
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.running = True
        self.start_btn.configure(text="⏹  Stop Monitoring")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_camera(self):
        self.running = False
        self.start_btn.configure(text="▶  Start Monitoring")
        if self.cap:
            self.cap.release()
        self.video_label.configure(image=None)
        self.progress.set(0)
        self.status_value.configure(text="---")

    # ===============================
    # ALARM
    # ===============================
    def play_alarm(self):
        if not self.alarm_on:
            self.alarm_on = True
            playsound(ALARM_SOUND)
            self.alarm_on = False

    # ===============================
    # REAL-TIME LOOP
    # ===============================
    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
            )

            # CNN preprocessing
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            sleepy_prob = float(self.model.predict(img, verbose=0).squeeze())
            current_time = time.time()

            # Logic
            if sleepy_prob >= self.sleepy_threshold:
                if self.sleepy_start_time is None:
                    self.sleepy_start_time = current_time
                elapsed = current_time - self.sleepy_start_time
                label = "DROWSY"
                color = (0, 0, 255)
                if elapsed >= self.sleepy_duration:
                    threading.Thread(target=self.play_alarm, daemon=True).start()
            else:
                self.sleepy_start_time = None
                label = "SAFE"
                color = (0, 255, 0)

            # Draw face rectangle (expanded)
            for (x, y, w, h) in faces:
                x1 = max(0, x - PADDING)
                y1 = max(0, y - PADDING)
                x2 = min(frame.shape[1], x + w + PADDING)
                y2 = min(frame.shape[0], y + h + PADDING)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Video text
            cv2.putText(
                frame,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                color,
                3,
                cv2.LINE_AA
            )

            # Update UI
            ui_color = COLORS["sleepy"] if label == "DROWSY" else COLORS["safe"]
            self.status_value.configure(text=label, text_color=ui_color)
            self.prob_value.configure(text=f"{sleepy_prob * 100:.1f} %")
            self.progress.set(sleepy_prob)

            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            ctk_image = ctk.CTkImage(image, image, size=(800, 600))
            self.video_label.configure(image=ctk_image)
            self.video_label.image = ctk_image

            self.update_idletasks()
            self.after(10)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = DriverMonitoringApp()
    app.mainloop()
