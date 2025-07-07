import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import csv

# ===================================
# Load trained model
# ===================================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load label map
with open('trainer/label_map.pickle', 'rb') as f:
    label_map = pickle.load(f)

# Make ordered list: index = ID, value = name
# Ensure IDs are sorted so index matches!
names = [label_map[i] for i in sorted(label_map.keys())]
print(f"[INFO] Loaded labels (ID -> Name):")
for i, name in enumerate(names):
    print(f"  {i}: {name}")

# ===================================
# Attendance CSV setup
# ===================================
if not os.path.exists('attendance'):
    os.makedirs('attendance')

def mark_attendance(name):
    filename = "attendance/attendance.csv"
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time'])

    with open(filename, 'r') as f:
        if f"{name}" in f.read():
            return

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])
        print(f"[LOGGED] {name} at {timestamp}")

# ===================================
# Start webcam
# ===================================
cam = cv2.VideoCapture(0)
print("\n[INFO] Starting Face Recognition. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        print(f"[DEBUG] Predicted ID: {id}, Confidence: {confidence:.2f}")

        if confidence < 80 and id < len(names):
            name = names[id]
            mark_attendance(name)
            label = f"{name} ({round(100 - confidence)}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

