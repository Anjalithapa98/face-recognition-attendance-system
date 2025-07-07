import cv2
import numpy as np
from PIL import Image
import os
import pickle  # For saving the label map

# ================================
# CONFIG
# ================================
dataset_path = 'dataset'  # Folder with your face data
haar_cascade_path = 'haarcascade_frontalface_default.xml'

# ================================
# Initialize
# ================================
face_detector = cv2.CascadeClassifier(haar_cascade_path)
faces = []
labels = []
label_map = {}  # ID → Name
current_label = 0

# ================================
# Loop through each person's folder
# ================================
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing: {person_name}")

    label_map[current_label] = person_name  # Assign unique ID

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        gray_image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_np = np.array(gray_image, 'uint8')

        detected_faces = face_detector.detectMultiScale(image_np)
        for (x, y, w, h) in detected_faces:
            faces.append(image_np[y:y+h, x:x+w])
            labels.append(current_label)

    current_label += 1

# ================================
# Train recognizer
# ================================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# ================================
# Save model & label map
# ================================
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.save('trainer/trainer.yml')

# Save label map using pickle
with open('trainer/label_map.pickle', 'wb') as f:
    pickle.dump(label_map, f)

# ================================
# Done!
# ================================
print(f"\n[INFO] Training complete.")
print(f"Total users trained: {len(label_map)}")
print(f"Labels: {label_map}")
