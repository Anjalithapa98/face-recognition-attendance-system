import cv2
import numpy as np
from PIL import Image
import os
import pickle  
dataset_path = 'dataset'  
haar_cascade_path = 'haarcascade_frontalface_default.xml'

face_detector = cv2.CascadeClassifier(haar_cascade_path)
faces = []
labels = []
label_map = {} 
current_label = 0
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing: {person_name}")

    label_map[current_label] = person_name  

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        gray_image = Image.open(image_path).convert('L') 
        image_np = np.array(gray_image, 'uint8')

        detected_faces = face_detector.detectMultiScale(image_np)
        for (x, y, w, h) in detected_faces:
            faces.append(image_np[y:y+h, x:x+w])
            labels.append(current_label)

    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.save('trainer/trainer.yml')

with open('trainer/label_map.pickle', 'wb') as f:
    pickle.dump(label_map, f)

print(f"\n[INFO] Training complete.")
print(f"Total users trained: {len(label_map)}")
print(f"Labels: {label_map}")
