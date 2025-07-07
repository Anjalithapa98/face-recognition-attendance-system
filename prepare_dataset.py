import cv2
import os
import numpy as np

SOURCE_DIR = "Raw_Photo"
TARGET_DIR = "dataset"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def augment_face(face):
    augmented = []

    # Original
    augmented.append(face)

    # Flip
    augmented.append(cv2.flip(face, 1))

    # Rotate small angles
    for angle in [-10, 10, -20, 20]:
        h, w = face.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(face, M, (w, h))
        augmented.append(rotated)

    # Adjust brightness
    for alpha in [0.8, 1.2, 1.4]:
        bright = cv2.convertScaleAbs(face, alpha=alpha, beta=0)
        augmented.append(bright)

    # Slight shifts
    for tx in [-10, 10]:
        M_shift = np.float32([[1, 0, tx], [0, 1, 0]])
        shifted = cv2.warpAffine(face, M_shift, (w, h))
        augmented.append(shifted)

    return augmented

for person in os.listdir(SOURCE_DIR):
    src_folder = os.path.join(SOURCE_DIR, person)
    dst_folder = os.path.join(TARGET_DIR, person)
    os.makedirs(dst_folder, exist_ok=True)

    count = 0

    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (200, 200))

            augmented_faces = augment_face(face_crop)

            for face_aug in augmented_faces:
                count += 1
                save_path = os.path.join(dst_folder, f"{count}.jpg")
                cv2.imwrite(save_path, face_aug)

                if count >= 50:
                    break

        if count >= 50:
            break

    print(f"[INFO] Created {count} images for {person}")
